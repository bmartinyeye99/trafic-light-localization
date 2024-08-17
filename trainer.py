import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torchmetrics import R2Score
from torchvision.ops import box_iou
import wandb


def adjust_image_for_plotting(image):
    image = image.squeeze(0)
    # Move the channel to the last dimension if necessary
    if image.shape[0] == 3:  # Assuming 3 channels (RGB)
        image = image.permute(1, 2, 0)

    # Convert to numpy if it's a tensor
    if not isinstance(image, np.ndarray):
        image = image.cpu().numpy()

    # Normalize image data to 0-1 for plotting, if necessary
    if image.min() < 0 or image.max() > 1:
        image = (image - image.min()) / (image.max() - image.min())

    return image


def draw_boxes(image, pred_box, true_box):
    _fig, ax = plt.subplots(1)
    ax.imshow(adjust_image_for_plotting(image))

    box = pred_box[0]

    # Draw predicted boxes in green
    # Normalize box coordinates
    x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
    x0 = ((x0 + 1) / 2) * 682
    y0 = ((y0 + 1) / 2) * 512
    x1 = ((x1 + 1) / 2) * 682
    y1 = ((y1 + 1) / 2) * 512
    # Draw rectangle
    rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                             linewidth=1, edgecolor='green', facecolor='none')
    ax.add_patch(rect)

    tbox = true_box

    # Normalize box coordinates
    x0, y0, x1, y1 = tbox[0], tbox[1], tbox[2], tbox[3]
    x0 = ((x0 + 1) / 2) * 682
    y0 = ((y0 + 1) / 2) * 512
    x1 = ((x1 + 1) / 2) * 682
    y1 = ((y1 + 1) / 2) * 512

    rect1 = patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0, linewidth=1, edgecolor='red', facecolor='none')
    ax.add_patch(rect1)

    plt.show()


def mixup_loss(criterion, pred, y_a, y_b, alpha=0.7):
    lam = torch.tensor(np.random.beta(alpha, alpha))
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def de_normalize_boxes(boxes):
    # Convert from [-1, 1] to [0, 1]
    boxes = (boxes + 1) / 2.0
    # Convert from [0, 1] to [0, width] or [0, height]
    boxes[:, [0, 2]] *= 682  # Adjust x coordinates
    boxes[:, [1, 3]] *= 512  # Adjust y coordinates
    return boxes


class Trainer:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device, memory_format=torch.channels_last)

        # Create the optimizer
        self.optimizer = torch.optim.NAdam(
            model.parameters(), lr=self.cfg.learning_rate)

        self.localization_loss_fn = nn.SmoothL1Loss()

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.max_epochs, eta_min=0)

        self.scaler = GradScaler()

        self.r2_score = R2Score().to(self.device)  # Initialize the R2Score metric

    def setup(self, datamodule):
        # Setup data
        self.datamodule = datamodule
        self.datamodule.setup(self.cfg)

    def fit(self):
        with wandb.init(
            project='Traffic light localization',
            config=self.cfg,
        ) as run:
            for epoch in range(self.cfg.max_epochs):
                print('Epoch ', epoch)

                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Learning Rate: {current_lr}")

                self.train_epoch(self.model,
                                 self.datamodule.dataloader_train, run, epoch)

                self.validate_epoch(self.model,
                                    self.datamodule.dataloader_valid, run, epoch)

                print('-------------------------------------------\n\n')

                self.scheduler.step()  # Update learning rate

        self.test(self.model, self.datamodule.dataloader_test)

        print('\n\n')
        print('-----------------------------------------------')
        self.draw(self.model, self.datamodule.dataloader_draw)

    def train_epoch(self, model, dataloader, run, epoch):
        model.train()
        running_loss = 0.0
        total = 0

        with tqdm(dataloader, desc=f"Training") as progress:
            for x, y, z in progress:
                x = x.to(self.device, memory_format=torch.channels_last)
                y = y.to(self.device)
                z = z.to(self.device)

                self.optimizer.zero_grad()

                with autocast():

                    # Forward pass (bbox_regressions returned by your model)
                    bbox_regressions = self.model(x)

                    if torch.equal(torch.tensor([-5, -5, -5, -5]).to(self.device), z):
                        # Calculate mixed-up loss
                        loss = mixup_loss(self.localization_loss_fn,
                                          bbox_regressions, y, z)

                    else:
                        # Calculate regression loss
                        loss = self.localization_loss_fn(bbox_regressions, y)

                    # Backward pass and optimize
                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.optimizer)

                clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                self.scaler.step(self.optimizer)

                running_loss += loss.item() * x.size(0)
                total += x.size(0)
                avg_loss = running_loss / total  # Calculate average loss
                progress.set_postfix(avg_loss=avg_loss)
                self.scaler.update()

        run.log({"Epoch Train Loss": avg_loss}, step=epoch)

    def validate_epoch(self, model, dataloader, run, epoch):
        model.eval()  # Set model to evaluation mode
        running_loss = 0.0
        total = 0

        preds_accumulated = []  # List to store de normalized predictions
        targets_accumulated = []  # List to store de normalized ground truths

        # Reset the metric for next epoch's use
        self.r2_score.reset()

        with torch.no_grad():  # No need to compute gradients
            with tqdm(dataloader, desc=f"Validation") as progress:
                for x, y, _z in progress:
                    x = x.to(self.device, memory_format=torch.channels_last)
                    y = y.to(self.device)

                    # Forward pass only
                    with autocast():
                        bbox_regressions = model(x)

                        # Flatten predictions
                        preds = bbox_regressions.view(-1)
                        targets = y.view(-1)  # Flatten targets
                        self.r2_score.update(preds, targets)

                        # Accumulate denormalized predictions and ground truths
                        tr_pred = preds.view(-1, 4)
                        preds_accumulated.append(de_normalize_boxes(tr_pred))
                        targets_accumulated.append(de_normalize_boxes(y))

                        loss = self.localization_loss_fn(bbox_regressions, y)

                    running_loss += loss.item() * x.size(0)
                    total += x.size(0)
                    avg_loss = running_loss / total  # Calculate average loss
                    progress.set_postfix(avg_loss=avg_loss)

        run.log({"Epoch Validation Loss": avg_loss}, step=epoch)

        # Concatenate all accumulated predictions and targets
        all_preds = torch.cat(preds_accumulated, dim=0)
        all_targets = torch.cat(targets_accumulated, dim=0)

        # Compute IoU scores using all predictions and targets
        all_ious = box_iou(all_preds, all_targets)
        avg_iou = all_ious.diagonal().mean().item()  # Compute average IoU

        final_r2 = self.r2_score.compute()

        print(f"Validation R2 Score: {final_r2.item()}")
        print(f"Validation Average IoU: {avg_iou}")

        run.log({"Validation R2 Score": final_r2.item()}, step=epoch)
        run.log({"Validation Average IoU": avg_iou}, step=epoch)

    def test(self, model, dataloader):
        model.eval()  # Set model to evaluation mode
        running_loss = 0.0
        total = 0

        preds_accumulated = []  # List to store de normalized predictions
        targets_accumulated = []  # List to store de normalized ground truths

        # Reset the metric for next epoch's use
        self.r2_score.reset()

        with torch.no_grad():  # No need to compute gradients
            with tqdm(dataloader, desc=f"Testing") as progress:
                for x, y, _z in progress:
                    x = x.to(self.device, memory_format=torch.channels_last)
                    y = y.to(self.device)

                    # Forward pass only
                    with autocast():
                        bbox_regressions = model(x)

                        # Flatten predictions
                        preds = bbox_regressions.view(-1)
                        targets = y.view(-1)  # Flatten targets
                        self.r2_score.update(preds, targets)

                        # Accumulate denormalized predictions and ground truths
                        tr_pred = preds.view(-1, 4)
                        preds_accumulated.append(de_normalize_boxes(tr_pred))
                        targets_accumulated.append(de_normalize_boxes(y))

                        loss = self.localization_loss_fn(bbox_regressions, y)

                    running_loss += loss.item() * x.size(0)
                    total += x.size(0)
                    avg_loss = running_loss / total  # Calculate average loss
                    progress.set_postfix(avg_loss=avg_loss)

        # Concatenate all accumulated predictions and targets
        all_preds = torch.cat(preds_accumulated, dim=0)
        all_targets = torch.cat(targets_accumulated, dim=0)

        # Compute IoU scores using all predictions and targets
        all_ious = box_iou(all_preds, all_targets)
        avg_iou = all_ious.diagonal().mean().item()  # Compute average IoU

        final_r2 = self.r2_score.compute()

        print(f"Testing R2 Score: {final_r2.item()}")
        print(f"Testing Average IoU: {avg_iou}")

    def draw(self, model, dataloader):
        print('Showing predictions on images!')
        print('Red is ground truth and green is predicted value!')

        model.eval()  # Set model to evaluation mode

        preds_accumulated = []  # List to store de normalized predictions
        targets_accumulated = []  # List to store de normalized ground truths

        # Reset the metric for next epoch's use
        self.r2_score.reset()
        with torch.no_grad():  # No need to compute gradients
            for x, y, _z in dataloader:
                x = x.to(self.device, memory_format=torch.channels_last)
                y = y.to(self.device)

                # Forward pass only
                with autocast():
                    preds = model(x).view(-1)  # Flatten predictions
                    targets = y.view(-1)  # Flatten targets

                    self.r2_score.update(preds, targets)

                    # Accumulate denormalized predictions and ground truths
                    tr_pred = preds.view(-1, 4)
                    preds_accumulated.append(de_normalize_boxes(tr_pred))
                    targets_accumulated.append(de_normalize_boxes(y))

                    pred_boxes = tr_pred.cpu()  # This is your first set of predicted boxes
                    true_boxes = targets.cpu()

                    draw_boxes(x.cpu(), pred_boxes.numpy(),
                               true_boxes.numpy())

        # Concatenate all accumulated predictions and targets
        all_preds = torch.cat(preds_accumulated, dim=0)
        all_targets = torch.cat(targets_accumulated, dim=0)

        # Compute IoU scores using all predictions and targets
        all_ious = box_iou(all_preds, all_targets)
        avg_iou = all_ious.diagonal().mean().item()  # Compute average IoU

        final_r2 = self.r2_score.compute()

        print(f"Validation R2 Score: {final_r2.item()}")
        print(f"Validation Average IoU: {avg_iou}")
