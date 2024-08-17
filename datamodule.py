import torch
import torchvision.transforms as TF


from dataset import NumpyToTensor, create_traffic_light_dataset


class DataModule:
    def __init__(self):

        self.input_transform = TF.Compose([
            NumpyToTensor(),
            TF.Resize(512),  # Resize the shorter side to 512 pixels
        ])

        self.dataset_train, self.dataset_val, self.dataset_test, self.dataset_draw = create_traffic_light_dataset(
            self.input_transform)

    def setup(self, cfg):
        self.dataloader_train = torch.utils.data.dataloader.DataLoader(
            self.dataset_train,
            batch_size=cfg.batch_size,    # Batch size hyper-parameter
            shuffle=True,                 # Iterate over samples in random order
            num_workers=cfg.num_workers   # Parallel processing of input samples
        )

        self.dataloader_valid = torch.utils.data.dataloader.DataLoader(
            self.dataset_val,
            batch_size=cfg.batch_size,    # Batch size hyper-parameter
            shuffle=False,                # Iterate over samples in random order
            num_workers=cfg.num_workers   # Parallel processing of input samples
        )

        self.dataloader_test = torch.utils.data.dataloader.DataLoader(
            self.dataset_test,
            batch_size=cfg.batch_size,    # Batch size hyper-parameter
            shuffle=False,                # Iterate over samples in random order
            num_workers=cfg.num_workers   # Parallel processing of input samples
        )

        self.dataloader_draw = torch.utils.data.dataloader.DataLoader(
            self.dataset_draw,
            batch_size=1,                 # Batch size hyper-parameter
            shuffle=True,                 # Iterate over samples in random order
            num_workers=cfg.num_workers   # Parallel processing of input samples
        )
