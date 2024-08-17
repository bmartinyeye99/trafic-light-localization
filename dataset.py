import os
from typing import Callable
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import imgaug as ia
import imgaug.augmenters as iaa


class NumpyToTensor(Callable):
    def __call__(self, x):
        x = np.transpose(x, axes=(2, 0, 1))       # HWC -> CHW
        x = torch.from_numpy(x) / 255.0         # <0;255>UINT8 -> <0;1>
        return x.float()                        # cast as 32-bit flow


def blend_images(image_path1, image_path2, alpha=0.7):
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Blend the images
    blended = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
    return blended


def augment_data(df):
    print('Augmenting data....')
    root = '.scratch/data/augmentations'

    if not os.path.exists(root):
        os.makedirs(root)

    root = root + '/'

    augmented_df = []

    df2 = df.sample(frac=1, random_state=42).reset_index(drop=True)

    for row, row2 in zip(df.itertuples(index=True, name='Pandas'), df2.itertuples(index=True, name='Pandas')):
        # Load the image
        file_name = row.file
        filepath = row.filepath
        file_name2 = row2.file
        filepath2 = row2.filepath

        image = cv2.imread(filepath)

        x1, y1, x2, y2 = row.x1, row.y1, row.x2, row.y2
        x1_2, y1_2, x2_2, y2_2 = row2.x1, row2.y1, row2.x2, row2.y2

        # Define the bounding box
        bbs = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        ], shape=image.shape)

        # Augmenters
        flip_augmenter = iaa.Fliplr(1.0)  # Always apply flip
        rotate_augmenter = iaa.Affine(rotate=10)  # Rotate by 10 degrees

        flipped = root + 'flipped_' + file_name
        # Apply flip
        image_flipped, bbs_flipped = flip_augmenter(
            image=image, bounding_boxes=bbs)

        if not os.path.exists(flipped):
            cv2.imwrite(flipped, image_flipped)

        rotated = root + 'rotated_' + file_name
        # Apply rotation
        image_rotated, bbs_rotated = rotate_augmenter(
            image=image, bounding_boxes=bbs)

        if not os.path.exists(rotated):
            cv2.imwrite(rotated, image_rotated)

        rotated_flipped = root + 'rotated_flipped_' + file_name
        # Apply rotation and then flip
        image_rotated_flipped, bbs_rotated_flipped = flip_augmenter(
            image=image_rotated, bounding_boxes=bbs_rotated)

        if not os.path.exists(rotated_flipped):
            cv2.imwrite(rotated_flipped, image_rotated_flipped)

        blend_image = root + 'blended_' + \
            file_name.replace(".jpg", "") + '_' + file_name2

        if not os.path.exists(blend_image):
            cv2.imwrite(blend_image, blend_images(filepath, filepath2))

        bbs_flipped = bbs_flipped.bounding_boxes[0]
        bbs_rotated = bbs_rotated.bounding_boxes[0]
        bbs_rotated_flipped = bbs_rotated_flipped.bounding_boxes[0]

        bl_image = {
            'filepath': blend_image,
            'label': row.label,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'file': row.file,
            'label2': row2.label,
            'x1_2': x1_2,
            'y1_2': y1_2,
            'x2_2': x2_2,
            'y2_2': y2_2,
        }

        fl_row = {
            'filepath': flipped,
            'label': row.label,
            'x1': bbs_flipped.x1,
            'y1': bbs_flipped.y1,
            'x2': bbs_flipped.x2,
            'y2': bbs_flipped.y2,
            'file': row.file
        }

        rt_row = {
            'filepath': rotated,
            'label': row.label,
            'x1': bbs_rotated.x1,
            'y1': bbs_rotated.y1,
            'x2': bbs_rotated.x2,
            'y2': bbs_rotated.y2,
            'file': row.file
        }

        fl_rt_row = {
            'filepath': rotated_flipped,
            'label': row.label,
            'x1': bbs_rotated_flipped.x1,
            'y1': bbs_rotated_flipped.y1,
            'x2': bbs_rotated_flipped.x2,
            'y2': bbs_rotated_flipped.y2,
            'file': row.file
        }

        augmented_df.append(bl_image)
        augmented_df.append(fl_row)
        augmented_df.append(rt_row)
        augmented_df.append(fl_rt_row)

    aug_df = pd.DataFrame(augmented_df)

    df = pd.concat([df, aug_df], ignore_index=True)
    print('Augmentation finished!')
    print('Augmented shape: ', aug_df.shape)

    return df


def load_combined_dataset(dirs):
    combined_data = []

    root = '.scratch/data/'
    annotations = 'Annotations/'
    csv_name = 'frameAnnotationsBOX.csv'

    for dir_name in dirs:
        csv_path = root + annotations + dir_name + csv_name
        img_path = root + dir_name + 'frames'

        data = pd.read_csv(csv_path, delimiter=';')
        data = setup(data, img_path)

        # Count the occurrences of each value
        value_counts = data['filepath'].value_counts()

        # Filter to keep only those that occur exactly once
        result_df = data[data['filepath'].isin(
            value_counts[value_counts == 1].index)]

        combined_data.append(result_df)

    combined_dataset = pd.concat(combined_data, ignore_index=True)

    return combined_dataset


def create_traffic_light_dataset(input_transform):

    dirs = ['daySequence1/',
            'daySequence2/',
            'nightSequence1/',
            'nightSequence2/',
            'dayTrain/dayClip1/',
            'dayTrain/dayClip2/',
            'dayTrain/dayClip3/',
            'dayTrain/dayClip4/',
            'dayTrain/dayClip5/',
            'dayTrain/dayClip6/',
            'dayTrain/dayClip7/',
            'dayTrain/dayClip8/',
            'dayTrain/dayClip9/',
            'dayTrain/dayClip10/',
            'dayTrain/dayClip11/',
            'dayTrain/dayClip12/',
            'dayTrain/dayClip13/',
            'nightTrain/nightClip1/',
            'nightTrain/nightClip2/',
            'nightTrain/nightClip3/',
            'nightTrain/nightClip4/',]

    print("Creating dataset...")
    combined_dataset = load_combined_dataset(dirs)

    # Splitting the dataset into 90% for training + validation (70% + 20%) and 10% for testing
    train_val, df_test = train_test_split(
        combined_dataset, test_size=0.1, random_state=42)

    df_train, df_val = train_test_split(
        train_val, test_size=0.2, random_state=42)

    df_train = augment_data(df_train)
    df_draw = pd.concat([df_test[:5], df_test[-5:]])

    print('Training dataset shape: ', df_train.shape)
    print('Validation dataset shape: ', df_val.shape)
    print('Testing dataset shape: ', df_test.shape)

    dataset_train = TrafficLightDataset(df_train, input_transform)
    dataset_val = TrafficLightDataset(df_val, input_transform)
    dataset_test = TrafficLightDataset(df_test, input_transform)
    dataset_draw = TrafficLightDataset(df_draw, input_transform)

    print("Dataset splitted (70%, 20%, 10%)!")

    return dataset_train, dataset_val, dataset_test, dataset_draw


def setup(df, root_dir):
    df.drop(['Origin file', 'Origin frame number',
                            'Origin track', 'Origin track frame number'], axis=1, inplace=True)
    df.columns = ['filepath', 'label',
                  'x1', 'y1', 'x2', 'y2']

    df['file'] = df['filepath'].str.split('/').str[-1]

    df['filepath'] = root_dir + '/' + df['file']

    new_columns = ['label2', 'x1_2', 'y1_2', 'x2_2', 'y2_2']

    # Adding new columns
    for column in new_columns:
        df[column] = None

    return df


class TrafficLightDataset(Dataset):
    def __init__(self, annotations, image_transform=None):
        self.annotations = annotations

        self.image_transform = image_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_location = str(self.annotations.iloc[idx, 0])
        print(img_location)

        # Load image
        image = self._load_image(img_location)
        if (self.image_transform is not None):
            image = self.image_transform(image)

        x0 = self.annotations.iloc[idx, 2]
        y0 = self.annotations.iloc[idx, 3]
        x1 = self.annotations.iloc[idx, 4]
        y1 = self.annotations.iloc[idx, 5]

        x0_2 = -5
        y0_2 = -5
        x1_2 = -5
        y1_2 = -5

        cell_value = self.annotations.iloc[idx, 8]
        if pd.notnull(cell_value):
            x0_2 = self.annotations.iloc[idx, 8]
            y0_2 = self.annotations.iloc[idx, 9]
            x1_2 = self.annotations.iloc[idx, 10]
            y1_2 = self.annotations.iloc[idx, 11]

        bbox = torch.tensor([x0, y0, x1, y1], dtype=torch.float32)
        bbox2 = torch.tensor([x0_2, y0_2, x1_2, y1_2], dtype=torch.float32)

        bbox = self._adjust_bbox(bbox, 1280, 960)
        if pd.notnull(cell_value):
            bbox2 = self._adjust_bbox(bbox2, 1280, 960)

        return image, bbox, bbox2

    def _load_image(self, file_path):
        image = cv2.imread(file_path,
                           cv2.IMREAD_COLOR)      # <H;W;C>
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _adjust_bbox(self, bbox, original_width, original_height):
        # original_size and new_size are (width, height)
        original_width, original_height = 1280, 960
        new_height = 512

        # Calculate the scale factor based on the height (since that's the shorter side in this case)
        scale_factor = new_height / original_height

        # Apply the scale factor to the width
        new_width = int(original_width * scale_factor)

        # Determine the scaling factors for width and height
        scale_w = new_width / original_width
        scale_h = new_height / original_height

        x0, y0, x1, y1 = bbox

        # Resize the bounding boxes
        resized_x0 = x0 * scale_w
        resized_y0 = y0 * scale_h
        resized_x1 = x1 * scale_w
        resized_y1 = y1 * scale_h

        # Normalize the coordinates to -1 to 1
        adjusted_bbox = torch.tensor([
            (resized_x0 / new_width) * 2 - 1,
            (resized_y0 / new_height) * 2 - 1,
            (resized_x1 / new_width) * 2 - 1,
            (resized_y1 / new_height) * 2 - 1
        ])

        return adjusted_bbox
