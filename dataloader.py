import os
import cv2
import pdb
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, sampler
from torchvision.datasets.folder import pil_loader
from sklearn.model_selection import train_test_split, StratifiedKFold
from utils import to_multi_label
import albumentations
from albumentations import torch as AT


class ImageDataset(Dataset):
    """training dataset."""

    def __init__(self, df, images_folder, size, mean, std, phase="train"):
        """
        Args:
            fold: for k fold CV
            images_folder: the folder which contains the images
            df_path: data frame path, which contains image ids
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.phase = phase
        self.size = size
        self.images_folder = images_folder
        self.df = df
        self.num_samples = self.df.shape[0]
        self.fnames = self.df["id_code"].values
        self.labels = self.df["diagnosis"].values.astype("int64")
        self.num_classes = len(np.unique(self.labels))
        self.labels = to_multi_label(self.labels, self.num_classes) # [1]
        # self.labels = np.eye(self.num_classes)[self.labels]
        self.transform = get_transforms(phase, size, mean, std)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        label = self.labels[idx]
        image = np.load(os.path.join(self.images_folder, 'npy', fname + '.npy'))
        image = self.transform(image=image)["image"]
        return fname, image, label

    def __len__(self):
        # return 100
        return len(self.df)


def get_transforms(phase, size, mean, std):
    list_transforms = [
        # albumentations.Resize(size, size) # now doing this in __getitem__()
        albumentations.Normalize(mean=mean, std=std, p=1),
    ]
    if phase == "train":
        list_transforms.extend(
            [
                albumentations.Rotate(limit=360, p=0.5),
                #albumentations.Transpose(p=0.5),
                albumentations.Flip(p=0.5),
                albumentations.RandomScale(scale_limit=0.1),
                #albumentations.OneOf(
                #    [
                #        albumentations.CLAHE(clip_limit=2),
                #        albumentations.IAASharpen(),
                #        albumentations.IAAEmboss(),
                #        albumentations.RandomBrightnessContrast(),
                #        albumentations.JpegCompression(),
                #        albumentations.Blur(),
                #        albumentations.GaussNoise(),
                #    ],
                #    p=0.5,
                #),
                #albumentations.HueSaturationValue(p=0.5),
                #albumentations.ShiftScaleRotate(
                #    shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.5
                #),
            ]
        )

    list_transforms.extend(
        [
            albumentations.Resize(size, size), # because RandomScale is applied
            AT.ToTensor(),
        ]
    )
    return albumentations.Compose(list_transforms)


def get_sampler(df):
    labels, label_counts = np.unique(df['diagnosis'].values, return_counts=True) # [2]
    #class_weights = max(label_counts) / label_counts # higher count, lower weight
    #class_weights = class_weights / sum(class_weights)
    class_weights = [1, 2, 1, 2, 2]
    print("classes, weights", labels, class_weights)
    dataset_weights = [class_weights[idx] for idx in df['diagnosis']]
    datasampler = sampler.WeightedRandomSampler(dataset_weights, len(df))
    return datasampler


def provider(
    fold, images_folder, df_path, phase, size, mean, std, batch_size=8, num_workers=4
):
    df = pd.read_csv(df_path)
    bad_indices = np.load('data/bad_train_indices.npy')
    df = df.drop(df.index[bad_indices]) # remove duplicates having diff diagnosis
    kfold = StratifiedKFold(5, shuffle=True, random_state=69)  # 20 splits
    train_idx, val_idx = list(kfold.split(df["id_code"], df["diagnosis"]))[fold]
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    df = train_df if phase == "train" else val_df

    image_dataset = ImageDataset(df, images_folder, size, mean, std, phase)
    datasampler = get_sampler(df) if phase == "train" else None
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        sampler=None #datasampler
    ) # shuffle and sampler are mutually exclusive args
    return dataloader


if __name__ == "__main__":
    phase = "train"
    num_workers = 0
    fold = 0
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    size = 224

    root = os.path.dirname(__file__)  # data folder
    images_folder = os.path.join(root, "data/train_images/")  # contains train images
    df_path = os.path.join(root, "data/train.csv")  # contains train ids and labels

    dataloader = provider(
        fold, images_folder, df_path, phase, size, mean, std, num_workers=num_workers
    )
    total_labels = []
    total_len = len(dataloader)
    for idx, batch in enumerate(dataloader):
        fnames, images, labels = batch
        print("%d/%d" % (idx, total_len), images.shape, labels.shape, labels)
        total_labels.extend(labels.tolist())
        pdb.set_trace()
    print(np.unique(total_labels, return_counts=True))


"""
Footnotes:

https://github.com/btgraham/SparseConvNet/tree/kaggle_Diabetic_Retinopathy_competition

[1] CrossEntropyLoss doesn't expect inputs to be one-hot, but indices
[2] .value_counts() returns in descending order of counts (not sorted by class numbers :)

"""
