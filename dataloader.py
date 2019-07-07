import os
import cv2
import pdb
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader
from sklearn.model_selection import train_test_split, StratifiedKFold
import albumentations
from albumentations import torch as AT


class ImageDataset(Dataset):
    """training dataset."""

    def __init__(self, fold, images_folder, df_path, size, mean, std, phase="train"):
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
        df = pd.read_csv(df_path)

        # K fold CV
        kfold = StratifiedKFold(10, shuffle=True, random_state=69)  # 20 splits
        train_idx, val_idx = list(kfold.split(df["id_code"], df["diagnosis"]))[fold]
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        self.df = train_df if phase == "train" else val_df
        self.num_samples = self.df.shape[0]
        self.fnames = self.df["id_code"].values
        self.labels = self.df["diagnosis"].values.astype("int64")
        self.num_classes = len(np.unique(self.labels))
        # self.labels = np.eye(self.num_classes)[self.labels] # to one-hot
        # CrossEntropyLoss doesn't expect inputs to be one-hot, but indices
        self.transform = get_transforms(phase, size, mean, std)

    def __getitem__(self, idx):
        IMG_SIZE = self.size
        fname = self.fnames[idx]
        img_path = os.path.join(self.images_folder, fname + ".png")
        label = self.labels[idx]

        image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = cv2.addWeighted(
            image, 4, cv2.GaussianBlur(image, (0, 0), IMG_SIZE / 10), -4, 128
        )  # Ben Graham's preprocessing method [1]

        # (IMG_SIZE, IMG_SIZE) -> (IMG_SIZE, IMG_SIZE, 3)
        image = image.reshape(IMG_SIZE, IMG_SIZE, 1)
        image = np.repeat(image, 3, axis=-1)

        image = self.transform(image=image)["image"]
        return fname, image, label

    def __len__(self):
        # return 100
        return len(self.df)


def get_transforms(phase, size, mean, std):
    list_transforms = [
        # albumentations.Resize(size, size) # now doing this in __getitem__()
    ]
    if phase == "train":
        list_transforms.extend(
            [
                albumentations.RandomRotate90(p=0.5),
                albumentations.Transpose(p=0.5),
                albumentations.Flip(p=0.5),
                albumentations.OneOf(
                    [
                        albumentations.CLAHE(clip_limit=2),
                        albumentations.IAASharpen(),
                        albumentations.IAAEmboss(),
                        albumentations.RandomBrightnessContrast(),
                        albumentations.JpegCompression(),
                        albumentations.Blur(),
                        albumentations.GaussNoise(),
                    ],
                    p=0.5,
                ),
                albumentations.HueSaturationValue(p=0.5),
                albumentations.ShiftScaleRotate(
                    shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.5
                ),
            ]
        )

    list_transforms.extend(
        [albumentations.Normalize(mean=mean, std=std, p=1), AT.ToTensor()]
    )
    return albumentations.Compose(list_transforms)


def provider(
    fold, images_folder, df_path, phase, size, mean, std, batch_size=8, num_workers=4
):
    image_dataset = ImageDataset(fold, images_folder, df_path, size, mean, std, phase)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=(phase == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )
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
    for batch in dataloader:
        fnames, images, labels = batch
        print(images.shape, labels.shape)
        pdb.set_trace()


"""
Footnotes:

https://github.com/btgraham/SparseConvNet/tree/kaggle_Diabetic_Retinopathy_competition

"""
