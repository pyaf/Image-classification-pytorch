import os
import cv2
import pdb
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

    def __init__(self, fold, data_root, df_path, phase="train"):
        """
        Args:
            df (pd.DataFrame): a pandas DataFrame with image path and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.phase = phase
        self.data_root = data_root
        df = pd.read_csv(df_path)
        kfold = StratifiedKFold(20, shuffle=True, random_state=69) # 20 splits
        train_idx, val_idx = list(kfold.split(df['id'], df['label']))[fold]
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        self.df = train_df if phase=="train" else val_df
        self.fnames = self.df['id'].values
        self.labels = self.df['label'].values.astype('float32')
        self.num_samples = self.df.shape[0]
        self.transform = get_transforms( phase )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img_path = os.path.join(self.data_root, fname + '.tif')
        label = self.labels[idx]
        image = cv2.imread(img_path)
        image = self.transform(image=image)['image']
        return fname, image, label

    def __len__(self):
        #return 1000
        return len(self.df)


def get_transforms(phase):
    list_transforms = [
        #albumentations.Resize(112, 112)
    ]
    if phase == "train":
        list_transforms.extend([
            albumentations.RandomRotate90(p=0.5),
            albumentations.Transpose(p=0.5),
            albumentations.Flip(p=0.5),
            albumentations.OneOf([
                albumentations.CLAHE(clip_limit=2),
                albumentations.IAASharpen(),
                albumentations.IAAEmboss(),
                albumentations.RandomBrightnessContrast(),
                albumentations.JpegCompression(),
                albumentations.Blur(),
                albumentations.GaussNoise()], p=0.5),
            albumentations.HueSaturationValue(p=0.5),
            albumentations.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.5)])

    list_transforms.extend([
        albumentations.Normalize(p=1),
        AT.ToTensor(),
    ])
    return albumentations.Compose(list_transforms)


def provider(fold, phase, batch_size=8, num_workers=4):
    root = os.path.dirname(__file__)
    data_root = os.path.join(root, "data/train/")
    df_path = os.path.join(root, "data/train_labels.csv")
    image_dataset = ImageDataset(
                        fold,
                        data_root,
                        df_path,
                        phase)
    dataloader = DataLoader(
                    image_dataset,
                    batch_size=batch_size,
                    shuffle=phase == "train",
                    num_workers=num_workers,
                    pin_memory=True,
                )
    return dataloader


if __name__ == "__main__":
    phase = "train"
    num_workers = 16
    dataloader = provider(phase, num_workers=num_workers)
    for batch in dataloader:
        fnames, images, labels = batch
        print(images.shape, labels.shape)
    pdb.set_trace()

