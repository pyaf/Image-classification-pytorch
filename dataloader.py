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
        # self.labels = to_multi_label(self.labels, self.num_classes)  # [1]
        # self.labels = np.eye(self.num_classes)[self.labels]
        self.transform = get_transforms(phase, size, mean, std)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        label = self.labels[idx]
        path = os.path.join(self.images_folder, "npy_bengrahm_color", fname + ".npy")
        image = np.load(path)
        image = self.transform(image=image)["image"]
        return fname, image, label

    def __len__(self):
        #return 100
        return len(self.df)

def get_transforms(phase, size, mean, std):
    list_transforms = [
        # albumentations.Resize(size, size) # now doing this in __getitem__()
    ]
    if phase == "train":
        list_transforms.extend(
            [
                albumentations.Rotate(limit=180, p=0.5),
                albumentations.Transpose(p=0.5),
                albumentations.Flip(p=0.5),
                albumentations.RandomScale(scale_limit=0.1),
                #albumentations.OneOf(
                #   [
                #       albumentations.CLAHE(clip_limit=2),
                #       albumentations.IAASharpen(),
                #       albumentations.IAAEmboss(),
                #       albumentations.RandomBrightnessContrast(),
                #       albumentations.JpegCompression(),
                #       albumentations.Blur(),
                #       albumentations.GaussNoise(),
                #   ],
                #   p=0.5,
                #),
                #albumentations.HueSaturationValue(p=0.5),
                #albumentations.ShiftScaleRotate(
                #   shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.5
                #),
            ]
        )

    list_transforms.extend(
        [

            albumentations.Normalize(mean=mean, std=std, p=1),
            albumentations.Resize(size, size),  # because RandomScale is applied
            AT.ToTensor(normalize=None), # [6]
        ]
    )
    return albumentations.Compose(list_transforms)

def get_sampler(df, class_weights=None):
    if class_weights is None:
        labels, label_counts = np.unique(
            df["diagnosis"].values, return_counts=True
        )  # [2]
        # class_weights = max(label_counts) / label_counts # higher count, lower weight
        # class_weights = class_weights / sum(class_weights)
        class_weights = [1, 1, 1, 1, 1]
    print("weights", class_weights)
    dataset_weights = [class_weights[idx] for idx in df["diagnosis"]]
    datasampler = sampler.WeightedRandomSampler(dataset_weights, len(df))
    return datasampler

def resampled(df):

    ''' resample `total` data points from old data, following the dist of org data '''
    def sample(obj, total=7800): # [5]
        return obj.sample(n=int(count_dict[obj.name] * total), replace=False)

    count_dict = {
        0: 0.5,
        2: 0.28,
        1: 0.11,
        4: 0.09,
        3: 0.06
    } # approx dist of org data, notice the order of keys

    sampled_df = df.groupby('diagnosis').apply(sample).reset_index(drop=True)
    return sampled_df


def provider(
    fold,
    total_folds,
    images_folder,
    df_path,
    phase,
    size,
    mean,
    std,
    class_weights=None,
    batch_size=8,
    num_workers=4,
    num_samples=4000,
):
    df = pd.read_csv(df_path)
    HOME = os.path.abspath(os.path.dirname(__file__))

    #bad_indices = np.load(os.path.join(HOME, "data/bad_train_indices.npy"))
    #dup_indices = np.load(
    #    os.path.join(HOME, "data/dups_with_same_diagnosis.npy")
    #)  # [3]
    #duplicates = df.iloc[dup_indices]
    #all_dups = np.array(list(bad_indices) + list(dup_indices))
    #df = df.drop(df.index[all_dups])  # remove duplicates and split train/val
    #''' line 163 also commented out'''

    #print('num_samples:', num_samples)
    #if num_samples: # [4]
    #    df = df.iloc[:num_samples]

    ''' to be used only with old data training '''
    #df = resampled(df)
    #print(f'sampled df shape: {df.shape}')
    #print('data dist:\n',  df['diagnosis'].value_counts(normalize=True))

    kfold = StratifiedKFold(total_folds, shuffle=True, random_state=69)
    train_idx, val_idx = list(kfold.split(df["id_code"], df["diagnosis"]))[fold]
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    #train_df = train_df.append(duplicates, ignore_index=True)  # add dups, not bad ones

    #train_df = pd.read_csv('data/train_old.csv')
    #val_df = pd.read_csv('data/train12.csv')

    df = train_df if phase == "train" else val_df

    image_dataset = ImageDataset(df, images_folder, size, mean, std, phase)

    datasampler = None
    if phase == "train" and class_weights:
        datasampler = get_sampler(df, class_weights=class_weights)
    print('datasampler:', datasampler)

    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False if datasampler else True,
        sampler=datasampler,
    )  # shuffle and sampler are mutually exclusive args
    return dataloader


if __name__ == "__main__":
    phase = "train"
    num_workers = 0
    fold = 0
    total_folds = 5
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    size = 256

    root = os.path.dirname(__file__)  # data folder
    data_folder = "data"
    # train_df_name = 'train.csv'
    train_df_name = "train12.csv"
    num_samples = None #5000
    class_weights = None #[1, 1, 1, 1, 1]
    batch_size = 1
    # data_folder = 'external_data'
    images_folder = os.path.join(root, data_folder, "train_images/")  #
    df_path = os.path.join(root, data_folder, train_df_name)  #

    dataloader = provider(
        fold,
        total_folds,
        images_folder,
        df_path,
        phase,
        size,
        mean,
        std,
        class_weights=class_weights,
        batch_size=batch_size,
        num_workers=num_workers,
        num_samples=num_samples,
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
[3]: bad_indices are those which have conflicting diagnosises, duplicates are those which have same duplicates, we shouldn't let them split in train and val set, gotta maintain the sanctity of val set
[4]: used when the dataframe include external data and we want to sample limited number of those
[5]: as replace=False,  total samples can be a finite number so that those many number of classes exist in the dataset, and as the count_dist is approx, not normalized to 1, 7800 is optimum, totaling to ~8100 samples

[6]: albumentations.Normalize will divide by 255, subtract mean and divide by std. output dtype = float32. ToTensor converts to torch tensor and divides by 255 if input dtype is uint8.
"""
