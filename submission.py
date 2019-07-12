import pdb
import os
import cv2
import time
import torch
import scipy
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import albumentations
from albumentations import torch as AT
from torchvision.datasets.folder import pil_loader
import torch.utils.data as data
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from models import Model, get_model
from utils import get_preds
from image_utils import *


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--ckpt_path",
        dest="ckpt_path",
        help="relative path to the saved model checkpoint to be used",
    )
    # parser.add_argument(
    #    "-m",
    #    "-- model_name",
    #    dest="model_name",
    #    help="Model name",
    #    default="se_resnet50",
    # )
    parser.add_argument(
        "-p",
        "--predict_on",
        dest="predict_on",
        help="predict on train or test set, options: test or train",
        default="resnext101_32x4d",
    )
    # parser.add_argument(
    #    "-f",
    #    "--fold",
    #    dest="fold",
    #    help="which fold the model was trained on?",
    #    default="resnext101_32x4d",
    # )

    return parser


def get_best_threshold(model, fold, total_folds):
    """
    root: the folder with the images
    model: the model to use for prediction
    fold: which are we talking abt? gotta get the val set
    total_folds: required for val set
    train_df: training dataframe
    """

    train_df_path = "data/train.csv"
    train_df = pd.read_csv(train_df_path)
    bad_indices = np.load("data/bad_train_indices.npy")
    df = train_df.drop(train_df.index[bad_indices])  # remove duplicates
    kfold = StratifiedKFold(total_folds, shuffle=True, random_state=69)
    train_idx, val_idx = list(kfold.split(df["id_code"], df["diagnosis"]))[fold]
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    # a dataloader for val set
    root = "data/train_images"
    valset = DataLoader(
        TestDataset(root, val_df, size, mean, std, False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if use_cuda else False,
    )
    print("Getting predictions on validation set for fold %d..." % fold)
    val_pred = get_predictions(model, valset, False)

    def compute_score_inv(threshold):
        # pdb.set_trace()
        y1 = val_pred > threshold
        y1 = get_preds(y1, num_classes)
        y2 = val_df.diagnosis.values
        score = cohen_kappa_score(y1, y2, weights="quadratic")
        return 1 - score

    print("Getting the best threshold..")
    simplex = scipy.optimize.minimize(compute_score_inv, 0.5, method="nelder-mead")

    best_threshold = simplex["x"][0]
    print("best threshold: %s" % best_threshold)
    return best_threshold


class TestDataset(data.Dataset):
    def __init__(self, root, df, size, mean, std, tta=True):
        self.root = root
        self.size = size
        self.fnames = list(df["id_code"])
        self.num_samples = len(self.fnames)
        #self.transform = albumentations.Compose(
        #    []
        #)
        self.TTA = (
            [
                #albumentations.RandomRotate90(p=1),
                albumentations.Transpose(p=1),
                albumentations.Flip(p=1),
                albumentations.RandomScale(scale_limit=0.1),
                albumentations.Compose(
                    [
                        #albumentations.RandomRotate90(p=0.8),
                        albumentations.Transpose(p=0.8),
                        albumentations.Flip(p=0.8),
                        albumentations.RandomScale(scale_limit=0.1),
                    ]
                ),
            ]
        )
        self.last_transform = albumentations.Compose(
            [
                albumentations.Normalize(mean=mean, std=std, p=1),
                albumentations.Resize(size, size),
                AT.ToTensor()
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname + ".png")
        # image = load_image(path, size)
        # image = load_ben_gray(path)
        image = load_ben_color(path, size=self.size, crop=False)

        images = [
            self.last_transform(image=image)["image"]
        ]
        if self.TTA:
            for aug in self.TTA:
                aug_img = aug(image=image)["image"]
                #aug_img = self.transform(image=aug_img)["image"]
                aug_img = self.last_transform(image=aug_img)["image"]
                images.append(aug_img)
        return torch.stack(images, dim=0)

    def __len__(self):
        return self.num_samples


def get_predictions(model, testset, use_tta):
    """return all predictions on testset in a list"""
    num_images = len(testset)
    predictions = []
    for i, batch in enumerate(tqdm(testset)):
        if use_tta:
            for images in batch:  # images.shape [n, 3, 96, 96] where n is num of 1+tta
                preds = torch.sigmoid(model(images.to(device))) # [n, num_classes]
                #pred = torch.argmax(preds.cpu().mean(dim=1), dim=1).item()
                predictions.append(preds.mean(dim=0).detach().tolist())
        else:
            preds = model(batch[:, 0].to(device))
            # preds = torch.argmax(preds.cpu(), dim=1).tolist() # CELoss
            # preds = (torch.sigmoid(preds) > threshold).cpu().numpy()
            # preds = get_preds(preds, num_classes)
            # predictions.extend(preds.tolist())

            preds = torch.sigmoid(preds).detach().tolist()
            predictions.extend(preds)
        # if i==10:break

    return np.array(predictions)


def get_model_name_fold(ckpt_path):
    # example ckpt_path = weights/9-7_{modelname}_fold0_text/ckpt.pth
    model_folder = ckpt_path.split("/")[1]  # 9-7_{modelname}_fold0_text
    #model_name = model_folder.split("_")[1]  # modelname
    model_name = "_".join(model_folder.split("_")[1:-2])  # modelname
    fold = model_folder.split("_")[-2]  # fold0
    fold = fold.split("fold")[-1]  # 0
    return model_name, int(fold)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    ckpt_path = args.ckpt_path
    predict_on = args.predict_on
    model_name, fold = get_model_name_fold(ckpt_path)

    print("Using model: %s | fold: %s" % (model_name, fold))
    if predict_on == "test":
        print("Predicting on test set")
        root = "data/test_images/"
        sample_submission_path = "data/sample_submission.csv"
        sub_path = ckpt_path.replace(".pth", ".csv")
    else:
        print("Predicting on train set")
        root = "data/train_images/"
        sample_submission_path = "data/train.csv"
        sub_path = ckpt_path.replace(".pth", "_train.csv")

    total_folds = 5
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    use_cuda = True
    use_tta = True
    num_classes = 5
    num_workers = 4
    batch_size = 8
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    if use_cuda:
        cudnn.benchmark = True

    print("Using trained model at %s" % ckpt_path)
    model = get_model(model_name, num_classes, pretrained=None)
    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])
    epoch = state["epoch"]
    best_threshold = state["best_threshold"]
    #best_threshold = 0.6 # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    model.to(device)
    model.eval()
    print("Model from Epoch:", epoch)
    print("best threshold: ", best_threshold)
    # best_threshold = get_best_threshold(model, fold, total_folds)

    df = pd.read_csv(sample_submission_path)
    testset = DataLoader(
        TestDataset(root, df, size, mean, std, use_tta),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if use_cuda else False,
    )

    predictions = get_predictions(model, testset, use_tta)
    predictions = predictions > best_threshold
    predictions = get_preds(predictions, num_classes)

    test_sub = pd.read_csv(sample_submission_path)
    for idx, pred in enumerate(tqdm(predictions)):
        test_sub.loc[idx, "diagnosis"] = pred

    print("Saving predictions at %s" % sub_path)
    test_sub.to_csv(sub_path, index=False)
