import pdb
import os
import cv2
import time
from glob import glob
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
from utils import *
from image_utils import *


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--model_folder_path",
        dest="model_folder_path",
        help="relative path to the folder where model checkpoints are saved",
    )
    parser.add_argument(
        "-p",
        "--predict_on",
        dest="predict_on",
        help="predict on train or test set, options: test or train",
        default="resnext101_32x4d",
    )
    return parser


class TestDataset(data.Dataset):
    def __init__(self, root, df, size, mean, std, tta=4):
        self.root = root
        self.size = size
        self.fnames = list(df["id_code"])
        self.num_samples = len(self.fnames)
        self.tta = tta
        self.TTA = albumentations.Compose(
            [
                albumentations.Rotate(limit=180, p=0.5),
                albumentations.Transpose(p=0.5),
                albumentations.Flip(p=0.5),
                albumentations.RandomScale(scale_limit=0.1),

            ]
        )
        self.transform = albumentations.Compose(
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
        image = load_ben_color(path, size=self.size, crop=True)

        images = [self.transform(image=image)["image"]]
        for _ in range(self.tta): # perform ttas
            aug_img = self.TTA(image=image)["image"]
            aug_img = self.transform(image=aug_img)["image"]
            images.append(aug_img)
        return torch.stack(images, dim=0)

    def __len__(self):
        return self.num_samples


def get_predictions(model, testset, tta):
    """return all predictions on testset in a list"""
    num_images = len(testset)
    predictions = []
    for i, batch in enumerate(tqdm(testset)):
        if tta:
            for images in batch:  # images.shape [n, 3, 96, 96] where n is num of 1+tta
                preds = model(images.to(device)) # [n, num_classes]
                predictions.append(preds.mean(dim=0).detach().tolist())
        else:
            preds = model(batch[:, 0].to(device))
            preds = preds.detach().tolist() #[1]
            predictions.extend(preds)

    return np.array(predictions)


def get_model_name_fold(model_folder_path):
    # example ckpt_path = weights/9-7_{modelname}_fold0_text/
    model_folder = model_folder_path.split("/")[1]  # 9-7_{modelname}_fold0_text
    model_name = "_".join(model_folder.split("_")[1:-2])  # modelname
    fold = model_folder.split("_")[-2]  # fold0
    fold = fold.split("fold")[-1]  # 0
    return model_name, int(fold)


if __name__ == "__main__":
    '''
    Generates predictions on train/test set using the ckpts saved in the model folder path
    and saves them in npy_folder in npy format which can be analyses later for different
    thresholds
    '''
    parser = get_parser()
    args = parser.parse_args()
    model_folder_path = args.model_folder_path
    predict_on = args.predict_on
    model_name, fold = get_model_name_fold(model_folder_path)

    if predict_on == "test":
        sample_submission_path = "data/sample_submission.csv"
    else:
        sample_submission_path = "data/train.csv"

    tta = 4 # number of augs in tta
    start_epoch = 0
    end_epoch = 26

    root = f"data/{predict_on}_images/"
    size = 300
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    #mean = (0, 0, 0)
    #std = (1, 1, 1)
    use_cuda = True
    num_classes = 1
    num_workers = 8
    batch_size = 16
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        cudnn.benchmark = True
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    df = pd.read_csv(sample_submission_path)
    testset = DataLoader(
        TestDataset(root, df, size, mean, std, tta),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if use_cuda else False,
    )

    model = get_model(model_name, num_classes, pretrained=None)
    model.to(device)
    model.eval()

    npy_folder = os.path.join(model_folder_path, "%s_npy" % predict_on)
    mkdir(npy_folder)

    print(f"\nUsing model: {model_name} | fold: {fold}")
    print(f"Predicting on: {predict_on} set")
    print(f"Root: {root}")
    print(f"size: {size}")
    print(f"mean: {mean}")
    print(f"std: {std}")
    print(f"Saving predictions at: {npy_folder}")
    print(f"From epoch {start_epoch} to {end_epoch}")
    print(f"Using tta: {tta}\n")

    for epoch in range(start_epoch, end_epoch+1):
        print(f"Using ckpt{epoch}.pth")
        ckpt_path = os.path.join(model_folder_path, "ckpt%d.pth" % epoch)
        state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state["state_dict"])
        best_thresholds = state["best_thresholds"]
        print(f"Best thresholds: {best_thresholds}")
        preds = get_predictions(model, testset, tta)

        pred_labels = predict(preds, best_thresholds)
        print(np.unique(pred_labels, return_counts=True))

        mat_to_save = [preds, best_thresholds]
        np.save(os.path.join(npy_folder, f"{predict_on}_ckpt{epoch}.npy"), mat_to_save)
        print("Predictions saved!")


'''
Footnotes

[1] a cuda variable can be converted to python list with .detach() (i.e., grad no longer required) then .tolist(), apart from that a cuda variable can be converted to numpy variable only by copying the tensor to host memory by .cpu() and then .numpy
'''
