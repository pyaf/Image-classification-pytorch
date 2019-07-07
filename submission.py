import pdb
import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import time

from torch.utils.data import DataLoader

from models import Model, get_model

# from torchvision import albumentations
import albumentations
from albumentations import torch as AT
from torchvision.datasets.folder import pil_loader
import torch.utils.data as data


class TestDataset(data.Dataset):
    def __init__(self, root, sample_submission_path, size, mean, std, tta=True):
        self.root = root
        df = pd.read_csv(sample_submission_path)
        self.fnames = list(df["id_code"])
        self.num_samples = len(self.fnames)
        self.transform = albumentations.Compose([albumentations.Resize(size, size)])
        self.TTA = (
            [
                albumentations.RandomRotate90(p=1),
                albumentations.Transpose(p=1),
                albumentations.Flip(p=1),
                albumentations.Compose(
                    [
                        albumentations.RandomRotate90(p=0.8),
                        albumentations.Transpose(p=0.8),
                        albumentations.Flip(p=0.8),
                    ]
                ),
            ]
            if tta
            else None
        )
        self.last_transform = albumentations.Compose(
            [albumentations.Normalize(mean=mean, std=std, p=1), AT.ToTensor()]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img_path = os.path.join(self.root, fname + ".png")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # ****************************
        images = [
            self.last_transform(image=self.transform(image=image)["image"])["image"]
        ]
        if self.TTA:
            for aug in self.TTA:
                aug_img = aug(image=image)["image"]
                aug_img = self.transform(image=aug_img)["image"]
                aug_img = self.last_transform(image=aug_img)["image"]
                images.append(aug_img)
        return torch.stack(images, dim=0)

    def __len__(self):
        return self.num_samples


def get_predictions(model_name, num_classes, ckpt, use_tta):
    print("Using trained model at %s" % ckpt)
    net = get_model(model_name, num_classes, pretrained=None)
    state = torch.load(ckpt, map_location=lambda storage, loc: storage)
    net.load_state_dict(state["state_dict"])
    epoch = state["epoch"]
    net.to(device)
    net.eval()

    num_images = len(testset)
    predictions = []
    for i, batch in enumerate(tqdm(testset)):
        if use_tta:
            for images in batch:  # images.shape [n, 3, 96, 96] where n is num of 1+tta
                preds = net(images.to(device)).detach()
                predictions.append(torch.argmax(preds.cpu().mean(dim=1), dim=1).item())
        else:
            preds = net(batch[:, 0].to(device)).detach()
            # pdb.set_trace()
            predictions.extend(torch.argmax(preds.cpu(), dim=1).tolist())
        # if i==10:break
    return predictions


if __name__ == "__main__":

    model_name = "resnext101_32x4d"
    fold = 1
    ckpt = "ckpt30.pth"
    ckpt_path = "weights/6-7_%s_fold%s/%s" % (model_name, fold, ckpt)
    sub_path = ckpt_path.replace(".pth", "_train.csv")
    print("Saving predictions at %s" % sub_path)
    size = 224
    #size = 96
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # mean = (0.5, 0.5, 0.5)
    # std = (0.5, 0.5, 0.5)

    use_cuda = True
    use_tta = False
    torch.set_num_threads(12)
    num_classes = 5
    num_workers = 2
    batch_size = 4
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    #root = "data/test_images/"
    #sample_submission_path = "data/sample_submission.csv"
    root = 'data/train_images/'
    sample_submission_path = 'data/train.csv'
    testset = DataLoader(
        TestDataset(root, sample_submission_path, size, mean, std, use_tta),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if use_cuda else False,
    )
    if use_cuda:
        cudnn.benchmark = True

    predictions = get_predictions(model_name, num_classes, ckpt_path, use_tta)
    test_sub = pd.read_csv(sample_submission_path)
    for idx, pred in enumerate(tqdm(predictions)):
        test_sub.loc[idx, "diagnosis"] = pred

    test_sub.to_csv(sub_path, index=False)
