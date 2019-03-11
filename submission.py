import pdb
import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import time
import pydicom

# from ssd import build_ssd
from torch.utils.data import DataLoader
from densenet import densenet169

# from dataloader import CLASSES
from models import Model, get_model

# from torchvision import albumentations
import albumentations
from albumentations import torch as AT
from torchvision.datasets.folder import pil_loader
import torch.utils.data as data


class TestDataset(data.Dataset):
    def __init__(self, root, sample_submission_path, tta=True, mean=(104, 117, 123)):
        self.root = root
        df = pd.read_csv(sample_submission_path)
        self.fnames = list(df["id"])
        self.num_samples = len(self.fnames)
        self.transform = albumentations.Compose(
            [
                albumentations.Resize(224, 224),
                albumentations.Normalize(p=1),
                AT.ToTensor(),
            ]
        )
        self.TTA = [
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
        ] if tta else None

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img_path = os.path.join(self.root, fname + ".tif")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # ****************************
        if self.TTA:
            images = [self.transform(image=image)["image"]]
            for aug in self.TTA:
                aug_img = aug(image=image)['image']
                aug_img = self.transform(image=aug_img)["image"]
                images.append(aug_img)
            return torch.stack(images, dim=0)

        image = self.transform(image=image)["image"]
        return image

    def __len__(self):
        return self.num_samples


def get_predictions(model_name, ckpt):
    print("Using trained model at %s" % ckpt)
    #net = Model(model_name, 1)
    net = get_model(model_name)
    state = torch.load(ckpt, map_location=lambda storage, loc: storage)
    net.load_state_dict(state["state_dict"])
    epoch = state["epoch"]
    net.to(device)
    net.eval()

    num_images = len(testset)
    predictions = []
    for i, batch in enumerate(tqdm(testset)):
        for images in batch: # images.shape [n, 3, 96, 96] where n is num of tta
            preds = torch.sigmoid(net(images.to(device))).detach()
            predictions.append(preds.cpu().mean().item())
        #if i==10:break
    return predictions


if __name__ == "__main__":

    models = {
                #"se_resnext50_32x4d_v3": [
                #        "weights/9Mar_se_resnext50_32x4d_v3_fold0/ckpt18.pth",
                #        "weights/9Mar_se_resnext50_32x4d_v3_fold1/ckpt18.pth",
                #        "weights/9Mar_se_resnext50_32x4d_v3_fold2/ckpt18.pth"
                #    ]
                "nasnetamobile":[
                    "weights/11Mar_nasnetamobile_fold0/ckpt17.pth"
                    ]
            }
    #************************************************
    # V.IMP: check input image size, and cv2.cvtColor(only used in nasnet)
    #**********************************************
    #sub_name = "9Mar_se_resnext50_32x4d_v3_fold0_1_2_allckpt18.csv"
    sub_name = "11Mar_nasnetamobile_fold0_ckpt17.csv"
    use_cuda = True
    use_tta = True
    torch.set_num_threads(12)
    num_workers = 4
    batch_size = 8
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda: torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else: torch.set_default_tensor_type("torch.FloatTensor")

    root = "data/test/"
    sample_submission_path = "data/sample_submission.csv"
    testset = DataLoader(
        TestDataset(root, sample_submission_path, use_tta),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if use_cuda else False)
    if use_cuda: cudnn.benchmark = True

    all_predictions = []
    for idx, model_name in enumerate(models.keys()):
        for ckpt in models[model_name]:
            predictions = get_predictions(model_name, ckpt)
            all_predictions.append(predictions)

    #pdb.set_trace()
    predictions = np.asarray(all_predictions)

    test_sub = pd.read_csv(sample_submission_path)
    sub_path = "weights/ensemble/" + sub_name
    print("Saving predictions at %s" % sub_path)
    for col_idx in tqdm(range(predictions.shape[1])):
        test_sub.loc[col_idx, "label"] = predictions[:, col_idx].mean()  # .at not working
    test_sub.to_csv(sub_path, index=False)





