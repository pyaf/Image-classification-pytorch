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

from argparse import ArgumentParser

# from torchvision import albumentations
import albumentations
from albumentations import torch as AT
from torchvision.datasets.folder import pil_loader
import torch.utils.data as data


class TestDataset(data.Dataset):
    def __init__(self, root, sample_submission_path, size, mean, std, tta=True):
        self.root = root
        self.size = size
        df = pd.read_csv(sample_submission_path)
        self.fnames = list(df["id_code"])
        self.num_samples = len(self.fnames)
        self.transform = albumentations.Compose([
                albumentations.Normalize(mean=mean, std=std, p=1),
            ])
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
            [
                albumentations.Resize(size, size),
                AT.ToTensor()
            ]
        )

    def __getitem__(self, idx):
        IMG_SIZE = self.size
        fname = self.fnames[idx]
        img_path = os.path.join(self.root, fname + ".png")
        image = cv2.imread(img_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # ****************************
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = cv2.addWeighted(
           image, 4, cv2.GaussianBlur(image, (0, 0), IMG_SIZE / 10), -4, 128
        )  # Ben Graham's preprocessing method [1]

        ## (IMG_SIZE, IMG_SIZE) -> (IMG_SIZE, IMG_SIZE, 3)
        image = image.reshape(IMG_SIZE, IMG_SIZE, 1)
        image = np.repeat(image, 3, axis=-1)

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
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--ckpt name",
        dest="ckpt_path",
        help="ckpt path of the ckpt to use",
        metavar="FOLDER",
    )
    parser.add_argument(
        "-m",
        "-- model_name",
        dest="model_name",
        help="Model name",
        default="resnext101_32x4d",
    )
    parser.add_argument(
        "-p",
        "--predict_on",
        dest="predict_on",
        help="predict on train or test set, options: test or train",
        default="resnext101_32x4d",
    )


    args = parser.parse_args()
    ckpt_path = args.ckpt_path
    model_name = args.model_name
    predict_on = args.predict_on
    print("Using model: %s" % model_name)

    if predict_on == 'test':
        print('Predicting on test set')
        root = "data/test_images/"
        sample_submission_path = "data/sample_submission.csv"
        sub_path = ckpt_path.replace(".pth", ".csv")
    else:
        print('Predicting on train set')
        root = "data/train_images/"
        sample_submission_path = "data/train.csv"
        sub_path = ckpt_path.replace(".pth", "_train.csv")

    print("Saving predictions at %s" % sub_path)
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # mean = (0.5, 0.5, 0.5)
    # std = (0.5, 0.5, 0.5)

    use_cuda = True
    use_tta = False
    #torch.set_num_threads(12)
    num_classes = 5
    num_workers = 4
    batch_size = 8
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

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
