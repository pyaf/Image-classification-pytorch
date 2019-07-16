
# References

# Pre-trained weights:
# https://github.com/facebookresearch/WSL-Images

# Borrowed a lot from abhishek, so give him an upvote:
# https://www.kaggle.com/abhishek/very-simple-pytorch-training-0-59
# https://www.kaggle.com/abhishek/pytorch-inference-kernel-lazy-tta

# Parameters

lr = 1e-5
img_size = 224
batch_size = 32
n_epochs = 10
coef = [0.5, 1.5, 2.5, 3.5]

# Libraries

import numpy as np
import pandas as pd
import torchvision
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import torch.optim as optim
import os
from torchvision.models.resnet import ResNet, Bottleneck
import cv2
from albumentations import Compose, HorizontalFlip, VerticalFlip, Rotate
from albumentations.pytorch import ToTensor

# Functions

class RetinopathyDatasetTrain(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('../input/aptos2019-blindness-detection/train_images', self.data.loc[idx, 'id_code'] + '.png')
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (img_size, img_size))
        label = torch.tensor(self.data.loc[idx, 'diagnosis'])
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return {'image': image, 'labels': label}

class RetinopathyDatasetTest(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('../input/aptos2019-blindness-detection/test_images', self.data.loc[idx, 'id_code'] + '.png')
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (img_size, img_size))
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return {'image': image}

def _resnext(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    model.load_state_dict(torch.load('../input/resnext101-32x8/ig_resnext101_32x8-c38310e5.pth'))
    return model

def resnext101_32x8d_wsl(progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnext('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)

# DataLoader

transform_train = Compose([
    HorizontalFlip(),
    Rotate(limit=365),
    VerticalFlip(),
    ToTensor()
])

transform_test = Compose([
    ToTensor()
])

train_dataset = RetinopathyDatasetTrain(csv_file='../input/aptos2019-blindness-detection/train.csv', transform=transform_train)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Model

model = resnext101_32x8d_wsl()

for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(2048, 1)

criterion = torch.nn.MSELoss()
device = torch.device("cuda:0")
plist = [{'params': model.parameters(), 'lr': 1e-5}]
optimizer = optim.Adam(plist, lr=1e-5)

model = model.to(device)
model.train()

# Training

for epoch in range(n_epochs):

    nb_tr_steps = 0
    tr_loss = 0

    if epoch == 1:

        # Unfreeze lower layers after the first epoch

        for param in model.parameters():
            param.requires_grad = True

    for step, batch in enumerate(tqdm(train_data_loader)):

        inputs = batch["image"]
        labels = batch["labels"].view(-1, 1)

        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        tr_loss += loss.item()
        nb_tr_steps += 1


    epoch_loss = tr_loss / len(train_data_loader)
    print('Training Loss: {:.4f}'.format(epoch_loss))

# Inference

test_dataset = RetinopathyDatasetTest(csv_file='../input/aptos2019-blindness-detection/sample_submission.csv', transform=transform_test)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

for param in model.parameters():
    param.requires_grad = False

model.eval()

test_pred = np.zeros((len(test_dataset), 1))

for i, x_batch in enumerate(tqdm(test_data_loader)):
    x_batch = x_batch["image"]
    pred = model(x_batch.to(device))
    test_pred[i * batch_size:(i + 1) * batch_size] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)

for i, pred in enumerate(test_pred):
    if pred < coef[0]:
        test_pred[i] = 0
    elif pred >= coef[0] and pred < coef[1]:
        test_pred[i] = 1
    elif pred >= coef[1] and pred < coef[2]:
        test_pred[i] = 2
    elif pred >= coef[2] and pred < coef[3]:
        test_pred[i] = 3
    else:
        test_pred[i] = 4

sample = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
sample.diagnosis = test_pred.astype(int)
sample.to_csv("submission.csv", index=False)
