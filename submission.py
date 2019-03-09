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
#from ssd import build_ssd
from torch.utils.data import DataLoader
from densenet import densenet169
#from dataloader import CLASSES
from models import Model
#from torchvision import albumentations
import albumentations
from albumentations import torch as AT
from torchvision.datasets.folder import pil_loader
import torch.utils.data as data


class TestDataset(data.Dataset):
    def __init__(self, root, sample_submission_path, mean=(104, 117, 123)):
        self.root = root
        df = pd.read_csv(sample_submission_path)
        self.fnames = list(df['id'])
        self.num_samples = len(self.fnames)
        self.transform = albumentations.Compose([
            #albumentations.Resize(112, 112),
            albumentations.Normalize(p=1),
            AT.ToTensor()])

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img_path = os.path.join(self.root, fname + '.tif')
        image = cv2.imread(img_path)
        image = self.transform(image=image)['image']
        return image

    def __len__(self):
        return self.num_samples

if __name__ == "__main__":
    use_cuda = True
    model_name = "se_resnext50_32x4d"
    #model_name = "se_resnet50"
    trained_model_path = 'weights/9Mar_%s_v2_fold0/model.pth' % model_name
    torch.set_num_threads(12)
    num_workers = 4
    batch_size = 64

    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    #net = densenet169()
    net = Model(model_name, 1)
    state = torch.load(trained_model_path, map_location= lambda storage, loc: storage)
    net.load_state_dict(state["state_dict"])
    epoch = state["epoch"]
    net.to(device)
    net.eval()
    # load data
    root = "data/test/"
    sample_submission_path = "data/sample_submission.csv"
    sub_path = trained_model_path.split('.')[0] + '_ep_%s.csv' % epoch
    test_sub = pd.read_csv(sample_submission_path)
    print("Using trained model at %s" % trained_model_path)
    print('Saving predictions at %s' % sub_path)

    testset = DataLoader(
            TestDataset(root, sample_submission_path),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if use_cuda else False)
    if use_cuda:
        cudnn.benchmark = True
    num_images = len(testset)
    predictions = []
    for i, batch in enumerate(tqdm(testset)):
        preds = torch.sigmoid(net(batch.to(device))).detach()      # forward pass
        predictions.extend(list(preds.cpu()))
    pdb.set_trace()
    for j, pred in enumerate(tqdm(predictions)):
        test_sub.loc[j, 'label'] = pred.item() # .at not working
        #pdb.set_trace()
    test_sub.to_csv(sub_path, index=False)
