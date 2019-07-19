import os
import pdb
import time
from datetime import datetime
import _thread

from apex import amp
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from collections import defaultdict

# from ssd import build_ssd
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboard_logger import Logger
from utils import *
from dataloader import provider
from shutil import copyfile
from models import Model, get_model

import warnings
warnings.filterwarnings("ignore")

HOME = os.path.abspath(os.path.dirname(__file__))
now = datetime.now()
date = "%s-%s" % (now.day, now.month)
# print(HOME)


class Trainer(object):
    def __init__(self):
        #remark = open("remark.txt", "r").read()
        remark = "Fine tuning on new samples, fold 2"
        self.fold = 2
        self.total_folds = 7
        self.class_weights = None #[1, 1, 1, 1, 1]
        #self.model_name = "resnext101_32x4d_v0"
        #self.model_name = "resnext101_32x16d"
        #self.model_name = "se_resnet50_v0"
        #self.model_name = "densenet121"
        self.model_name = "efficientnet-b5"
        ext_text = "bgccpold"
        date = "19-7"
        self.num_samples = None #5000
        self.folder = f"weights/{date}_{self.model_name}_fold{self.fold}_{ext_text}"
        self.resume = False
        self.pretrained = True
        self.pretrained_path = "weights/18-7_efficientnet-b5_fold0_bgccold/ckpt19.pth"
        self.resume_path = os.path.join(HOME, self.folder, "ckpt4.pth")
        #self.train_df_name = "train.csv"
        self.train_df_name = "train12.csv"
        #self.train_df_name = "train_old.csv"
        #data_folder = 'external_data'
        self.num_workers = 8
        self.batch_size = {"train": 20, "val": 8}
        self.num_classes = 1
        self.top_lr = 3e-5
        self.ep2unfreeze = 0
        self.num_epochs = 70
        #self.base_lr = self.top_lr * 0.001
        self.base_lr = None
        self.momentum = 0.95
        self.size = 256
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        #self.mean = (0, 0, 0)
        #self.std = (1, 1, 1)
        # self.epoch_2_lr = {1: 2, 3: 5, 5: 2, 6:5, 7:2, 9:5} # factor to scale base_lr
        # self.weight_decay = 5e-4
        self.best_qwk = 0
        self.best_loss = float("inf")
        self.start_epoch = 0
        self.phases = ["train", "val"]
        self.cuda = torch.cuda.is_available()
        torch.set_num_threads(12)
        self.device = torch.device("cuda:0" if self.cuda else "cpu")
        data_folder = 'data'
        self.images_folder = os.path.join(HOME, data_folder, "train_images")
        self.df_path = os.path.join(HOME, data_folder, self.train_df_name)
        self.save_folder = os.path.join(HOME, self.folder)
        self.model_path = os.path.join(self.save_folder, "model.pth")
        self.ckpt_path = os.path.join(self.save_folder, "ckpt.pth")
        self.tensor_type = "torch%s.FloatTensor" % (".cuda" if self.cuda else "")
        torch.set_default_tensor_type(self.tensor_type)
        self.net = get_model(self.model_name, self.num_classes)
        #self.criterion = torch.nn.CrossEntropyLoss()
        #self.criterion = torch.nn.BCELoss() # requires sigmoid pred inputs
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.Adam(
                   self.net.parameters(),
                   lr=self.top_lr,
        )
        #self.optimizer = optim.SGD(
        #    [
        #        {"params": self.net.model.parameters()},
        #        {"params": self.net.classifier.parameters(), "lr": self.top_lr},
        #    ],
        #    lr=self.base_lr if self.base_lr else self.top_lr,  # 1e-7#self.lr * 0.001,
        #    momentum=self.momentum,
        #)
        # weight_decay=self.weight_decay)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=3, verbose=True
        )
        logger = logger_init(self.save_folder)
        self.log = logger.info
        if self.resume or self.pretrained:
            self.load_state()
        else:
            self.initialize_net()
        self.net = self.net.to(self.device)

        # Mixed precision training
        self.net, self.optimizer = amp.initialize(
                self.net,
                self.optimizer,
                opt_level="O1",
                verbosity=0
        )

        if self.cuda: cudnn.benchmark = True
        self.tb = {
                x: Logger(os.path.join(self.save_folder, "logs", x))
                for x in ["train", "val"]
        } # tensorboard logger, see [3]
        mkdir(self.save_folder)
        self.dataloaders = {
            phase: provider(
                self.fold,
                self.total_folds,
                self.images_folder,
                self.df_path,
                phase,
                self.size,
                self.mean,
                self.std,
                class_weights = self.class_weights,
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
                num_samples=self.num_samples
            )
            for phase in ["train", "val"]
        }
        save_hyperparameters(self, remark)

    def load_state(self): # [4]
        if self.resume:
            path = self.resume_path
            self.log("Resuming training, loading {} ...".format(path))
        elif self.pretrained:
            path = self.pretrained_path
            self.log("loading pretrained, {} ...".format(path))
        state = torch.load(path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(state["state_dict"])

        if self.resume:
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_loss = state["best_loss"]
            self.best_qwk = state["best_qwk"]
            self.start_epoch = state["epoch"] + 1
            if self.start_epoch > 5:
                #self.base_lr = self.top_lr
                #print(f"Base lr = Top lr = {self.top_lr}")
                pass

        if self.cuda:
            for opt_state in self.optimizer.state.values():
                for k, v in opt_state.items():
                    if torch.is_tensor(v):
                        opt_state[k] = v.to(self.device)

    def initialize_net(self):
        # using `pretrainedmodels` library, models are already pretrained
        pass

    def forward(self, images, targets):
        #pdb.set_trace()
        images = images.to(self.device)
        #targets = targets.type(torch.LongTensor).to(self.device) # [1]
        targets = targets.type(torch.FloatTensor).to(self.device)
        targets = targets.view(-1, 1) # [n] -> [n, 1] V. imp for MSELoss
        outputs = self.net(images)
        #outputs = torch.sigmoid(outputs) # no sigmoid for regression mode
        loss = self.criterion(outputs, targets)
        return loss, outputs.detach()

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch, self.save_folder)
        self.log("Starting epoch: %d | phase: %s " % (epoch, phase))
        batch_size = self.batch_size[phase]
        start = time.time()
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0
        total_iters = len(dataloader)
        total_images = len(dataloader)
        for iteration, batch in enumerate(dataloader):
            fnames, images, targets = batch
            #pdb.set_trace()
            self.optimizer.zero_grad()
            loss, outputs = self.forward(images, targets)
            if phase == "train":
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                #loss.backward()
                self.optimizer.step()
            running_loss += loss.item()
            # pdb.set_trace()
            meter.update(targets, outputs)
            if iteration % 100 == 0:
                iter_log(self.log, phase, epoch, iteration, total_iters, loss, start)
                # break
        best_thresholds = meter.get_best_thresholds()
        epoch_loss = running_loss / total_images
        qwk = epoch_log(self.log, self.tb, phase, epoch, epoch_loss, meter, start)
        torch.cuda.empty_cache()
        return epoch_loss, qwk, best_thresholds

    def train(self):
        t0 = time.time()
        for epoch in range(self.start_epoch, self.num_epochs+1):
            t_epoch_start = time.time()
            if epoch == self.ep2unfreeze:
                for params in self.net.parameters():
                    params.requires_grad = True
                #self.base_lr = self.top_lr
                #self.optimizer = adjust_lr(self.base_lr, self.optimizer)

            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "best_qwk": self.best_qwk,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            val_loss, val_qwk, best_thresholds = self.iterate(epoch, "val")
            state["best_thresholds"] = best_thresholds
            torch.save(state, self.ckpt_path) # [2]
            self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
            #if val_qwk > self.best_qwk:
                self.log("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                #state["best_qwk"] = self.best_qwk = val_qwk
                torch.save(state, self.model_path)
            copyfile(
                self.ckpt_path, os.path.join(self.save_folder, "ckpt%d.pth" % epoch)
            )
            print_time(self.log, t_epoch_start, "Time taken by the epoch")
            print_time(self.log, t0, "Total time taken so far")
            self.log("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    model_trainer = Trainer()
    model_trainer.train()


'''Footnotes
[1]: Crossentropy loss functions expects targets to be in labels (not one-hot) and of type
LongTensor, BCELoss expects targets to be FloatTensor

[2]: the ckpt.pth is saved after each train and val phase, val phase is neccessary becausue we want the best_threshold to be computed on the val set., Don't worry, the probability of your system going down just after a crucial training phase is low, just wait a few minutes for the val phase :p

[3]: one tensorboard logger for train and val each, in same folder, so that we can see plots on the same graph

[4]: if pretrained is true, a model state from self.pretrained path will be loaded, if self.resume is true, self.resume_path will be loaded, both are true, self.resume_path will be loaded
'''

