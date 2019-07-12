import os
import pdb
import time
from datetime import datetime
import _thread
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from collections import defaultdict

# from ssd import build_ssd
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboard_logger import configure, log_value
from utils import *
from dataloader import provider
from shutil import copyfile
from models import Model, get_model

HOME = os.path.abspath(os.path.dirname(__file__))
now = datetime.now()
date = "%s-%s" % (now.day, now.month)
# print(HOME)


class Trainer(object):
    def __init__(self):
        remark = """
        with ben grahm's color version, test
                """
        self.fold = 0
        self.total_folds = 5
        self.class_weights = [1, 1.3, 1, 1.3, 1]
        # self.model_name = "resnext101_32x4d"
        # self.model_name = "se_resnet50_v0"
        self.model_name = "densenet121"
        ext_text = "bengrahmscolortest"
        self.folder = f"weights/{date}_{self.model_name}_fold{self.fold}_{ext_text}"
        print(f"model: {self.folder}")
        self.resume = False
        self.num_workers = 8
        self.batch_size = {"train": 32, "val": 8}
        self.num_classes = 5
        self.top_lr = 1e-4
        self.base_lr = self.top_lr * 0.001
        # self.base_lr = None
        self.momentum = 0.95
        self.size = 224
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        # mean = (0.5, 0.5, 0.5)
        # std = (0.5, 0.5, 0.5)
        # self.epoch_2_lr = {1: 2, 3: 5, 5: 2, 6:5, 7:2, 9:5} # factor to scale base_lr with
        # self.weight_decay = 5e-4
        self.best_loss = float("inf")
        self.start_epoch = 0
        self.num_epochs = 1000
        self.phases = ["train", "val"]
        self.cuda = torch.cuda.is_available()
        torch.set_num_threads(12)
        self.device = torch.device("cuda:0" if self.cuda else "cpu")
        data_folder = "data"
        # data_folder = 'external_data'
        self.images_folder = os.path.join(HOME, data_folder, "train_images")
        self.df_path = os.path.join(HOME, data_folder, "train.csv")
        self.save_folder = os.path.join(HOME, self.folder)
        self.model_path = os.path.join(self.save_folder, "model.pth")
        self.ckpt_path = os.path.join(self.save_folder, "ckpt.pth")
        self.tensor_type = (
            "torch.cuda.FloatTensor" if self.cuda else "torch.FloatTensor"
        )
        torch.set_default_tensor_type(self.tensor_type)
        self.net = Model(self.model_name, self.num_classes)
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = torch.nn.BCELoss()  # requires sigmoid pred inputs
        # self.optimizer = optim.SGD(
        #            self.net.parameters(),
        #            lr=self.top_lr,
        #            momentum=self.momentum,
        # )
        self.optimizer = optim.SGD(
            [
                {"params": self.net.model.parameters()},
                {"params": self.net.classifier.parameters(), "lr": self.top_lr},
            ],
            lr=self.base_lr if self.base_lr else self.top_lr,  # 1e-7#self.lr * 0.001,
            momentum=self.momentum,
        )
        # weight_decay=self.weight_decay)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=2, verbose=True
        )
        logger = logger_init(self.save_folder)
        self.log = logger.info
        if self.resume:
            self.resume_net()
        else:
            self.initialize_net()
        self.net = self.net.to(self.device)
        if self.cuda:
            cudnn.benchmark = True
        configure(os.path.join(self.save_folder, "logs"), flush_secs=5)
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
                class_weights=self.class_weights,
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in ["val"]
        }
        save_hyperparameters(self, remark)

    def resume_net(self):
        self.resume_path = os.path.join(self.save_folder, "ckpt.pth")
        self.log("Resuming training, loading {} ...".format(self.resume_path))
        state = torch.load(self.resume_path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(state["state_dict"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.cuda:
            for opt_state in self.optimizer.state.values():
                for k, v in opt_state.items():
                    if torch.is_tensor(v):
                        opt_state[k] = v.to(self.device)
        self.best_loss = state["best_loss"]
        self.start_epoch = state["epoch"] + 1

    def initialize_net(self):
        pass

    def forward(self, images, targets):
        images = images.to(self.device)
        # targets = targets.type(torch.LongTensor).to(self.device) # [1]
        targets = targets.type(torch.FloatTensor).to(self.device)
        outputs = self.net(images)
        outputs = torch.sigmoid(outputs)
        loss = self.criterion(outputs, targets)
        return loss, outputs

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
            self.optimizer.zero_grad()
            loss, outputs = self.forward(images, targets)
            running_loss += loss.item()
            # pdb.set_trace()
            outputs = outputs.detach()
            meter.update(targets.cpu(), outputs.cpu())
            if iteration % 50 == 0:
                iter_log(self.log, phase, epoch, iteration, total_iters, loss, start)
                # break
        best_threshold = 0.5
        if phase == "val":
            best_threshold = meter.get_best_threshold()
        epoch_loss = running_loss / total_images
        epoch_log(self.log, phase, epoch, epoch_loss, meter, start)
        torch.cuda.empty_cache()
        return epoch_loss, best_threshold

    def validate(self):
        t0 = time.time()
        for epoch in range(self.start_epoch, self.num_epochs):
            t_epoch_start = time.time()
            ckpt_path = os.path.join(self.save_folder, "ckpt%d.pth" % epoch)
            print("Loading ckpt: %s" % ckpt_path)
            state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            self.net.load_state_dict(state["state_dict"])
            val_loss, best_threshold = self.iterate(epoch, "val")
            print_time(self.log, t0, "Total time taken so far")


if __name__ == "__main__":
    model_trainer = Trainer()
    model_trainer.validate()


"""Footnotes
[1]: Crossentropy loss functions expects targets to be in labels (not one-hot) and of type
LongTensor, BCELoss expects targets to be FloatTensor

[2]: the ckpt.pth is saved after each train and val phase, val phase is neccessary becausue we want the best_threshold to be computed on the val set., Don't worry, the probability of your system going down just after a crucial training phase is low, just wait a few minutes for the val phase :p

"""
