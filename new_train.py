import os
import pdb
import time
import _thread
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from collections import defaultdict
#from ssd import build_ssd
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboard_logger import configure, log_value
from utils import iter_log, epoch_log, logger, print_time, Meter, mkdir
from dataloader import provider
from shutil import copyfile
from models import Model

HOME = os.path.dirname(__file__)
print(HOME)


class Trainer(object):
    def __init__(self, fold):
        model_name = "se_resnext50_32x4d"
        folder = 'weights/7Mar_%s_fold%s' % (model_name, fold)
        self.resume = False
        self.fold = fold
        self.num_workers = 16
        self.batch_size = {'train': 32, 'val': 16}
        self.lr = 7e-5 #4e-4 #0.00007 #1e-3
        self.momentum = 0.95
        self.weight_decay = 5e-4
        self.best_loss = float("inf")
        self.start_epoch = 0
        self.num_epochs = 1000
        self.phases = ['train', 'val']
        self.cuda = torch.cuda.is_available()
        torch.set_num_threads(12)
        self.device = torch.device("cuda:0" if self.cuda else "cpu")
        self.save_folder = os.path.join(HOME, folder)
        self.model_path = os.path.join(self.save_folder, "model.pth")
        self.ckpt_path = os.path.join(self.save_folder, "ckpt.pth")
        self.tensor_type = "torch.cuda.FloatTensor" if self.cuda else "torch.FloatTensor"
        torch.set_default_tensor_type(self.tensor_type)
        self.net = Model(model_name, 1)
        self.optimizer = optim.SGD([
                        {"params": self.net.layer0.parameters()},
                        {"params": self.net.layer1.parameters()},
                        {"params": self.net.layer2.parameters()},
                        {"params": self.net.layer3.parameters()},
                        {"params": self.net.layer4.parameters()},
                        {"params": self.net.avg_pool.parameters()},
                        {"params": self.net.last_linear.parameters(), "lr": self.lr}],
                        lr=self.lr * 0.001,
                        momentum=self.momentum,
                        weight_decay=self.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=2, verbose=True)
        self.criterion = torch.nn.BCEWithLogitsLoss()
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
        self.dataloaders = {phase: provider(self.fold, phase,
                                    batch_size=self.batch_size[phase],
                                    num_workers=self.num_workers)
                                    for phase in ["train", "val"]}

    def resume_net(self):
        self.resume_path = os.path.join(self.save_folder, "ckpt.pth")
        self.log("Resuming training, loading {} ...".format(self.resume_path))
        state = torch.load(self.resume_path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(state["state_dict"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.best_loss = state["best_loss"]
        self.start_epoch = state["epoch"] + 1

    def initialize_net(self):
        pass

    def forward(self, images, targets):
        images = images.to(self.device)
        targets = targets.view(-1, 1).to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, targets)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(2, phase, epoch, self.save_folder)
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
            if phase == "train":
                loss.backward()
                self.optimizer.step()
            running_loss += loss.item()

            #pdb.set_trace()
            outputs = torch.sigmoid(outputs).detach()
            meter.update(targets.cpu(), outputs.cpu())
            if iteration % 500 == 0:
                iter_log(phase, epoch, iteration, total_iters, loss, start)
                #break
        epoch_loss = running_loss / total_images
        epoch_log(phase, epoch, epoch_loss, meter, start)
        torch.cuda.empty_cache()
        return epoch_loss


    def train(self):
        t0 = time.time()
        for epoch in range(self.start_epoch, self.num_epochs):
            t_epoch_start = time.time()
            self.iterate(epoch, 'train')
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

            torch.save(state, self.ckpt_path)
            val_loss = self.iterate(epoch, 'val')
            self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                self.log("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, self.model_path)
            #if epoch and epoch % 2 == 0:
            copyfile(self.ckpt_path, os.path.join(self.save_folder, "ckpt%d.pth" % epoch))
            print_time(t_epoch_start, 'Time taken by the epoch')
            print_time(t0, 'Total time taken so far')
            self.log("\n" + "=" * 60 + "\n")


if __name__ == '__main__':
    fold = 0
    model_trainer = Trainer(fold=0)
    model_trainer.train()

