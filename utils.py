import os
import pdb
import cv2
import time
import torch
import scipy
import logging
import traceback
import numpy as np
from datetime import datetime

# from config import HOME
from tensorboard_logger import log_value, log_images
from torchnet.meter import ConfusionMeter
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import cohen_kappa_score
from matplotlib import pyplot as plt
from pycm import ConfusionMatrix

plt.switch_backend("agg")


def logger_init(save_folder):
    mkdir(save_folder)
    logging.basicConfig(
        filename=os.path.join(save_folder, "log.txt"),
        filemode="a",
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    console = logging.StreamHandler()
    logger = logging.getLogger(__name__)
    logger.addHandler(console)
    return logger


def to_multi_label(target, classes):
    """[0, 0, 1, 0] to [1, 1, 1, 0]"""
    multi_label = np.zeros((len(target), classes))
    for i in range(len(target)):
        j = target[i] + 1
        multi_label[i][:j] = 1
    return np.array(multi_label)


def get_preds(arr, num_cls):
    """ takes in thresholded predictions (num_samples, num_cls) and returns (num_samples,)
    [3], arr needs to be a numpy array, NOT torch tensor"""
    mask = arr == 0
    # pdb.set_trace()
    return np.clip(np.where(mask.any(1), mask.argmax(1), num_cls) - 1, 0, num_cls - 1)


def predict(X, coef):
    # [0.15, 2.4, ..] -> [0, 2, ..]
    X_p = np.copy(X)
    for i, pred in enumerate(X_p):
        if pred < coef[0]:
            X_p[i] = 0
        elif pred >= coef[0] and pred < coef[1]:
            X_p[i] = 1
        elif pred >= coef[1] and pred < coef[2]:
            X_p[i] = 2
        elif pred >= coef[2] and pred < coef[3]:
            X_p[i] = 3
        else:
            X_p[i] = 4
    return X_p.astype('int')


def compute_score_inv(thresholds, predictions, targets):
    predictions = predict(predictions, thresholds)
    score = cohen_kappa_score(predictions, targets, weights="quadratic")
    return 1 - score


class Meter:
    def __init__(self, phase, epoch, save_folder):
        self.predictions = []
        self.targets = []
        self.phase = phase
        self.epoch = epoch
        self.save_folder = os.path.join(save_folder, "logs")
        #self.num_classes = 5  # hard coded, yeah, I know
        self.best_thresholds = [0.5, 1.5, 2.5, 3.5] # initial guess

    def update(self, targets, outputs):
        '''targets, outputs are detached CUDA tensors'''
        # get multi-label to single label
        #targets = torch.sum(targets, 1) - 1 # no multilabel target in regression
        targets = targets.type(torch.LongTensor)
        outputs = outputs.flatten() # [n, 1] -> [n]
        # outputs = torch.sum((outputs > 0.5), 1) - 1

        #pdb.set_trace()
        self.targets.extend(targets.tolist())
        self.predictions.extend(outputs.tolist())
        # self.predictions.extend(torch.argmax(outputs, dim=1).tolist()) #[2]

    def get_best_thresholds(self):
        '''Epoch over, let's get targets in np array [6]'''
        self.targets = np.array(self.targets)

        if self.phase == "train":
            return self.best_thresholds

        """Used in the val phase of iteration, see [4]"""
        self.predictions = np.array(self.predictions)
        simplex = scipy.optimize.minimize(
            compute_score_inv,
            self.best_thresholds,
            args=(self.predictions, self.targets),
            method="nelder-mead",
        )
        self.best_thresholds = simplex["x"]
        print("Best thresholds: %s" % self.best_thresholds)
        return self.best_thresholds

    def get_cm(self):
        #pdb.set_trace()
        thresholds = self.best_thresholds
        self.predictions = predict(self.predictions, self.best_thresholds)
        cm = ConfusionMatrix(self.targets, self.predictions)
        qwk = cohen_kappa_score(self.targets, self.predictions, weights="quadratic")
        return cm, qwk


def plot_ROC(roc, targets, predictions, phase, epoch, folder):
    roc_plot_folder = os.path.join(folder, "ROC_plots")
    mkdir(os.path.join(roc_plot_folder))
    fpr, tpr, thresholds = roc_curve(targets, predictions)
    roc_plot_name = "ROC_%s_%s_%0.4f" % (phase, epoch, roc)
    roc_plot_path = os.path.join(roc_plot_folder, roc_plot_name + ".jpg")
    fig = plt.figure(figsize=(10, 5))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(fpr, tpr, marker=".")
    plt.legend(["diagonal-line", roc_plot_name])
    fig.savefig(roc_plot_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)  # see footnote [1]

    plot = cv2.imread(roc_plot_path)
    log_images(roc_plot_name, [plot], epoch)


def print_time(log, start, string):
    diff = time.time() - start
    log(string + ": %02d:%02d" % (diff // 60, diff % 60))


def adjust_lr(lr, optimizer):
    """ Update the lr of base model
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    for param_group in optimizer.param_groups[:-1]:
        param_group["lr"] = lr
    return optimizer


def iter_log(log, phase, epoch, iteration, epoch_size, loss, start):
    diff = time.time() - start
    log(
        "%s epoch: %d (%d/%d) loss: %.4f || %02d:%02d",
        phase,
        epoch,
        iteration,
        epoch_size,
        loss.item(),
        diff // 60,
        diff % 60,
    )


def epoch_log(log, tb, phase, epoch, epoch_loss, meter, start):
    diff = time.time() - start
    cm, qwk = meter.get_cm()
    acc = cm.overall_stat["Overall ACC"]
    tpr = cm.overall_stat["TPR Macro"] #[7]
    ppv = cm.overall_stat["PPV Macro"]
    cls_tpr = cm.class_stat['TPR']
    cls_ppv = cm.class_stat['PPV']

    print()
    log(
        "%s %d |  loss: %0.4f | QWK: %0.4f | ACC: %0.4f | TPR: %0.4f | PPV: %0.4f \n"
        % (phase, epoch, epoch_loss, qwk, acc, tpr, ppv)
    )

    cls_tpr = {x: "%0.4f" % y for x, y in cls_tpr.items()}
    cls_ppv = {x: "%0.4f" % y for x, y in cls_ppv.items()}

    log('Class TPR: %s' % cls_tpr)
    log('Class PPV: %s' % cls_ppv)
    log(cm.print_normalized_matrix())
    log("Time taken for %s phase: %02d:%02d \n", phase, diff // 60, diff % 60)

    # tensorboard
    logger = tb[phase]
    logger.log_value("loss", epoch_loss, epoch)
    logger.log_value("ACC", acc, epoch)
    logger.log_value("QWK", qwk, epoch)
    logger.log_value("TPR", tpr, epoch)
    logger.log_value("PPV", ppv, epoch)

    # save pycm confusion
    obj_path = os.path.join(meter.save_folder, f"cm{phase}_{epoch}")
    cm.save_obj(obj_path, save_stat=True, save_vector=False)

    return qwk


def mkdir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def save_hyperparameters(trainer, remark):
    hp_file = os.path.join(trainer.save_folder, "parameters.txt")
    time_now = datetime.now()
    augmentations = trainer.dataloaders['train'].dataset.transform.transforms
    # pdb.set_trace()
    string_to_write =  \
        f"Time: {time_now}\n" + \
        f"model_name: {trainer.model_name}\n" + \
        f"train_df_name: {trainer.train_df_name}\n" + \
        f"images_folder: {trainer.images_folder}\n" + \
        f"resume: {trainer.resume}\n" + \
        f"pretrained: {trainer.pretrained}\n" + \
        f"pretrained_path: {trainer.pretrained_path}\n" + \
        f"folder: {trainer.folder}\n" + \
        f"fold: {trainer.fold}\n" + \
        f"total_folds: {trainer.total_folds}\n" + \
        f"num_samples: {trainer.num_samples}\n" + \
        f"sampling class weights: {trainer.class_weights}\n" + \
        f"size: {trainer.size}\n" + \
        f"top_lr: {trainer.top_lr}\n" + \
        f"base_lr: {trainer.base_lr}\n" + \
        f"num_workers: {trainer.num_workers}\n" + \
        f"batchsize: {trainer.batch_size}\n" + \
        f"momentum: {trainer.momentum}\n" + \
        f"mean: {trainer.mean}\n" + \
        f"std: {trainer.std}\n" + \
        f"start_epoch: {trainer.start_epoch}\n" + \
        f"batchsize: {trainer.batch_size}\n" + \
        f"augmentations: {augmentations}\n" + \
        f"criterion: {trainer.criterion}\n" + \
        f"optimizer: {trainer.optimizer}\n" + \
        f"remark: {remark}\n"

    with open(hp_file, "a") as f:
        f.write(string_to_write)
    print(string_to_write)




"""Footnotes:

[1]: https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures

[2]: Used in cross-entropy loss, one-hot to single label

[3]: # argmax returns earliest/first index of the maximum value along the given axis
 get_preds ka ye hai ki agar kisi output me zero nahi / sare one hain to 5 nahi to jis index par pehli baar zero aya wahi lena hai, example:
[[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 0, 1, 1, 0], [0, 0, 0, 0, 0]]
-> [4, 1, 0, 0]
baki clip karna hai (0, 4) me, we can get -1 for cases with all zeros.

[4]: get_best_threshold is used in the validation phase, during each phase (train/val) outputs and targets are accumulated. At the end of train phase a threshold of 0.5 is used for
generating the final predictions and henceforth for the computation of different metrics.
Now for the validation phase, best_threshold function is used to compute the optimum threshold so that the qwk is minimum and that threshold is used to compute the metrics.

It can be argued ki why are we using 0.5 for train, then, well we used 0.5 for both train/val so far, so if we are computing this val set best threshold, then not only it can be used to best evaluate the model on val set, it can also be used during the test time prediction as it is being saved with each ckpt.pth

[5]: np.array because it's a list and gets converted to np.array in get_best_threshold function only which is called in val phase and not training phase

[6]: It's important to keep these two in np array, else ConfusionMatrix takes targets as strings. -_-

[7]: macro mean average of all the classes. Micro is batch average or sth.
"""
