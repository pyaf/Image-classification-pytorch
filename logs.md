# Logs for the competition

8 Mar, se_resnext50_32x4d with no Maxpool, modified Linear with out_features=1
input size 112x112,

8Mar-9Mar se_resnext50_32x4d v2 model: with no Maxpool, pad, avg_pool, flatten, dropout(0.3), linear
top_lr = 1e-4, base_lr = 2*top_lr till first 10 epochs. input image size 96x96
self.top_lr = 7e-5 #4e-4 #0.00007 #1e-3
self.base_lr = self.top_lr * 0.001
self.momentum = 0.95
self.epoch_2_lr = {1: 2, 3: 5, 5: 2, 6:5, 7:2, 9:5} # factor to scale base_lr with

9Mar: se_resnext50 v3 model: no maxpool, adaptive avg pool, flatten, dropout, linear,
input image size of 96x96, top_lr=7e-5, base_lr as the previous one.

* It is better at 32 batch size for train&val , 64 will run with a slight decrease in time, taking up almost all of the GPU memory*

9Mar v3, fold 2 trained with top_lr 1e-4
9Mar v3, fold 3 to be trained with top_lr = 5e-4

11Mar nasnetamobile models with input 3x224x224 trained with RGB for first time.
with only last_linear layer modified, use get_model function with std, mean of imagenet ONLY for prediction


11Mar nasnetamobile_v2 implemented, with std mean of 0.5 each (as recommended by pretrainedmodels project), with adaptiveAvgPool2d, dropout, linear, top_lr = 7e-5, base_lr = top_lr * .001 ====>>>> really really bad performance. BECAUSE of not using same mean and std conf in submission script -_-

11Mar nasnetamobile_v2 trained on fold3 with top_lr = 7e-5, base_lr = top_lr * .01
