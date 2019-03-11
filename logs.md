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
