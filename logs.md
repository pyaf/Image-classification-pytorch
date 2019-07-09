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



# ADPOS diabetic retina

Observations:

1. I used 96x96 image size with Resnet101 on 5Jul model, results were not good. Gotta increase the input image size
2. 143 Images in the training set have duplicates (compared with hashes), out of those 79 have duplicates with different diagnosis label. Test set has 8 duplicates.


# NOTES:

1. Previous competition data: https://www.kaggle.com/tanlikesmath/diabetic-retinopathy-resized
2. Model training slow? (GPU utilization low?), speed up the dataloader, the __getitem__() is the culprit, too much preprocessing before transformation? save the damn thing in npy files bro --> 95% + utilization of GPU
3.

8 Jul: Tried with different class weights, augmentation techniques, bran's preprocessing of gray images, removing the bad training examples (duplicates with conflicting diagnosises), tried resnext, resnet, densenet, its variants, nothing worked. the model did improve a lot on train/val sets but failed to generalize on test set. Today I also changed models from Cross-entropy to BCE loss per output neuron.
Target: 3 = [1, 1, 1, 0, 0]; prediction: [1, 1, 0, 0, 1] => 1
Worked good on train/val but failed on the test set :(
I also increased the val set from 10% to 20% this time.

9 Jul: 12 midnight, I think it’s more about the data and the hyper-parameters than the model, just select a model, say resnext101_32x4d (it has the highest score so far) go for hyper-paramerter tuning and try to understand what’s going wrong instead of wildly trying to do random experiments leading to disinterest in the problem.

I’ve created new npy dataset data/npy_gray/ with only imread, BGR2GRAY, resize, reshape, repeat, save. Have put the model on training.
I need an excel sheet with all performance metrics for each model which I’m training.
And if there is some change in the datapipeline it is atmost import to mention is in detail in the sheet and not only that, the submission.py also needs to be updated accordingly.

Implemented get_best_threshold for val phase during training, it will be saved with every checkpoint :), now in submisison.py just load the best threshold for the checkpoint you wanna use that threshold gave the best results on the val set during training, use that threshold to predict the test set. If we choose the model checkpoint on the basis of it's loss on val set why can't we choose classification threshold on the basis of that val set? :)

I've created a npy file with only read, bgr2rbg, resize, will be training the model on this dataset, using modified class weights for data sampling kinda close to 1 for each class. Don't wanna disturb the data distribution as the dataset is very small.

Some informations:

* data/train_images/npy * files are images after cv2.read, BGR2GRAY, resize (224), addWeighted gaussian blur, reshape, repeat, save

* data/train_images/npy_rgb * files are images after cv2.read, BGR2RGB, resize (224), save




# Models on training:

9 Jul: weights/9-7_densenet121_fold0_rbg to be trained on npy_rgb with class weights [1, 1.5, 1, 1.5, 1.5] with best threshold on val set being saved with each checkpoint. *The submission.py* needs to be modified accordingly. Gotta remove the best_thresshold function from there.

