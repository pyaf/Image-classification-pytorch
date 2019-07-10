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
3. If GPU utilization is ~98% you can't help it, it's not the CPU which is the bottelneck here.
4. As I'm removing only bad ones from dataset, and there are still many duplicates in the there, so it is possible that those duplicates are distributed in train-val set, make sure all those duplicates are either in val set or train set!!!!!!!!!!!!!!!!!!!
5. So, now after that we are choosing the threshold by get_best_threshold function, the predictions are coming out to be in [1, 1, 1, 0, 0] manner like no 0 and then 1, no gap in between! I don't know how! you won't find any output to be [1, 1, 0, 1, 0] or like something similar.


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

didn't help!



9-7_densenet121_fold0_rgb_ext: Training the model on external data, class weights to 1, 1.5, 1, 1.5, 1.5, total folds =10



# Files informations:

* data/bad_train_indices.npy: list of indices in the train.csv which have a duplicate in the training data and that duplicate has a different diagnosis. We don't want our model to train on these images.
* data/duplicates.npy: list of lists containing id_code's of duplicates in train.csv, i.e., all duplicates in train.csv
* data/dups_with_same_diagnosis.npy: list of indices in train.csv which belong to images which have duplicates in train_images and their diagnosis is same
* data/duplicate_test_ids: buggy, it has hashes of duplicate test images, gotta fix it
* data/train_images/npy * files are images after cv2.read, BGR2GRAY, resize (224), addWeighted gaussian blur, reshape, repeat, save
* data/train_images/npy_rgb * files are images after cv2.read, BGR2RGB, resize (224), save
* data/train_images/npy_bengrahm : files images, preprocesses according to ben grahm's preprocessing technique, read, bgr2gray, resize, addweighted, reshape, repeat, img size = 224
* data/train_images/npy_bengrahm_color : all data (internal + external) using bengrahm's color method, no cropping of retina, only 224 resizing
* data/train_all.csv: contains train df of all data (internal and external), train.csv of external appended to train.csv of internal data, so bad indices still hold correct, I gotta checkout for duplicates in external data also
*


# Models on training:

9 Jul: weights/9-7_densenet121_fold0_rbg to be trained on npy_rgb with class weights [1, 1.5, 1, 1.5, 1.5] with best threshold on val set being saved with each checkpoint. *The submission.py* needs to be modified accordingly. Gotta remove the best_thresshold function from there.

* weights/9-7_densenet121_fold0_rbg_ext: same as above, trained on npy_rgb of external dataset, the one from previouss competition, was performing poorly on original train set, total_folds=10

* 10 Jul: weights/10-7_densenet121_fold0_rgb: trained with val set sanctity on original dataset, total_folds = 5: LB: 0.66
* 10 Jul: weights/10-7_densenet121_fold0_rgb_cw1: with classweights 1, 1.3, 1, 1.3, 1, as the previous model is getting biased towards class 4,

FUCK! the sampler was None all the while!!!!!!!!!!!
moved class weights to Trainer class, if class_weights is None, the shuffle will be True, else there there will be weighted sampler.

retraining the cw1 model, will check in teh morning, also submit the latest test_prediction kernel on kaggle, it's made the predictions at ckpt20.pth, gotta compare it with ckpt30.pth and model.pth

So, I made the submissions. model.pth (at epoch 43) has 0.667, ckpt30.pth : 0.65, ckpt20.pth 0.65

* 10 Jul: weights/10-7_densenet121_fold0_bengrahms: without rotate, with transpose, with ben grahms' preprocessing, submissions.py modified accordingly: LB: 0.625
* 10 Jul: weights/10-7_densenet121_fold0_bengrahmscolor: same as above, with color of bengrahms method

I'm color with bengrahms for the external data also, will be saving the npy files in data/train_images/npy_bengrahm_color
GO for RESNEXT101 models

I think there *is* some data leak, the plots of val set are too similar to train sets, I've removed validation part from training script, I'm training a new version 10-7_densenet121_fold0_bengrahmscolortest similar to previous one, I'll compare the model performance on the validation set.

*****
THERE'S SOMETHING FISHY, there's some leakage in train/val training i dunno :(:(:(

*****
So looks like the change in the val qwk (a lil bit 0.02) is because of stochasticity of the model, as the train dataloader has weighted random sampling and random augmentation etc, I've changed model.train(phase='train') to model.train() and model.eval() for respective cases let's see what happens: So, nothing fishy!

10:43PM: Now on the best model will be saved on the basis of validation qwk score instead of val loss.
new files created, with all data : external + competition one  check the files section of this log

* 10 Jul: weights/10-7_densenet121_fold0_bengrahmscolorall: training on all data npy (data/train_images/npy_bengrahm_color and data/train_all.csv) with equal class weights with Transpose, Flip, and Random Scale


# TODO:

[] Add kappa optimizer
[x] Insure val set and train set are disjoints, watch-out for duplicates!
[] Crop images in datapipeline, add external data too
[] Analyse the heatmaps of the basemodel of a trained model, I think AdaptiveAvgPooling is detrimental to the model learning,
[] checkout for duplicates in external data too


# Revelations:

* mpimg.imread returns normalized image in [0, 1], cv2.imread returns it in [0, 255] (255 as maximum pixel value)
* plt.imshow needs image array to be in [0, 1] or [0, 255]
* albumentations' rotate shits in the interpolated area.
