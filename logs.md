# Logs for the competition

# ADPOS diabetic retina

# Models on training:


### 8 Jul

Tried with different class weights, augmentation techniques, bran's preprocessing of gray images, removing the bad training examples (duplicates with conflicting diagnosises), tried resnext, resnet, densenet, its variants, nothing worked. the model did improve a lot on train/val sets but failed to generalize on test set. Today I also changed models from Cross-entropy to BCE loss per output neuron.
Target: 3 = [1, 1, 1, 0, 0]; prediction: [1, 1, 0, 0, 1] => 1
Worked good on train/val but failed on the test set :(
I also increased the val set from 10% to 20% this time.


### 9 Jul

12 midnight, I think it’s more about the data and the hyper-parameters than the model, just select a model, say resnext101_32x4d (it has the highest score so far) go for hyper-paramerter tuning and try to understand what’s going wrong instead of wildly trying to do random experiments leading to disinterest in the problem.

I’ve created new npy dataset data/npy_gray/ with only imread, BGR2GRAY, resize, reshape, repeat, save. Have put the model on training.
I need an excel sheet with all performance metrics for each model which I’m training.
And if there is some change in the datapipeline it is atmost import to mention is in detail in the sheet and not only that, the submission.py also needs to be updated accordingly.

Implemented get_best_threshold for val phase during training, it will be saved with every checkpoint :), now in submisison.py just load the best threshold for the checkpoint you wanna use that threshold gave the best results on the val set during training, use that threshold to predict the test set. If we choose the model checkpoint on the basis of it's loss on val set why can't we choose classification threshold on the basis of that val set? :)

I've created a npy file with only read, bgr2rbg, resize, will be training the model on this dataset, using modified class weights for data sampling kinda close to 1 for each class. Don't wanna disturb the data distribution as the dataset is very small.

didn't help!



`9-7_densenet121_fold0_rgb_ext`: Training the model on external data, class weights to 1, 1.5, 1, 1.5, 1.5, total folds =10


### 9 Jul

9 Jul: `weights/9-7_densenet121_fold0_rbg` to be trained on npy_rgb with class weights [1, 1.5, 1, 1.5, 1.5] with best threshold on val set being saved with each checkpoint. *The submission.py* needs to be modified accordingly. Gotta remove the best_thresshold function from there.

* `weights/9-7_densenet121_fold0_rbg_ext` : same as above, trained on npy_rgb of external dataset, the one from previouss competition, was performing poorly on original train set, total_folds=10

### 10 Jul

* 10 Jul: `weights/10-7_densenet121_fold0_rgb` : trained with val set sanctity on original dataset, total_folds = 5: LB: 0.66
* 10 Jul: `weights/10-7_densenet121_fold0_rgb_cw1` : with classweights 1, 1.3, 1, 1.3, 1, as the previous model is getting biased towards class 4,

FUCK! the sampler was None all the while!!!!!!!!!!!
moved class weights to Trainer class, if class_weights is None, the shuffle will be True, else there there will be weighted sampler.

retraining the cw1 model, will check in teh morning, also submit the latest test_prediction kernel on kaggle, it's made the predictions at ckpt20.pth, gotta compare it with ckpt30.pth and model.pth

So, I made the submissions. model.pth (at epoch 43) has 0.667, ckpt30.pth : 0.65, ckpt20.pth 0.65

* 10 Jul: `weights/10-7_densenet121_fold0_bengrahms` : without rotate, with transpose, with ben grahms' preprocessing, submissions.py modified accordingly: LB: 0.625
* 10 Jul: `weights/10-7_densenet121_fold0_bengrahmscolor` : same as above, with color of bengrahms method

I'm color with bengrahms for the external data also, will be saving the npy files in data/train_images/npy_bengrahm_color
GO for RESNEXT101 models

I think there *is* some data leak, the plots of val set are too similar to train sets, I've removed validation part from training script, I'm training a new version 10-7_densenet121_fold0_bengrahmscolortest similar to previous one, I'll compare the model performance on the validation set.

*****
THERE'S SOMETHING FISHY, there's some leakage in train/val training i dunno :(:(:(

*****
So looks like the change in the val qwk (a lil bit 0.02) is because of stochasticity of the model, as the train dataloader has weighted random sampling and random augmentation etc, I've changed model.train(phase='train') to model.train() and model.eval() for respective cases let's see what happens: So, nothing fishy!

10:43PM: Now on the best model will be saved on the basis of validation qwk score instead of val loss.
new files created, with all data : external + competition one  check the files section of this log

* 10 Jul: `weights/10-7_densenet121_fold0_bengrahmscolorall` : training on all data npy (data/train_images/npy_bengrahm_color and data/train_all.csv) with equal class weights with Transpose, Flip, and Random Scale: LB: 0.63

### 11 Jul

* 11 Jul: `weights/11-7_resnext101_32xd_fold0_bengrahmscolorall` : same as above with resnext101 model, batch size reduced to 16, class weights [1, 1.2, 1, 1.2, 1.2]

I gotta continue from my best performing model, I've to develop on top of it from now on. I've already wasted a lot of time in random hit and trial experiments
OKAY, LISTEN:

I will copy the augmentation, preprocessing configuration from the so-far-best-performing-model (0.70 one) and reproduce the results. FIRST. THEN, I'll add 50% new samples from external data and see the results.

So, I'm training a ResNeXt model with top 5k of train_all, (3.5k of train.csv and rest from ext), it's getting biased towards 2 and 3, retraining with class weights [1.5, 1, 1.5, 1, 1]
wrote test.py to generate test predictions for each ckpt, boi there's hell lot of variation in between the epochs. We gotta use tta for sure. 4 augmentations, batch size 8, takes 1:04 minutes for each predictions. Ok, so after using tta, there's huge variation in predictions even in adjacent checkpoints, is it the external dataset which is the cause?
tta predictions:
in order of epoch number, classes, number of predictions per class:
epoch 18 (array([0, 1, 2, 3, 4]), array([343,  90, 928, 435, 132]))
epoch 19 (array([0, 1, 2, 3, 4]), array([ 370,   95, 1022,  342,   99]))
epoch 20 (array([0, 1, 2, 3, 4]), array([310, 105, 899, 431, 183]))
epoch 21 (array([0, 1, 2, 3, 4]), array([401,  90, 792, 379, 266]))
epoch 22 (array([0, 1, 2, 3, 4]), array([ 406,  107, 1061,  283,   71]))
epoch 23 (array([0, 1, 2, 3, 4]), array([ 351,  114, 1103,  262,   98]))
epoch 24 (array([0, 1, 2, 3, 4]), array([ 347,  106, 1150,  249,   76]))
epoch 25 (array([0, 1, 2, 3, 4]), array([ 324,   95, 1067,  286,  156]))
epoch 26 (array([0, 1, 2, 3, 4]), array([ 317,   98, 1021,  332,  160]))
epoch 27 (array([0, 1, 2, 3, 4]), array([ 400,  115, 1289,  118,    6]))
epoch 28 (array([0, 1, 2, 3, 4]), array([ 432,  138, 1221,  128,    9]))
epoch 29 (array([0, 1, 2, 3, 4]), array([ 369,  107, 1036,  291,  125]))
epoch 30 (array([0, 1, 2, 3, 4]), array([ 416,  126, 1242,  131,   13]))
epoch 31 (array([0, 1, 2, 3, 4]), array([302, 107, 998, 327, 194]))
epoch 32 (array([0, 1, 2, 3, 4]), array([ 340,   95, 1066,  297,  130]))
epoch 33 (array([0, 1, 2, 3, 4]), array([ 345,   90, 1046,  297,  150]))

chose epoch 22 ckpt, with tta got : (array([0, 1, 2, 3, 4]), array([ 405,  106, 1058,  291,   68])) LB: 0.704 (highest so far)
without tta: (array([0, 1, 2, 3, 4]), array([ 414,  115, 1037,  286,   76])) LB: 0.693

LISTEN: don't fall for oh let me make this BIG change quickly and train a model and see how it performs! it's BS. Understand what you have, get the best out of it, ANALYSE, ask WHY?, make small changes at a time, keep track of your shit.

Okay, so the question is, why is there so much of variation in the predictions on the test set? does that happen in the train set too? let me check it out.

If variation in train set persists: then why there's no variation in losses? or in qwk plots?
Okay, the variation occurs due to abrupt variation in the best_threshold, now the question is why is best threshold varying this much?
Let me generate test predictions for threshold 0.5

* KEEP UP WITH THE DISCUSSION FORUM FOR ANY COMPETITION YOU ARE PARTICIPATING *

* One thing is for sure, people are able to achieve above 0.9 val qwk scores (I've also achieved it one or few times.) end up getting good scores on LB and sometimes don't

* weights/11-7_resnext101_32x4d_v0_fold0_bengrahmscolor/ new version of resnext101, with a BN, Lin(2048, 2048), Lin(2048, 4), with no class weights, on comp. data only, with Flip aug only, submitting ckpt26.pth with tta at best_threshold, LB: 0.5

### 12 Jul

The public test set has a totally different distribution as compared to train set, and then we have private test which is ~ 10 times the public test set. So, the importance of validation set truely representing the test set is more than ever as we are using thresholds optimised at validation set.

>> few questions *
1. What about pre-training the model on previous year's dataset and then training on the current one
2. using cropped version of the dataset

* `weights/12-7_resnext101_32x4d_v0_fold0_bgcold`: follows points 1. of above

I've added a new version of resnext101: resnext101_32x4d_v1, replacing the AdaptiveAvgPool2d with AdaptiveConcatPool2d which concats the output of AdaptiveAvgPool2d and AdaptiveMaxPool2d, apparently it performs better: check [here](https://docs.fast.ai/layers.html#AdaptiveConcatPool2d).

* `weights/12-7_resnext101_32x4d_v1_fold0_bgcold` : same as v0, only diff: AdaptiveConcatPool2d used instead of AdaptiveAvgPool2d, with Lin(4096), Lin(2048) Lin(5)


after 4 hours of experimentations: it turns out that it's kinda same.

LISTEN: YOUR BIAS TOWARDS PYTORCH IS NOT GOOD. Keras has evolved, fastai is great. Be open to use anything at hand.

So, there's this keras' starter kernel with simple imread, resize preprocessing, vertflip, horzflip, zoom aug, 0.15 val set, densenet121 base model with adaptiveavgpool, lin(5) model, with multi label, binary cross entropy, saving model based on val set kappa score and boom it scores 0.73 on LB with 0.5 threshold o_O. his model is same as my densenet121 only with dropout(0.5) I had 0.3, but heck, okay so he is using 0.5 as threshold for val set kappa calculation during training.

So, what I'm gonna do now is reproduce this result in pytorch, and see what the heck am I doing wrong.

### 13 Jul
I've been playing with densenet keras starter kernel on kaggle, key takeaways are that generalization is a big issue in this competition + the threshold we use for predicition,

** What about 5 thresholds optimised for each class?

So, now I've resnext101_32x4d_v0/1 pretrained on old data, gotta use them as basemodels to train on org data.


* `13-7_resnext101_32x4d_v1_fold0_bgpreold` : training on bgcolor with lr 1e-5, starting with model pretrained on bgcolor old data (weights/12-7_resnext101_32x4d_v1_fold0_bgcold/ckpt16.pth), val set best thresholding has been set to 0.5. total_folds = 7 (val set: ~15%), model being saved according to best loss, qwk is unstable metric to save upon. I won't be training it for long, can't afford overfitting. >> So, I submitted ckpt26.pth with threshold optimised with best performing public submission.csv: LB: 0.687 The model is getting biased towards class 0

The distribution of old data is different than present data, we can't just add them up and train a model based on that, so now I'm gonna sample out data from old data such that it resembles the original data distribution

Org data distribution, class wise, normalized: array([0.49290005, 0.10103768, 0.27280175, 0.05270344, 0.08055707])
old data distribution, class wise, normalized: array([0.73478335, 0.06954962, 0.15065763, 0.02485338, 0.02015601])

As it can be seen, old data is heavily biased towards class 0, the reason for my previous model's behaviour.

So, now I'm gonna re-pretrain the model on old data and then train it on new one, fingers crossed, yayye never fucking give up.

*** mistake ***
* `13-7_resnext101_32x4d_v1_fold0_bgcoldsamp`: same as 12-7_resnext101_32x4d_v1_fold0_bgcold, but trained on 20k samples, sampled according to dist. of current comp.'s dataset.
*
>> Forgot to update the lr to 1e-3, it is being trained on 1e-4 that's why it's so slow to plateau :( ek aadmi itni saari cheese kare to kare kaise?

### 14 Jul

I've stopped training at epoch 53, models' metrics are still improving (remember it was trained with 1e-4) but I want to check it's perf on test data right now.
I think 1e-4 is a good lr, though model is slow to learn, I've never achived qwk:train/val as ~0.90/0.95 on old data.

test predicting using 13-7_resnext101_32x4d_v1_fold0_bgcoldsamp is good, it would be wise to train on a mixed dataset with dist same as comp.'s dataset
I MADE A MISTAKE: this above ^ model was trained on new data only with 20k sampled according to the actual distribution :facepalm:
despite the mistake, for which I wasted so much time, a key takeaway is 1e-5 is better lr than 1e-4, I achieved 0.95 on val set

*** renaming 13-7_resnext101_32x4d_v1_fold0_bgcoldsamp to bgc20ksamp, sorry

LB: 0.63 for ckpt51.pth, th: 0.65, val qwk: 0.95, train qwk: 0.90

** BUG **: the 20k images were sampled with replace=True, so the val set was not disjoint to train set :( :(, so the previous model was overfitting.
and 1e-5 if painfully slow. use 1e-4

*** mistake over ***

* `14_7-resnext101_32x4d_v1_fold0_bgcoldsamp`: training on 8k images sampled from old data, sampled acc to dist of current data., replace=False, top_lr=1e-4
epoch 23: val loss > train loss but the val loss is still decreasing. choosing ckpt27, comparing this with the model trained on whole old data: this one has comparable qwk scores, but bad acc and loss.

* `14_7-resnext101_32x4d_v1_fold0_bgcpos` : starting with previous models ckpt27.pth, total_folds=6, training on bgc of current data. val loss > train loss at ckpt 12., submitted ckpt15.pth with optimised kappa (wrt submission.csv): LB: 0.71

The LB scores are generated on a private test set, so those public submission.csvs are not that useful, they don't represent the true picture..

Now, optimising 5 coefficient thresholds one for each class. Looks promising. Will be plotting the val qwk using optimized thresholds (5, one for each class) from now on. Still, best model will be saved using val loss, anyway I'm not using the best model for any of my submissions, gotta analyse the outputs of all the ckpts before deciding which one to choose and which thresholds to choose. Look at the optimised thresholds for ckpt13: [0.44824953 0.65033665 0.54202567 0.40581288 0.47455201], all these thresholds vary so much, it's imp to use class wise optimised thresholds.

ckpt13: val set optimised thresholds: [0.5091, 0.4916, 0.5216, 0.4965, 0.4971]: (array([0, 1, 2, 3, 4]), array([ 236,  261, 1194,  182,   55])): LB: 0.700
ckpt13: test submission optimised thres:  [0.44824953, 0.65033665, 0.54202567, 0.40581288, 0.47455201]: (array([0, 1, 2, 3, 4]), array([ 338,  168, 1103,  255,   64])): LB: 0.717

I have no idea what so ever.

Got few papers:

Automated Detection of Diabetic Retinopathy using Deep Learning: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5961805/
Grader Variability and the Importance of Reference Standards for Evaluating Machine Learning Models for Diabetic Retinopathy: https://www.aaojournal.org/article/S0161-6420(17)32698-2/fulltext

Got `messidor` dataset, 1200 images with improved image quality and reliable labels,
`npy_bengrahm_color`: now has *256x256* sized images from all three categories of datasets.

category 1: competition dataset, png images ~3.3k
category 2: messidor dataset, tif images: 1.2k
category 3: old competition dataset, jpeg images ~ 33k

`15-7_resnext101_32x4d_v0_fold0_bgcold256`: Will be pretraining a model on sampled cat 2 dataset, with 256 img size, resnext_v0. (not v1), total_folds=7,
Will be training a model pretrained on cat 2 dataset, on cat 1 and 3 combined dataset with img_size: 256
takeaway: 1e-4 for pretraing is too high, retraining with 3e-5.

### 15 Jul

Stopped previous training, cropping the retina in the images and centering the images using `load_ben_color_cropped` [prepare_npy.ipynb], training the previous model with cropped old images (tol=7): At epoch 28: val qwk ~= train qwk: 0.80, choosing ckpt32: train/val qwk: 0.84/0.82 >>>>>> mistake <<<<<<< Forgot to change `size` variable in `Trainer` :(, retraining with 256 image size doesn't have much affect on the plots though. Choosing ckpt


`15-7_resnext101_32x4d_v0_fold0_bgcpold256`: total_folds=6, train12.csv i.e., together with messidor dataset, starting with `weights/15-7_resnext101_32x4d_v0_fold0_bgcold256/ckpt30.pth`, with random rotate (180 degrees)
ep 28: acc: .77/.80, qwk: .87/.89, selecting ckpt21 and 31 for kernel submission with their val best threshold. Remember, all submissions are done with tta.

ckpt: LB, val qwk at optimized coeffs
21: 0.743, 0.8805, (array([0, 1, 2, 3, 4]), array([ 233,  248, 1102,  276,   69]))
24: 0.753, 0.8777,
25: 0.765, 0.8927 with random scale tta
26: 0.764, 0.8874, (array([0, 1, 2, 3, 4]), array([ 324,  187, 1114,  245,   58]))
26 submitted again with randomscale in tta: LLB: 0.767
28: 0.745, 0.8895, (array([0, 1, 2, 3, 4]), array([ 233,  248, 1102,  276,   69]))
31: 0.751, 0.8953, (array([0, 1, 2, 3, 4]), array([ 356,  228, 1062,  217,   65]))

ckpt25 submitted with randomscale in tta. LB: 0.765

** key takeaway is tta can slightly improve model performances

There's a lot of variation in the thresholds for each ckpt, as well as the total pred counts on the test data, How to select the best ckpt?

Added image meta data as late fusion to the keras model, train acc is 96%, val acc is 76%, model.predict is predict perfect [1, 0, 0, 0, 0], and val qwk is zero

* `16_7-resnext101_32x4d_v0_fold0_bgc256reg`: Regression mode, classes=1, MSELoss, initial thresholds = [0.5, 1.5, 2.5, 3.5], boi this thing is quick, CPU: 70% continuously, GPU 98%, 1:30 sec per epoch, why were the previous models low on CPU usage?

For MSELoss the targets should be flattened ([n, 1]), spent a lot of time figuring that out. model wasn't training otherwise.
ckpt 21: qwk : .85/.89, the predictions don't look promising.

* `16_7-resnext101_32x4d_v0_fold0_bdccold256reg`: pretraining resnext on sampled 256 sized ben grahm center cropped old data images in regression mode. ckpt28.pth looks optimum.

* `17_7-resnext101_32x4d_v0_fold0_bgcpcold256reg`: started with pretrained weights of previous model (ckpt28.pth), trained on train12.csv, ckpt22, ckpt 24, 25 look good acc to val best threshold and the predictions. Generated test predictions.

ckpt22: when I mistakenly submitted it with ckpt25's best thresholds I got 0.763, when submitted with it's own best threshold: I got 0.73

*Tip* : Don't follow train qwk, it's initial best thresholds [0.5, 1.5,..] which are used.


* `17_7-resnext101_32x4d_v0_fold0_bgcpcold256regft`: same as previous model, started with 1e-6 top lr, experimenting with fine tuning. Testing ckpt21: LB: 0.72
* `17_7-resnext101_32x4d_v0_fold0_bgc3t12v`: Old data as training, new data as validation till epoch 17, then fine tuned on new data only, with 1e-6, the test predictions are not promising.

* `17-7_resnext101_32x16d_v0_fold0_bgccold`: insta trained resnext 16d model. Training on sampled old data. Heck!! forgot to remove imagenet mean and std, argghh

#### 18 July

* `18-7_resnext101_32x16d_v0_fold0_bgccold`: same as before with mean=0, std=1, lr=1e-5
It's taking a lot of time, setting up Comp Unit's GPU cluster.

* `18-7_resnext101_32x16d_v0_fold0_bgccpold`: Fine-tuning previous model on new data with lr=1e-6, model trained without removing duplicates (they are not much, still model is learning sth out of nothing)
All ckpts from 4 to 20 look promising!,
Each model.pth file is 2.3GB!, choosing ckpt 11, 17 based on highest val qwk, ckpt 12 based on lowest val loss.
ckpt 4 is special, before this the model loss is kinda confined, after this val loss jumps high though val qwk keeps increasing. This one's preds are closest to .77 subs, choosing this.
dataset name: 18resnext10132x16dbgcpold256reg

Yahooooooooooooooooooo!!!!!!!!!!!!!!!!!!

ckpt, qwk, LB, test preds
ckpt
ckpt 4, .89/.88, 0.785, (array([0, 1, 2, 3, 4]), array([ 299,  174, 1188,  207,   60]))
ckpt 5, , 0.875
ckpt 7 Not submitted, the public test predictions are not promising.
ckpt 11, 0.784
ckpt 20, , 0.785


Time to get started with EfficientNets
* `18-7_efficientnet-b5_fold0_bgccold`: EfficientNet-b5 pretrained on imagenet, training on sampled old data, batch size 20/8 with amp (my god!! why wasn't I using it so far). lr = 1e-3
Choosing *ckpt19.pth* acc: 0.65/0.66, loss:0.39/0.42, qwk: 0.85/0.85
* `18-7_efficientnet-b5_fold0_bgccpold`:
The curves keep improving, is there any leak in the train/val?

#### 19 July

*Mistake* the model freezing code was not written for this model as well as the resnext101_32x16d model. (I used required_grad=False, instead of requires_grad -_-) Still the models trained well.

Submitting: ckpt 49, 30, 39

ckpt 49, LB: 0.792 ohhhhhh boiiiiiiiii

ckpt, qwk, LB, np.uniques
ckpt 49, 0.898/0.8949, 0.792, (array([0, 1, 2, 3, 4]), array([ 340,  131, 1183,  223,   51]))
ckpt 30, 0.890, 0.8959, 0.786, (array([0, 1, 2, 3, 4]), array([ 343,   59, 1262,  216,   48]))
ckpt 39, 0.892/0.8940,, 0.796 ,(array([0, 1, 2, 3, 4]), array([ 341,  154, 1159,  222,   52]))

val set qwk is not a good metric to choose submission from.

EfficientNet kept on improving, though slowly, untill I stopped it at ep 50
Holy Cow, why the f was I copy pasting best_thresholds to kaggle kernel when it was stored in model state -_-?

*Ensembling ideas*

* Select the ckpts, predict on training set, average the predictions, get optimized thresholds, predict on test set, average the predictions, use train set optimized thresholds. Implemented.

efficientnet fold1 test predictions very unstable. predictions are bad. not choosing any in the ensemble. preds saved in npy_test

*WARNING*
As you are selecting models based on test pred count, and not the loss, chances are that you'll end up down on final leaderboard. This public test set may not be the correct dist. of private test data.

Retraining fold1,2,3 all duplicates in train set only.

fold1: Earlier I had added duplicates in the df (before train/val split), this time only good duplicates in train df, and mistakenly bad duplicates were not removed from the df before train/val split, plots are little bad compared to previous one obviously because the previous model had train val leak.
retraining: Without bad duplicates, duplicates in train set only, with lr 1e-4, (as 1e-5 is way to low, model takes 50 epochs to converge)
The model plots are showing significant improvement compared to previous models.
The lr was reduced to 1e-5 at ep 12, 1e-6 at ep 16, 1e-7 at ep 20

submitted fold1, ckpt 10: LB: 0.812 *MIND BLOWN*

I think 1e-5 is way too high lr, testing 3e-5 for fold2

meanwhile creating bengrahms color cropped 300 sized images at `data/train_images/bgcc300/`

#### 20 July

So, I gotta analyse these folds, fold 1 and 2 were trained without bad dups, and good dups in train set. That's why loss plots of fold 1 and 2 look reasonable.
I'm retraining fold 0 and 3 this with train/val sanctity. calling it `19-7_efficientnet-b5_fold0_bgccpold` with lr 3e-5

So, I've trained fold 0, 1, 2, 3 for efficientnet-b5 models (re: `19_7-efficientnet-b5_fold*_bgccpold`)
remember: fold1's ckpt10 got me to LB: 0.812, this 4 model ensemble can do wonders. What about instead of train set optimization of thresholds, we optimize them on combined val set of the four folds.

fold0: val loss> train loss after ep 3, but val qwk and acc keep increasing till ep 20, choosing 20. (lr 3e-5)
fold1: val loss> train loss after ep 10, same with other two, choosing ckpt 10. (lr: 1e-4)
fold2: vloss>tloss after ep 20, qwk and acc improve till ep 30, choosing 30. (lr: 3e-5)
fold3: plots don't improve after ep 12, choosing ep 15. (lr: 3e-5)

submissions

fold 1, ckpt10, (array([0, 1, 2, 3, 4]), array([ 331,  195, 1145,  193,   64]))
*same above model when submitted again, got 0.809, even after tta model predictions are +- 0.03 on LB*
fold 1, ckpt12: LB: 0.779
thresholds optimized at the overall val set are giving bad results, prob because model are too confident on other's val sets, leading to overfitted ensemble. base thresholds (0.5, 1.5, 2..5, 3.5) looks great.
submitting ensemble of f0,1,2,3 at base threshold submitted at 3:50PM, got results at 5:40 PM ~ 2 hours. LB: 0.80

*Analysing ckpt10 on train set*

Model is good with class 0, extremely good., 0.99 sensitivity, specificity, precision.
*This is a multiclass confusion matrix, TNR will be higher for almost every class, it's not a good metric to go for. Class wise sensitivity (TPR) and precision (PPV) are reliable metrics here*
It is over confident with class 2, a lot of class 1, 3, and 4 preds go to class 2.

TPR:
  0: 0.99,
  1: 0.55,
  2: 0.92,
  3: 0.45,
  4: 0.56

Precision:
  0: 0.99,
  1: 0.76,
  2: 0.75,
  3: 0.49,
  4: 0.89

Class 0: Model's good. very good.
Class 1: Model needs to see more examples.
Class 2: Model is getting overconfident. (Because the sampled old data had 30% examples from class 2)
Class 3: Model needs more examples.
Class 4: Model needs more examples.

'TPR Macro': 0.6984477516938137,
'PPV Macro': 0.781013449615591,

This is not good. You may be getting good results on LB, but remember this underperforming and biased model may cost you in the private LB.

fml, deleted logs of fold1 (everything, tb, obj files) *and* fold2 (only tb) by mistake using rm -r -_- -_-

So, gotta retrain on old data with better sampling. Model needs to see more of class 1, 3, and 4. Earlier sampling ratios:

0: 0.5,
2: 0.28,
1: 0.11,
4: 0.09,
3: 0.06

Training on old data as train set and new data as val set, with class weights [1, 2, 1, 2, 2],

`20-7_efficientnet-b3_fold0_bgcco300`: EfficientNet-b3 model with  bgcc image size 300, old data as training set, class weights [1, 2, 1, 2, 2], lr: 5e-5,

*Bugfix* model._fc.parameters() are two tensors (w and b), we need to set their parameters' requires_grad not model's.
Use num_workers=12; at num_worker=8, batch size=32: gpu mem ~7.5GB, but utilization ~90%, CPU utilization ~66%.
ep 12 looks relatively optimum. TPR: 0.68/0.65, PPV: 0.68/0.65
it's failing at class 1, not sure why. Though quite comparable with `18-7_efficientnet-b5_fold0_bgccold` model

#### 21 July

finetuning:
`21-7_efficientnet-b3_fold0_bgccpold`: class_weights: 1, 1.5, 1, 1.5, 1.5; num_workers=12, without bad duplicates, with good ones in train set. Choosing ckpt14 for submission.
model starts overfitting after ep 10, submitted ckpt14, ckpt12. The class 4 seems to be underfitted.
ckpt10 test prediction reg kernel output is ready, just submit in the morning. Predictions look bad, 400 for class 1.

`logger.py` was logging micro metrics in overall stats so far.

All experiment metrics for submitted ckpts will be logged in APTOS Google Sheet from now on.
After starting each experiment, I'll commit the code with the name of that experiment. It is no more a generic baseline model on github. It's code of aptos-blindness-detection competition

* `21-7_efficientnet-b5_fold1_bgccpo300`: started with `18-7_efficientnet-b5_fold0_bgccold`'s ckpt19 model, and fine tuning on 300 sized bgcc images. lr: 3e-5
lr reduced by 10 at ep 23, selecting 20 for submission.


As all models are using bgcc will drop this keyword from ext_text in model names.


# Questions and Ideas:

* A new experiment can be this: pretrain on a balanced external data, then use that model on to the original data,  what's say?
* ** What about analysing the images in training data where the model is failing, the ones which are good enough but model completely fails to recognize, we can give them more weights in the data sampler, what other techniques can be used?

* Train on old, use new data as validation set!!
* What about using Adam?, started using it in Resnext101_32x16d models
* People have had successes with 320 image size.
* People are talking about resizing such that aspect ratio remains intact. Hmmm.
* CV for resnext101_32x16d model
* multilabel stratified cross validation.
* go through this: https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/97860#579277
* https://www.kaggle.com/kosiew/rank-averaging-script
* Good coders are participating in multiple competitions at a time.
* Should the distance between the categories be same? like is mild DR equally spaced to Moderate as it is to No DR

# TODO:

[] Add kappa optimizer
[x] Insure val set and train set are disjoints, watch-out for duplicates!
[] Crop images in datapipeline, add external data too
[] Analyse the heatmaps of the basemodel of a trained model, I think AdaptiveAvgPooling is detrimental to the model learning,
[] checkout for duplicates in external data too
[] print all the hyperparameters and initial infos you can about the model training , it helps trust me

# Revelations:

* mpimg.imread returns normalized image in [0, 1], cv2.imread returns it in [0, 255] (255 as maximum pixel value)
* plt.imshow needs image array to be in [0, 1] or [0, 255]
* albumentations' rotate shits in the interpolated area.
* .detach() makes variable' grad_required=False, the variable still remains on CUDA, make as many computations as possible on CUDA, use .cpu() in rarest of the cases, or when you wanna change the tensors to numpy use .cpu().numpy() because it can't be directly changed to numpy variable without shifting it to host memory, makes sense.
* test.py with tta=False, takes about 2 mins for first predictions, about 16 seconds for subsequent predictions, boi now you know what pin_memory=True does.
* for tta you don't have to pass image from each augmentation and then take the average, one other approach is to predict multiple times and take average and as the augmentationss are random, you get whachyouwantmyboi.
* loc for pd series and iloc for pd df.
* The resume code had a bug, if you'd resume you'll start with base_lr = top_lr * 0.001, and if the start epoch was greater than say 10, it will remain the same.
* The public test data is 15% of total test data and is used for public LB scoring. The private test set (85% of total) Will be used for private LB scoring.
* Updated the dataset to a new version? Just reboot the kernel to reflect that update (no remove and add shit)
* First epoch may not have full utilization, next epoch it'll be full thanks to pin_memory.



# Things to check, just before starting the model training:

* train_df_name
* model_name
* fold and total fold (for val %)
* npy_folder_name for dataloader's __getitem__() function
* are you resampling images?
* self.size, self.top_lr, self.std, self.mean -> insta trained weights used so be careful
* self.ep2unfreeze
*



# Observations:

1. I used 96x96 image size with Resnet101 on 5Jul model, results were not good. Gotta increase the input image size
2. 143 Images in the training set have duplicates (compared with hashes), out of those 79 have duplicates with different diagnosis label. Test set has 8 duplicates.
3. If GPU utilization is ~98% you can't help it, it's not the CPU which is the bottelneck here.
4. As I'm removing only bad ones from dataset, and there are still many duplicates in the there, so it is possible that those duplicates are distributed in train-val set, make sure all those duplicates are either in val set or train set!!!!!!!!!!!!!!!!!!!
5. So, now after that we are choosing the threshold by get_best_threshold function, the predictions are coming out to be in [1, 1, 1, 0, 0] manner like no 0 and then 1, no gap in between! I don't know how! you won't find any output to be [1, 1, 0, 1, 0] or like something similar.
6. The val qwk is better than train qwk in the qwk plot, because val qwk is threshold optimized, and train qwk uses 0.5

# NOTES:

1. Previous competition data: https://www.kaggle.com/tanlikesmath/diabetic-retinopathy-resized
2. Model training slow? (GPU utilization low?), speed up the dataloader, the __getitem__() is the culprit, too much preprocessing before transformation? save the damn thing in npy files bro --> 95% + utilization of GPU
3.



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
* data/train_old.csv: contains id_code and diagnosis of external data only
* data/train12.csv: train.csv and train_messidor.csv combined, i.e., category 1 and category 2 dataset labels combined
* data/train_meta.csv: train images shapes and related information
* data/test_meta.csv: test images' shapes and related information, though I need to write shape extraction functions in the inference scripts also as kernel is run on hidden test data.
* data/train32.csv: old + messidor data
*


# remarks shortcut/keywords

* bengrahmscolor: bengramhms color images on orginal data only
* bgcold: bengrahms color images on old data only
* bgpreold: bengrahms color images, model pre-trained on old data
* bgc20ksamp: 20k images sampled from org dataset accord to its distribution.
* bgcoldsamp: same as above, with sampled data, according to the dist of org data
* bgcpos: using ben grahms color images, with model *p*retrained on *o*ld data *s*ampled acc to dist of present data.
* bgcold256: trained on old data with ben grahm color image size 256
* bgcpold256: using model pretrained on bgcold256, with image size 256
* bgcc256reg: ben grahm cropped images, size = 256, regression model
* bgccold256reg: ben grahm cropped images, size = 256, regression model, training on sampled old data
* bgc3t12v: with bgc, with cat 3 as train set, cat 1 and 2 as validation set



# Experts speak

Insightful comments by poeple

* Do we need to use the same image size as used by pretrained models.
if you are using almost any pre-trained model the size does not have to match the size model has been trained with. It would only be useful for imagenet like images where different size would mean different objects scale, but for completely different domain like this competition, I don't see any benefit in sticking to the original image size.

* Label smoothing:
https://towardsdatascience.com/label-smoothing-making-model-robust-to-incorrect-labels-2fae037ffbd0

 only pytorch not keras can make deterministic (training) results.

A good ensemble contains high performing models which are less correlated., each fold val set should be unique.







EXTRAS:

EfficientNet params:
  # (width_coefficient, depth_coefficient, resolution, dropout_rate)
  'efficientnet-b0': (1.0, 1.0, 224, 0.2),
  'efficientnet-b1': (1.0, 1.1, 240, 0.2),
  'efficientnet-b2': (1.1, 1.2, 260, 0.3),
  'efficientnet-b3': (1.2, 1.4, 300, 0.3),
  'efficientnet-b4': (1.4, 1.8, 380, 0.4),
  'efficientnet-b5': (1.6, 2.2, 456, 0.4),
  'efficientnet-b6': (1.8, 2.6, 528, 0.5),
  'efficientnet-b7': (2.0, 3.1, 600, 0.5),


General Lessons:
* Don't use full words in model folder names, use f0, instead of fold0
