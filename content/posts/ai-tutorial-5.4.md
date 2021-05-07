+++
title = "Ai Tutorial 5.4 Image Classification >2 types Improving Our Model"
date = "2021-05-07T14:42:55+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch","learning rate finder","freexing","epoch number"]
keywords = ["", ""]
description = ""
showFullContent = false
+++
[My Code
](https://colab.research.google.com/drive/1Rqum2194iz5nXH26PPoBMpKM71wQ4eYI?usp=sharing)

[Source Code
](https://colab.research.google.com/github/fastai/fastbook/blob/master/05_pet_breeds.ipynb#scrollTo=YOTrrdP7BuWd)

# Improving Our Model

we will explain a little bit more about transfer learning and how to fine-tune our pretrained model as best as possible, without breaking the pretrained weights.

## The Learning Rate Finder

if lr too small, many epochs to train our model,waste time,and every time we do a complete pass through the data, we give our model a chance to memorize it.also remember the validate data

set it very high frist,

```py
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1, base_lr=0.1)
# epoch train_loss valid_loss error_rate time
# 0 2.568456 6.223738 0.496617 01:07
# epoch train_loss valid_loss error_rate time
# 0 3.971391 2.541565 0.698917 01:12
```

the way to find the best LR:  
simple concept: use a very LR start,train a one mini-batch,> increase the LR by some percentage (e.g., doubling it each time),than repeat,until the loss gets worse, instead of better,This is the point where we know we have gone too far. We then select a learning rate a bit lower than this point. Our advice is to pick either:

1. One order of magnitude less than where the minimum loss was achieved (i.e., the minimum divided by 10)
2. The last point where the loss was clearly decreasing  

fastai will help you to find this 2 point Both these rules usually give around the same value

```py
# default start LR is 1e-3=10^-3
learn = cnn_learner(dls, resnet34, metrics=error_rate)
lr_min,lr_steep = learn.lr_find()

print(f"Minimum/10: {lr_min:.2e}, steepest point: {lr_steep:.2e}")
# Minimum/10: 1.00e-02, steepest point: 2.51e-03
```

`1e-3 mean 10^-3`

![sgd_LRFstep](/img/ai_t/t1/lrf.PNG)
for the picture,we can seeif LR > 1e-1,the loss increase,but 1-e-1 too high,becasu already leave the loss decrease phase  
we use 3e-3 at here(follow the book),we still can use 8.32e-03 and 2.09e-03

## Unfreezing and Transfer Learning

what is transfer learning??? We saw that the basic idea is that a pretrained model, trained potentially on millions of data points (such as ImageNet), is fine-tuned for some other task.  

Our challenge when fine-tuning is to replace the random weights in our added linear layers with weights that correctly achieve our desired task (classifying pet breeds) without breaking the carefully pretrained weights and the other layers. There is actually a very simple trick to allow this to happen: tell the optimizer to only update the weights in those randomly added final layers. Don't change the weights in the rest of the neural network at all. This is called freezing those pretrained layers.

進行微調時，我們面臨的挑戰是在不破壞經過精心訓練的砝碼和其他層的情況下，用能夠正確完成我們期望任務（對寵物品種進行分類）的砝碼替換添加的線性層中的隨機砝碼。 實際上，有一個很簡單的技巧可以使這種情況發生：告訴優化器僅更新那些隨機添加的最終層中的權重。 完全不要更改神經網絡其餘部分的權重。 這稱為凍結那些預訓練的層。  

When we create a model from a pretrained network fastai automatically freezes all of the pretrained layers for us. When we call the fine_tune method fastai does two things:

Trains the randomly added layers for one epoch, with all other layers frozen.  
Unfreezes all of the layers, and trains them all for the number of epochs requested

try implement

First of all we will train the randomly added layers for three epochs, using fit_one_cycle
fit_one_cycle is the suggested way to train models without using fine_tune. We'll see why later in the book; in short, what fit_one_cycle does is to start training at a low learning rate, gradually increase it for the first section of training, and then gradually decrease it again for the last section of training.

```py
# here only train the randomly added layers
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fit_one_cycle(3, 3e-3)

# epoch train_loss valid_loss error_rate time
# 0 1.149184 0.357759 0.112314 01:07
# 1 0.516031 0.269226 0.082544 01:07
# 2 0.307812 0.237481 0.071719 01:07
```

```py
# unfreeze the model
learn.unfreeze()
```

run lr_find again to find the LR, because having more layers to train, and weights that have already been trained for three epochs, means our previously found learning rate isn't appropriate any more:  

```py
learn.lr_find()
```

![lr2](/img/ai_t/t1/lr2.PNG)

**important**!!!!!!we should not use the lr_steep at here,because our model has been trained already. Here we have a somewhat flat area before a sharp increase, and we should take a point well before that sharp increase—for instance, 1e-5. The point with the maximum gradient isn't what we look for here and should be ignored.  

```py
#  train all layer
learn.fit_one_cycle(6, lr_max=1e-5)

# epoch train_loss valid_loss error_rate time
# 0 0.245116 0.232571 0.071042 01:12
# 1 0.244692 0.223327 0.069689 01:12
# 2 0.214002 0.217773 0.068336 01:13
# 3 0.194007 0.214042 0.066306 01:12
# 4 0.180974 0.212813 0.067659 01:11
# 5 0.183777 0.215303 0.064953 01:12
```

The deepest layers of model might not need as high a learning rate as the last ones, so we should probably use different learning rates for those—this is known as using discriminative learning rates.

## Discriminative Learning Rates
