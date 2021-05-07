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

each level can use different LR,at low level,we can use the lower LR,because they already trained,they have pretrained weights,useful for nearly any task,no need to change so much,at higher level,the pretrained weights is for   much more complex concepts, like "eye" and "sunset," which might not be useful in your task at all,use a faster lr to train them  
main point:  
use a lower learning rate for the early layers of the neural network, and a higher learning rate for the later layers (and especially the randomly added layers)

### Basic for the slice

```py
arr=list(range(10))
myslice = slice(5)
arr[myslice]  
# [0, 1, 2, 3, 4]
myslice = slice(1,5)
arr[myslice]  
# [1, 2, 3, 4]
myslice = slice(0,5,2)
arr[myslice]  
# [0, 2, 4]
```

```py
# lr_max=slice(1e-6,1e-4)
# mean lowest LR is 1e-6,the other layers will scale up to 1e-4
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fit_one_cycle(3, 3e-3)
learn.unfreeze()
learn.fit_one_cycle(12, lr_max=slice(1e-6,1e-4))

# epoch train_loss valid_loss error_rate time
# 0 1.131566 0.361410 0.111637 01:06
# 1 0.544027 0.264487 0.086604 01:06
# 2 0.316729 0.248465 0.083221 01:07
# epoch train_loss valid_loss error_rate time
# 0 0.256258 0.242825 0.085250 01:11
# 1 0.242427 0.238632 0.080514 01:11
# 2 0.233899 0.233360 0.083221 01:11
# 3 0.217217 0.217414 0.075778 01:11
# 4 0.189038 0.217263 0.070365 01:11
# 5 0.181181 0.207588 0.069012 01:11
# 6 0.158933 0.208005 0.070365 01:11
# 7 0.148363 0.205170 0.068336 01:11
# 8 0.135392 0.203676 0.069012 01:12
# 9 0.122220 0.203666 0.065629 01:11
# 10 0.130100 0.200204 0.065629 01:11
# 11 0.119578 0.205134 0.069689 01:11
```

Now the fine-tuning is working great!

we can see the loss chnage

```py
# plot the loss change
learn.recorder.plot_loss()
```

![pLoss](/img/ai_t/t1/p_loss.PNG)

the training loss keeps getting better and better. But notice that eventually the validation loss improvement slows, and sometimes even gets worse! This is the point at which the model is starting to over fit. In particular, the model is becoming overconfident of its predictions. But this does not mean that it is getting less accurate, necessarily. Take a look at the table of training results per epoch, and you will often see that the accuracy continues improving, even as the validation loss gets worse. In the end what matters is your accuracy, or more generally your chosen **metrics**, **not the loss**. The loss is just the function we've given the computer to help us to optimize.

## Number of Epochs

choose the number of epoch that you willing to wait,than watch the above picture, if you see that the metric are still getting better even in your final epochs, then you know that you have not trained for too long.

## Deeper Architectures

a model with more parameters(depper) can model your data more accurately.  

This is why, in practice, architectures tend to come in a small number of variants. For instance, the ResNet architecture that we are using in this chapter comes in variants with 18, 34, 50, 101, and 152 layer, pretrained on ImageNet. A larger (more layers and parameters; sometimes described as the "capacity" of a model) version of a ResNet will always be able to give us a better training loss, but it can suffer more from overfitting, because it has more parameters to overfit with.  

the other problem is,depper, will use more GPU RAM,an duse more time  

nearly all current NVIDIA GPUs support a special feature called**tensor cores** that can dramatically speed up neural network training, by 2-3x. They also require a lot less GPU memory. To enable this feature in fastai, just add to_fp16() after your Learner creation (you also need to import the module).  

You can't really know ahead of time what the best architecture for your particular problem is—you need to try training some. So let's try a ResNet-50 now with mixed precision:  

```py
  from fastai.callback.fp16 import *
learn = cnn_learner(dls, resnet50, metrics=error_rate).to_fp16()
learn.fine_tune(6, freeze_epochs=3)
# epoch train_loss valid_loss error_rate time
# 0 1.279959 0.309704 0.102842 01:05
# 1 0.590101 0.312733 0.101489 01:05
# 2 0.447781 0.294772 0.088633 01:05
# epoch train_loss valid_loss error_rate time
# 0 0.274948 0.280899 0.085250 01:07
# 1 0.299947 0.331522 0.089310 01:07
# 2 0.251186 0.292205 0.084574 01:07
# 3 0.159606 0.241466 0.068336 01:07
# 4 0.083857 0.210775 0.060893 01:07
# 5 0.054267 0.210627 0.060893 01:06
```

try small model first,than try big model
