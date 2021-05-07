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