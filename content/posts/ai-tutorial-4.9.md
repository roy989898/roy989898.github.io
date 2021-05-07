+++
title = "Ai Tutorial 4.9 SGD and Mini-Batches"
date = "2021-04-28T16:06:34+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch","SGD","batch","step"]
keywords = ["", ""]
description = ""
showFullContent = false
+++
[My Code](https://colab.research.google.com/drive/1rMfM4H92wklMLDydjnChmJMHoJ3OS6SL?usp=sharing)
[Source Code](https://colab.research.google.com/github/fastai/fastbook/blob/master/04_mnist_basics.ipynb)
we already have a SGD loss function,we can go to `Step`  
which is to change or update the weights based on the gradients. This is called an optimization step.

# basic Python

## string.ascii_lowercase

```py
string.ascii_lowercase
# abcdefghijklmnopqrstuvwxyz
```

# basic Pytorch

## enumerate

```py
#  L is something liek list
# enumerate() 函數用於將一個可遍歷的數據對象(如列表、元組或字符串)組合為一個索引序列，同時列出數據和數據下標，一般用在for 循環當中。
ds = L(enumerate(string.ascii_lowercase))
ds
# [(0, 'a'),(1, 'b'),(2, 'c'),(3, 'd'),(4, 'e'),(5, 'f'),(6, 'g'),(7, 'h'),(8, 'i'),(9, 'j')...]
```

# optimization step

![sgd_step](/img/ai_t/t1/sgd_step.PNG)

## why Mini-Batches

we can one item for 1 epoch,but this will be very slow,

### 1. single image size batch

if we ahve 256 picture,we predict 1 picture,tha we calculate the loss for the picture,than use the loss number to calculate the gradient,step the weight,next picture, `total 256 epoch`

### 2. 4 image size batch

we have 256/4= 64 bitch picture, we predict 4 picture at a time,we calcuate 4 loss for 4 picture,than use a loss number  to calculate 4 gradient number ,step the weight,next batch,`total 64 epoch`

So use mini btach more fast!!!!!!!!

## Other reason why Mini-Batches

another reason that use mini batch not calculating the gradient on individual data items is that, we nearly always do our training on an accelerator such as a GPU. These accelerators only perform well if they have lots of work to do at a time, so it's helpful if we can give them lots of data items to work on. Using mini-batches is one of the best ways to do this. However, if you give them too much data to work on at once, they run out of memory—making GPUs happy is also tricky!

## Use DataLoader to create batches

```py
coll = range(15)
dl = DataLoader(coll, batch_size=5, shuffle=True)
list(dl)

# [tensor([ 3, 12,  8, 10,  2]),
#  tensor([ 9,  4,  7, 14,  5]),
#  tensor([ 1, 13,  0,  6, 11])]
```
