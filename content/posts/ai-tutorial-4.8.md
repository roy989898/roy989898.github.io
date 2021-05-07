+++
title = "Ai Tutorial 4.8 The MNIST Loss Function"
date = "2021-04-28T12:44:41+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch","MNIST","Loss Function","view","cat"]
keywords = ["", ""]
description = ""
showFullContent = false
+++
[My Code](https://colab.research.google.com/drive/1rMfM4H92wklMLDydjnChmJMHoJ3OS6SL?usp=sharing)
[Source Code](https://colab.research.google.com/github/fastai/fastbook/blob/master/04_mnist_basics.ipynb)
# _MNIST Loss Function_

## some basic python

### zip

```py
# zip
a=[1,2,3,4]
b=[5,6,7,8]
list(zip(a,b))
# [(1, 5), (2, 6), (3, 7), (4, 8)]
```

### create array

```py
[1]*4
# [1, 1, 1, 1]
```

```py
tensor([1]*4 + [0]*3)
# tensor([1, 1, 1, 1, 0, 0, 0])
```

## Some basic pytorch functions

### horizontal tensor to vertical tensors

```py
tensor([1]*4 + [0]*3)
# tensor([1, 1, 1, 1, 0, 0, 0])
```

```py
tensor([1]*4 + [0]*3).unsqueeze(1)
# tensor([[1],
#         [1],
#         [1],
#         [1],
#         [0],
#         [0],
#         [0]])
```

### torch.cat

connect two tensors together
<https://blog.csdn.net/qq_39709535/article/details/80803003>

```py
A=torch.ones(4,3) #2x3的张量（矩阵）                                     
A
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]])
```

```py
B=2*torch.ones(4,3)
B
# tensor([[2., 2., 2.],
#         [2., 2., 2.],
#         [2., 2., 2.],
#         [2., 2., 2.]])
```

```py
C=torch.cat([A,B])
C.shape
# torch.Size([8, 3])
```

### Tensor.view

PyTorch allows a tensor to be a View of an existing tensor. View tensor shares the same underlying data with its base tensor.  

把原先tensor中的數據按照行優先的順序排成一個一維的數據（這裡應該是因為要求地址是連續存儲的），然後按照參數組合成其他維度的tensor。比如說是不管你原先的數據是[ [[1,2,3],[4,5,6]]]還是[1,2,3,4,5,6]，因為它們排成一維向量都是6個元素，所以只要view後面的參數一致，得到的結果都是一樣的。比如，
example

```py
a=torch.Tensor([[[1,2,3],[4,5,6]]])
print(a.view(3,2))


# tensor([[1., 2.],
#         [3., 4.],
#         [5., 6.]])
```

### torch.randn

create a list of random numberless

```py
torch.randn(8)
# tensor([ 0.9912,  0.4679, -0.2049, -0.7409,  0.3618,  1.9199, -0.2254, -0.3417])

```

```py
torch.randn((8,1))
# tensor([[ 0.3040],
#         [-0.6890],
#         [-1.1267],
#         [-0.2858],
#         [-1.0935],
#         [ 1.1351],
#         [ 0.7592],
#         [-3.5945]])
```

### matrix multiplication

```py
A@B

```

![rt](/img/ai_t/t1/matrix_m.PNG)
For instance, row 1, column 2 (the orange dot with a red border) is calculated as a1,1∗b1,2+a1,2∗b2,2

## MNIST Loss Function

### Prepare the train data

#### connect the photo

```py
stacked_threes.shape
# torch.Size([6131, 28, 28])
```

```py
stacked_sevens.shape
# torch.Size([6265, 28, 28])
```

```py
train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)
train_x.shape
# torch.Size([12396, 784])
```

the above acode,we first connect the stacked_threes(each pixel present by 0-1 number) and
for each picture , orginal is respenct by a 2d tensor,(28*28),turn to 1d tensor 784
`view(-1, 28*28)` mean 28*28 column,-1 mean not specific the row number,just make it can fit the content,becasue we have
`6131+6265=12396`

#### add the tag for each photo

```py

# assign the tag to each image
# We need a label for each image. We'll use `1` for 3s and `0` for 7s:
train_y = tensor([1]*len(threes) + [0]*len(sevens)).unsqueeze(1)
train_x.shape,train_y.shape
# (torch.Size([12396, 784]), torch.Size([12396, 1]))
# train_X,12396 images,each image total 784 pixels
#train_y,12396 tag,because eachpicture 1 tag,1 tag inf in each tag
```

#### prepare the Pytorch need format

A Dataset in PyTorch is required to return a tuple of (x,y) when indexed. Python provides a zip function which, when combined with list, provides a simple way to get this functionality:

```py
dset = list(zip(train_x,train_y))
x,y = dset[0]
x.shape,y
# x is  the image,t is the tag

# (torch.Size([784]), tensor([1]))
```

### Prepare the valid data

```py
valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_3_tens) + [0]*len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x,valid_y))
```

### create random init param

```py
def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()
```

```py
# weneed a vertical 2d array,show we need (28*28,1),not only 28*28
weights = init_params((28*28,1))
# weights
```

The function `weights*pixels` won't be flexible enough—it is always equal to 0 when the pixels are equal to 0 (i.e., its *intercept* is 0). You might remember from high school math that the formula for a line is `y=w*x+b`; we still need the `b`. We'll initialize it to a random number too:

```py
bias = init_params(1)
# why??????
bias
```

### Predict a image

```py
(train_x[0]*weights.T).sum() + bias
```

we can use a foor loop to calculate all the image pred,but this is slow  
so we use matrix multiplication  ,more fast can use GPU
we suggest you take a look at the Intro to Matrix Multiplication <https://www.youtube.com/watch?v=kT4Mp9EdVqs&ab_channel=KhanAcademy>

![rt](/img/ai_t/t1/matrix_m.PNG)
For instance, row 1, column 2 (the orange dot with a red border) is calculated as  a1,1∗b1,2+a1,2∗b2,2

### Predict  multi image

```py
weights.shape
# torch.Size([784, 1])

```

```py
train_x.shape
# torch.Size([12396, 784])

```

```py
# xb@weights + bias is the formula to predict is the image is 3 or 7,1 is 3,0 is 7
def linear1(xb): return xb@weights + bias
preds = linear1(train_x)
preds
```

```py
corrects = (preds>0.5).float() == train_y
corrects
# tensor([[ True],
#         [ True],
#         [ True],
#         ...,
#         [False],
#         [False],
#         [False]])
```

```py
corrects.float().mean().item()
# 0.49080348014831543
```

### first loss finction

suppose we had three images which we knew were a 3, a 7, and a 3. And suppose our model predicted with high confidence (0.9) that the first was a 3, with slight confidence (0.4) that the second was a 7, and with fair confidence (0.2), but incorrectly, that the last was a 7. This would mean our loss function would receive these values as its inputs:

```py
# 1 is 3,0 is 7
trgts  = tensor([1,0,1])
prds   = tensor([0.9, 0.4, 0.2])

```

C/CUDA speed
具体的意思可以理解为：针对于x而言，如果其中的每个元素都满足condition，就返回x的值；如果不满足condition，就将y对应位置的元素或者y的值(如果y为氮元素tensor的话)替换x的值，

```py
# low is good
# low is good
def mnist_loss(predictions, targets):
    return torch.where(targets==1, 1-predictions, predictions).mean()
```

```py
# example
torch.where(trgts==1, 1-prds, prds)
# torch.where(trgts==1, 1-prds, prds)
```

```py
mnist_loss(prds,trgts)
# tensor(0.4333)
```

### better loss finction

buts this mnist_loss has a problem , it assume the predict is alwasy0-1
we can use sigmoid function,it map all the value between 1 and 0

```py
def sigmoid(x): return 1/(1+torch.exp(-x))
plot_function(torch.sigmoid, title='Sigmoid', min=-4, max=4)
```

![sig](/img/ai_t/t1/sig.png)

```py
def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()
```

why select sigmoid()? becuase it can keep the mnist_loss function
has a meaningful derivative. It can't have big flat sections and large jumps, but instead must be reasonably smooth. This is why we designed a loss function that would respond to small changes in confidence level
This requirement means that sometimes it does not really reflect exactly what we are trying to achieve, but is rather a compromise between our real goal, and a function that can be optimized using its gradient.

為什麼選擇sigmoid（）？ 因為它可以保留mnist_loss函數
具有有意義的導數 它不能有較大的扁平部分和較大的跳動，而必須相當平滑。 這就是為什麼我們設計一個損失函數以響應置信度水平的微小變化的原因
此要求意味著有時它不能真正反映出我們要實現的目標，但實際上是我們實際目標與可以使用其梯度進行優化的功能之間的折衷。

## Loss vs Metrics

`Metrics`, are the numbers that we really care about. These are the values that are printed at the end of each epoch that tell us how our model is really doing.  when judging the performance of a model,we use metrics

`loss`,To drive automated learning, the loss must be a function that has a meaningful derivative. It can't have big flat sections and large jumps, but instead must be reasonably smooth. This is why we designed a loss function that would respond to small changes in confidence level.
