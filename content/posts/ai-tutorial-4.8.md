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
https://blog.csdn.net/qq_39709535/article/details/80803003

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