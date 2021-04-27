+++
title = "Ai Tutorial 4.1"
date = "2021-04-27T15:51:50+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch"]
keywords = ["", ""]
description = ""
showFullContent = false
+++
# _Get the number sample_
```py
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

```
```py
from fastai.vision.all import *
from fastbook import *

matplotlib.rc('image', cmap='Greys')

```

```py
path = untar_data(URLs.MNIST_SAMPLE)
Path.BASE_PATH = path
path.ls()
```
[Path('valid'),Path('train'),Path('labels.csv')]
train group use to train,valid group use to test

```py
(path/'train').ls()
# [Path('train/7'),Path('train/3')]
```
```py
threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()
threes
# (#6131) [Path('train/3/10.png'),Path('train/3/10000.png'),Path('train/3/10011.png'),Path('train/3/10031.png'),Path('train/3/10034.png'),Path('train/3/10042.png'),Path('train/3/10052.png'),Path('train/3/1007.png'),Path('train/3/10074.png'),Path('train/3/10091.png')...]

```
open a image to see
```
im3_path = threes[1]
im3 = Image.open(im3_path)
im3
```
![3](/img/ai_t/t1/3.png)

```py
array(im3).shape
# a 28 * 28 image
# (28, 28)
```
turn the image into a  2 d array
show row 4:10(not include),column 4:10(not include)
```py
array(im3)[4:10,4:10]
# array([[  0,   0,   0,   0,   0,   0],
#        [  0,   0,   0,   0,   0,  29],
#        [  0,   0,   0,  48, 166, 224],
#        [  0,  93, 244, 249, 253, 187],
#        [  0, 107, 253, 253, 230,  48],
#        [  0,   3,  20,  20,  15,   0]], dtype=uint8)
```

tensor is array use in pytorch
```py
tensor(im3)[4:10,4:10]
```
show the head of th 3,the colr is Grey,0,is white
```py
im3_t = tensor(im3)
df = pd.DataFrame(im3_t[4:15,4:22])
df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')
```
![3_h](/img/ai_t/t1/3_head.PNG)

# _First Try: Pixel Similarity Baseline_
we fins the average pixel value of 3 and 7,so we can define the ideal 3 ,7,then the image compare to the ideal 3 7,to identify is 3 or 7  

因此，這是第一個想法：我們如何找到3s的每個像素的平均像素值，然後對7s進行相同的處理。 這將為我們提供兩個組平均值，定義我們可以稱之為“理想”的3和7。然後，將圖像分類為一個數字或另一個數字，我們將看到圖像與這兩個理想數字中的哪一個最相似。 當然，這似乎總比沒有好，因此它將成為一個良好的基準。

### what is a baseline
A simple model which you are confident should perform reasonably well.  
您相信一個簡單的模型應該可以表現良好。 它應該很容易實現，也很容易測試，這樣您就可以測試每個改進的想法，並確保它們總是比基線更好。 不從合理的基准開始，很難知道您的超級幻想模型是否真的有用。 創建基準的一種好方法是執行我們在此處所做的工作：考慮一個簡單，易於實現的模型。 另一個好的方法是四處搜尋，以解決與您的問題相似的其他人，然後在您的數據集上下載並運行他們的代碼。 理想情況下，嘗試這兩個！

## step1
get the average of pixel values for each of our two groups
create a tensor containing all of our 3s stacked together

```py
# tensor(Image.open(o)) turn the image to a 2d array
# this way for loop more fast
seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
len(three_tensors),len(seven_tensors)
# (6131, 6265)
```

```py

# display a image in tensor
show_image(three_tensors[1]);

```
we want ot calculate each pixel avarge strange,so wee need to make a 3d tensor /array rank3 tensor each image to a 2d array,multi image become a 3d array

![rank](/img/ai_t/t1/rank.jpg)

```py
# change the whole tensorto the 0-1 floating point
stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_threes = torch.stack(three_tensors).float()/255
stacked_threes.shape #torch.Size([6131, 28, 28]),6131 images, each 28*28 pixels
```

rank is the number of axes or dimensions in a tensor; shape is the size of each axis of a tensor. above shape is [6131, 28, 28],rank is 3 (len(stacked_threes.shape)==3)
```py
# get the rank
len(stacked_threes.shape)
# or
stacked_threes.ndim  #3
```
```py
# calculate the mean of each pixel
mean3 = stacked_threes.mean(0)
show_image(mean3);
```
![ideal3](/img/ai_t/t1/ideal_3.png)

```py
mean7=stacked_sevens.mean(0)
show_image(mean7)
```
![ideal7](/img/ai_t/t1/ideal_7.png)

```py
# get a 3
a_3=stacked_threes[1]
show_image(a_3)
```

now we compare the ideal 3 and the real 3  
How do we compare the a_3 and the mean3?? 
1. compare each pixel,get the bas,calculate the avh of the absof each pixel,This is called the mean absolute difference or L1 norm

2. calculaet the (dif)^2,mean, than square root.This is called the root mean squared error (RMSE) or L2 norm.

```py
# try
# 1
dist_3_abs = (a_3 - mean3).abs().mean()
# 2
dist_3_sqr = ((a_3 - mean3)**2).mean().sqrt()
dist_3_abs,dist_3_sqr

# (tensor(0.1114), tensor(0.2021))
```
```py
dist_7_abs = (a_3 - mean7).abs().mean()
dist_7_sqr = ((a_3 - mean7)**2).mean().sqrt()
dist_7_abs,dist_7_sqr
# (tensor(0.1586), tensor(0.3021))

```
In both cases, the distance between our 3 and the "ideal" 3 is less than the distance to the ideal 7. So our simple model will give the right prediction in this case.

PyTorch already provides both of these as loss functions. You'll find these inside torch.nn.functional, which the PyTorch team recommends importing as F (and is available by default under that name in fastai):

Here mse stands for mean squared error, and l1 refers to the standard mathematical jargon for mean absolute value (in math it's called the L1 norm).
```py
F.l1_loss(a_3.float(),mean7), F.mse_loss(a_3,mean7).sqrt()
# (tensor(0.1586), tensor(0.3021))
```

## NumPy Arrays and PyTorch Tensors
they almost the same but  NumPy Arrays not support GPU
```py
data = [[1,2,3],[4,5,6]]
arr = array (data)
tns = tensor(data)
```

```py
arr  # numpy
# array([[1, 2, 3],
#        [4, 5, 6]])
```

```py
tns  # pytorch

# tensor([[1, 2, 3],
#         [4, 5, 6]])
```

```py
tns[1]
# get index 1
# tensor([4, 5, 6])

tns[:,1]
# all first axis,index 1 at ssecond axis
# tensor([2, 5])


tns[1,1:3]
# first axis :1,sendoc axis 1-3(exclude)
# tensor([5, 6])

tns+1
# tensor([[2, 3, 4],
#         [5, 6, 7]])

tns*1.5
# tensor([[1.5000, 3.0000, 4.5000],
#         [6.0000, 7.5000, 9.0000]])
```