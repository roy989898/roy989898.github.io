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
# Get the number sample
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