+++
title = "Ai Tutorial 6.2 Other Computer Vision Problems-Multi-Label Classification.2 Data Block"
date = "2021-05-16T15:29:00+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch","Multi-Label Classification","data block"]
keywords = ["", ""]
description = ""
showFullContent = false
+++
[My Code
](https://colab.research.google.com/drive/1VzYTbBKx-JPfJ1FaLHOhG1Hpf3GNdG5C?usp=sharing)

[Source Code
](https://colab.research.google.com/github/fastai/fastbook/blob/master/06_multicat.ipynb)

# Problems-Multi-Label Classification.2  Data Block

## Data Block

in fast api and pytorch,we use Dataset and DataLoader to access the data

_Pytorch_

* Dataset:A collection that returns a tuple of your independent(image) and dependent variable(tag) for a single item
* DataLoader:: An iterator that provides a stream of mini-batches, where each mini-batch is a tuple of a batch of independent variables and a batch of dependent variables

_Fastai provide_

* Datasets:: An object that contains a training Dataset and a validation Dataset
* DataLoaders:: An object that contains a training DataLoader and a validation DataLoader

```py
from fastai.vision.all import *
path = untar_data(URLs.PASCAL_2007)
# use the path read the csv to the dataframe
df = pd.read_csv(path/'train.csv')
df.head()
# fname labels is_valid
# 0 000005.jpg chair True
# 1 000007.jpg car True
# 2 000009.jpg horse person True
# 3 000012.jpg car False
# 4 000016.jpg bicycle True
```

try to build a Datablock,not correct

```py
dblock = DataBlock()
dsets = dblock.datasets(df)
```

```py
# datasets contain train dataset and train dataset
len(dsets.train),len(dsets.valid)

# (4009, 1002)
```

```py

x,y = dsets.train[0]
x,y
# we can see that the x an  y is the same,this is not right

# (fname       008663.jpg
#  labels      car person
#  is_valid         False
#  Name: 4346, dtype: object, fname       008663.jpg
#  labels      car person
#  is_valid         False
#  Name: 4346, dtype: object)
```

build a Data block in correct way
we need to tell datablock,what is indepedent vairable (x) and depedent vairable(y) in the data frame

```py
df
```

![df](/img/ai_t/t1/df.PNG)

tell date block that,fname is the x,u is the labels

```py
dblock = DataBlock(get_x = lambda r: r['fname'], get_y = lambda r: r['labels'])
dsets = dblock.datasets(df)
```

```py
dsets.train[0]
# ('005620.jpg', 'aeroplane')
```

same ,use function,not lambda

```py
def get_x(r): return r['fname']
def get_y(r): return r['labels']
dblock = DataBlock(get_x = get_x, get_y = get_y)
dsets = dblock.datasets(df)
dsets.train[0]

# ('002549.jpg', 'tvmonitor')

```

better,x is path, and y is more than two tag

```py
def get_x(r): return path/'train'/r['fname']
def get_y(r): return r['labels'].split(' ')
dblock = DataBlock(get_x = get_x, get_y = get_y)
dsets = dblock.datasets(df)
dsets.train[0]
# (Path('/root/.fastai/data/pascal_2007/train/002844.jpg'), ['train'])
```

more better,MultiCategoryBlock ,can return one-hot encoding

```py
# ategoryBlock return a number,MultiCategoryBlock return multi number
dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   get_x = get_x, get_y = get_y)
dsets = dblock.datasets(df)
dsets.train[0]
# 1 mean the image is the type,we can have a fix length of the array


# (PILImage mode=RGB size=500x375,
#  TensorMultiCategory([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]))
```

try to show the tag

```py
dsets.train[0][1]
# TensorMultiCategory([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
# this mean we have 20 avliable tag,and this picture tag is index 11 tag
```

get the 1 index number

```py
# get the index when ==1
torch.where(dsets.train[0][1]==1.)
# (TensorMultiCategory([11]),)
```

show the tag

```py
# show the tag
idxs = torch.where(dsets.train[0][1]==1.)[0]
idxs
# TensorMultiCategory([11])
```

```py
dsets.train.vocab[idxs]
# ['dog']
```
