+++
title = "Ai Tutorial 6.1 Other Computer Vision Problems-Multi-Label Classification"
date = "2021-05-16T15:03:29+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch","Multi-Label Classification","dataframe"]
keywords = ["", ""]
description = ""
showFullContent = false
+++
[My Code
](https://colab.research.google.com/drive/1VzYTbBKx-JPfJ1FaLHOhG1Hpf3GNdG5C?usp=sharing)

[Source Code
](https://colab.research.google.com/github/fastai/fastbook/blob/master/06_multicat.ipynb)

# Multi-Label Classification

a picture,can > 1 tag,or 0 tag

## pandas dataframe tutorial

the image that have more than one tag

```py
from fastai.vision.all import *
path = untar_data(URLs.PASCAL_2007)
```

```py
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

dataframe get a column

```py
df['fname']

# 0       000005.jpg
# 1       000007.jpg
# 2       000009.jpg
# 3       000012.jpg
# 4       000016.jpg
#            ...    
# 5006    009954.jpg
# 5007    009955.jpg
# 5008    009958.jpg
# 5009    009959.jpg
# 5010    009961.jpg
# Name: fname, Length: 5011, dtype: object
```

dataframe get a row buy index

```py
df.iloc[0]

# fname       000005.jpg
# labels           chair
# is_valid          True
# Name: 0, dtype: object
```

get all row,first column

```py
df.iloc[:,0]

```

get first row,all coumn

```py
df.iloc[0,:]

```

create dataframe

```py
df1=pd.DataFrame()
df1['a']=[1,2,3,4]
df1

# a
# 0 1
# 1 2
# 2 3
# 3 4
```

Dataframe  operator

```py
df1['b']=[10,20,30,40]
df1['a']+df1['b']

# 0    11
# 1    22
# 2    33
# 3    44
# dtype: int64
```
