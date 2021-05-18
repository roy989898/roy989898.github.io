+++
title = "Ai Tutorial 6.4  Other Computer Vision Problems-Multi-Label Binary Cross-Entropy Image and Point"
date = "2021-05-16T18:41:39+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch","Multi-Label Classification","Cross-Entropy"]
keywords = ["", ""]
description = ""
showFullContent = false
+++


# Image and Point

## key point model

A key point refers to a specific location represented in an image—in this case, we'll use images of people and we'll be looking for the center of the person's face in each image. That means we'll actually be predicting two values for each image: the row and column of the face center.

## Assemble the Data

```py
path = untar_data(URLs.BIWI_HEAD_POSE)
```

```py
#hide
Path.BASE_PATH = path
```

24 directories numbered from 01 to 24 (they correspond to the different people photographed), and a corresponding .obj file for each (we won't need them here)

```py
path.ls().sorted()
# (#50) [Path('01'),Path('01.obj'),Path('02'),Path('02.obj'),Path('03'),Path('03.obj'),Path('04'),Path('04.obj'),Path('05'),Path('05.obj')...]

```

```py
(path/'01').ls().sorted()
# (#1000) [Path('01/depth.cal'),Path('01/frame_00003_pose.txt'),Path('01/frame_00003_rgb.jpg'),Path('01/frame_00004_pose.txt'),Path('01/frame_00004_rgb.jpg'),Path('01/frame_00005_pose.txt'),Path('01/frame_00005_rgb.jpg'),Path('01/frame_00006_pose.txt'),Path('01/frame_00006_rgb.jpg'),Path('01/frame_00007_pose.txt')...]

```

get the image file

```py
img_files = get_image_files(path)
img_files
# (#15678) [Path('03/frame_00650_rgb.jpg'),Path('03/frame_00644_rgb.jpg'),Path('03/frame_00491_rgb.jpg'),Path('03/frame_00207_rgb.jpg'),Path('03/frame_00067_rgb.jpg'),Path('03/frame_00056_rgb.jpg'),Path('03/frame_00025_rgb.jpg'),Path('03/frame_00450_rgb.jpg'),Path('03/frame_00584_rgb.jpg'),Path('03/frame_00285_rgb.jpg')...]

```

```py
img_files[0]
# Path('03/frame_00650_rgb.jpg')

``` a function that use the image name to get the pose.txt

```py
# a function that use the image name to get the pose.txt
def img2pose(x): return Path(f'{str(x)[:-7]}pose.txt')
img2pose(img_files[0])
# Path('03/frame_00650_pose.txt')
```

```py
# see the first image
im = PILImage.create(img_files[0])
im.shape
# (480, 640)
```

```py
im.to_thumb(160)
```

![py](/img/ai_t/t1/py.PNG)

## get the head point

```py
cal = np.genfromtxt(path/'01'/'rgb.cal', skip_footer=6)
cal

# array([[517.679,   0.   , 320.   ],
#        [  0.   , 517.679, 240.5  ],
#        [  0.   ,   0.   ,   1.   ]])
```

```py
# the function to get the head point
def get_ctr(f):
    ctr = np.genfromtxt(img2pose(f), skip_header=3)
    c1 = ctr[0] * cal[0][0]/ctr[2] + cal[0][2]
    c2 = ctr[1] * cal[1][1]/ctr[2] + cal[1][2]
    return tensor([c1,c2])
```

```py
get_ctr(img_files[0])
tensor([447.6672, 277.1215])
```

we have 2 problem at here  

One important point to note is that we should not just use a random splitter. The reason for this is that the same people appear in multiple images in this dataset, but we want to ensure that our model can generalize to people that it hasn't seen yet. Each folder in the dataset contains the images for one person. Therefore, we can create a splitter function that returns true for just one person, resulting in a validation set containing just that person's images.  

The only other difference from the previous data block examples is that the second block is a PointBlock. This is necessary so that fastai knows that the labels represent coordinates; that way, it knows that when doing data augmentation, it should do the same augmentation to these coordinates as it does to the images:  

splitter=FuncSplitter(lambda o: o.parent.name=='13'), mean we want ot create validation set containing just that person's images.that contain in the document 13

```py
biwi = DataBlock(
    blocks=(ImageBlock, PointBlock),
    get_items=get_image_files,
    get_y=get_ctr,
    splitter=FuncSplitter(lambda o: o.parent.name=='13'),
    batch_tfms=[*aug_transforms(size=(240,320)), 
                Normalize.from_stats(*imagenet_stats)]
)
```

```py
# check is the data ok
dls = biwi.dataloaders(path)
dls.show_batch(max_n=9, figsize=(8,6))
```

![train_9](/img/ai_t/t1/train_9.PNG)

```py
# check the size
xb,yb = dls.one_batch()
xb.shape,yb.shape
# (torch.Size([64, 3, 240, 320]), torch.Size([64, 1, 2]))
```

```py
the location of the point
yb[0]
# TensorPoint([[-0.2325,  0.1644]], device='cuda:0')
```

## train a Model

we used y_range to tell fastai the range of our targets? We'll do the same here (coordinates in fastai and PyTorch are always rescaled between -1 and +1):

```py
# why a y_range???

learn = cnn_learner(dls, resnet18, y_range=(-1,1))
```

```py
# the y_range function
def sigmoid_range(x, lo, hi): return torch.sigmoid(x) * (hi-lo) + lo
```

the loss function in the learner

```py
# default use MSELoss
dls.loss_func
# FlattenedLoss of MSELoss()
```

pytorch 的nn.MSELoss()損失函數 <https://blog.csdn.net/weixin_38145317/article/details/103735784>  

This makes sense, since when coordinates are used as the dependent variable, most of the time we're likely to be trying to predict something as close as possible; that's basically what MSELoss (mean squared error loss) does. If you want to use a different loss function, you can pass it to cnn_learner using the loss_func parameter.

## metrics

Note also that we didn't specify any metrics. That's because the MSE is already a useful metric for this task (although it's probably more interpretable after we take the square root).

```py
min,steep=learn.lr_find()
```

![lrf6](/img/ai_t/t1/lrf6.PNG)

```py
min,steep
```

```py
lr = 1e-2  # book is 2e-2
learn.fine_tune(3, lr)
# epoch train_loss valid_loss time
# 0 0.048711 0.018842 01:48
# epoch train_loss valid_loss time
# 0 0.008920 0.010456 01:51
# 1 0.002904 0.000215 01:51
# 2 0.001400 0.000146 01:51
```

Generally when we run this we get a loss of around 0.0001, which corresponds to an average coordinate prediction error of:

```py
math.sqrt(0.0001)
```

learn.show_results(ds_idx=1, nrows=3, figsize=(6,8))

![pre_r6](/img/ai_t/t1/pre_r6.PNG)

# Conclusion

fastai will automatically try to pick the right one from the data you built, but if you are using pure PyTorch to build your DataLoaders, make sure you think hard when you have to decide on your choice of loss function, and remember that you most probably want:

nn.CrossEntropyLoss for single-label classification nn.BCEWithLogitsLoss for multi-label classification nn.MSELoss for regression
