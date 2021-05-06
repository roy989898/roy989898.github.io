+++
title = "Ai Tutorial 5.1"
date = "2021-05-06T16:15:02+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch","Image Classification"]
keywords = ["", ""]
description = ""
showFullContent = false
+++
# Image Classification

```py
#hide
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

```

```py
#hide
from fastbook import *

``````py
from fastai.vision.all import *
path = untar_data(URLs.PETS)
```

```py
#hide
Path.BASE_PATH = path
```

## See the image

```py
path.ls()

```

```py
(path/"images").ls()
```

## Get the file name

```py
fname = (path/"images").ls()[1]
# Path('images/havanese_158.jpg')
fname.name
# havanese_158.jpg
```

use regex to get the file name

```py
re.findall(r'(.+)_\d+.jpg$', fname.name)
# ['havanese']
```

## Prepare the data

```py
pets = DataBlock(blocks = (ImageBlock, CategoryBlock),
                 get_items=get_image_files, 
                 splitter=RandomSplitter(seed=42),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                 item_tfms=Resize(460),
                 batch_tfms=aug_transforms(size=224, min_scale=0.75))
dls = pets.dataloaders(path/"images")
```

```py

# this is presizing
#  item_tfms=Resize(460),
#                  batch_tfms=aug_transforms(size=224, min_scale=0.75))
```
we apply the aug_transforms at gpu to a batch of image,but before that,we need to make the image to the smae size,yhan pass the sendor to the GPU,
so we apply Resize(460) to each item at CPU.  
first step:  
Crop full width or height: This is in item_tfms, so it's applied to each individual image before it is copied to the GPU. It's used to ensure all images are the same size. On the training set, the crop area is chosen randomly. On the validation set, the center square of the image is always chosen.
we crop them to a bigger(460) square.Why bigger? because we wnat to have  have spare margin to allow further augmentation transforms on their inner regions without creating empty zones.if performed after resizing down to the augmented size, various common data augmentation transforms might introduce spurious empty zones, degrade data, or both .e.g. rotating an image by 45 degrees fills corner regions of the new bounds with emptiness, which will not teach the model anything.This transformation works by resizing to a square, using a large crop size. On the training set, the crop area is chosen randomly, and the size of the crop is selected to cover the entire width or height of the image, whichever is smaller.  
second step:  
Random crop and augment: This is in batch_tfms, 
at second step,GPU is used for all data augmentation,with a single interpolation(make the picture more clear) at the end.
On the training set, the random crop and any other augmentations are done first.  
我們將gpu上的aug_transforms應用於一批圖像，但是在此之前，我們需要將圖像調整為smae大小，然後將發送方傳遞到GPU，
因此我們將Resize（460）應用於CPU的每個項目。
第一步：
裁剪全寬或全高：這位於item_tfms中，因此在將其複製到GPU之前將其應用於每個單獨的圖像。用於確保所有圖像的尺寸相同。在訓練集上，隨機選擇作物面積。在驗證集上，始終選擇圖像的中心正方形。
我們將它們裁剪到更大的（460）平方。為什麼更大？因為我們擁有足夠的餘量以允許在其內部區域上進行進一步的增強變換而不會創建空白區域，如果在縮小大小到增大大小之後執行此操作，則各種常見的數據增強變換可能會引入虛假的空白區域，降級數據或同時出現這兩種情況。將圖像旋轉45度會以空的方式填充新邊界的角區域，這將不會對模型產生任何影響。此轉換通過使用大作物大小將尺寸調整為正方形來進行。在訓練集上，隨機選擇作物區域，並選擇作物的大小以覆蓋圖像的整個寬度或高度，以較小者為準。
第二步：
隨機裁剪和擴充：這在batch_tfms中，
第二步，將GPU用於所有數據擴充，最後進行一次插值（使畫面更清晰）。
在訓練集上，首先進行隨機裁剪和任何其他擴充。
![tr](/img/ai_t/t1/att_00060.png)