+++
title = "Ai Tutorial 5.1 Image Classification 1"
date = "2021-05-06T16:15:02+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch","Image Classification"]
keywords = ["", ""]
description = ""
showFullContent = false
+++
# Image Classification 1

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

## Checking and Debugging a DataBlock

Checking

```py
# show some image
dls.show_batch(nrows=1, ncols=3)

```

![check](/img/ai_t/t1/check.PNG)

we should check all the data,that the tag is correct,we can check by our eyes,or by the [model]({{< ref "posts/ai-tutorial-2#start-create-the-model" >}} "model")

Debugging

error because the image size is different

```py
#hide_output
pets1 = DataBlock(blocks = (ImageBlock, CategoryBlock),
                 get_items=get_image_files, 
                 splitter=RandomSplitter(seed=42),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'))
pets1.summary(path/"images")
```

```py
Setting-up type transforms pipelines
Collecting items from /home/jhoward/.fastai/data/oxford-iiit-pet/images
Found 7390 items
2 datasets of sizes 5912,1478
Setting up Pipeline: PILBase.create
Setting up Pipeline: partial -> Categorize

Building one sample
  Pipeline: PILBase.create
    starting from
      /home/jhoward/.fastai/data/oxford-iiit-pet/images/american_pit_bull_terrier_31.jpg
    applying PILBase.create gives
      PILImage mode=RGB size=500x414
  Pipeline: partial -> Categorize
    starting from
      /home/jhoward/.fastai/data/oxford-iiit-pet/images/american_pit_bull_terrier_31.jpg
    applying partial gives
      american_pit_bull_terrier
    applying Categorize gives
      TensorCategory(13)

Final sample: (PILImage mode=RGB size=500x414, TensorCategory(13))


Setting up after_item: Pipeline: ToTensor
Setting up before_batch: Pipeline: 
Setting up after_batch: Pipeline: IntToFloatTensor

Building one batch
Applying item_tfms to the first sample:
  Pipeline: ToTensor
    starting from
      (PILImage mode=RGB size=500x414, TensorCategory(13))
    applying ToTensor gives
      (TensorImage of size 3x414x500, TensorCategory(13))

Adding the next 3 samples

No before_batch transform to apply

Collating items in a batch
Error! It's not possible to collate your items in a batch
Could not collate the 0-th members of your tuples because got the following shapes
torch.Size([3, 414, 500]),torch.Size([3, 375, 500]),torch.Size([3, 500, 281]),torch.Size([3, 203, 300])
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-11-8c0a3d421ca2> in <module>
      4                  splitter=RandomSplitter(seed=42),
      5                  get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'))
----> 6 pets1.summary(path/"images")

~/git/fastai/fastai/data/block.py in summary(self, source, bs, show_batch, **kwargs)
    182         why = _find_fail_collate(s)
    183         print("Make sure all parts of your samples are tensors of the same size" if why is None else why)
--> 184         raise e
    185 
    186     if len([f for f in dls.train.after_batch.fs if f.name != 'noop'])!=0:

~/git/fastai/fastai/data/block.py in summary(self, source, bs, show_batch, **kwargs)
    176     print("\nCollating items in a batch")
    177     try:
--> 178         b = dls.train.create_batch(s)
    179         b = retain_types(b, s[0] if is_listy(s) else s)
    180     except Exception as e:

~/git/fastai/fastai/data/load.py in create_batch(self, b)
    125     def retain(self, res, b):  return retain_types(res, b[0] if is_listy(b) else b)
    126     def create_item(self, s):  return next(self.it) if s is None else self.dataset[s]
--> 127     def create_batch(self, b): return (fa_collate,fa_convert)[self.prebatched](b)
    128     def do_batch(self, b): return self.retain(self.create_batch(self.before_batch(b)), b)
    129     def to(self, device): self.device = device

~/git/fastai/fastai/data/load.py in fa_collate(t)
     44     b = t[0]
     45     return (default_collate(t) if isinstance(b, _collate_types)
---> 46             else type(t[0])([fa_collate(s) for s in zip(*t)]) if isinstance(b, Sequence)
     47             else default_collate(t))
     48 

~/git/fastai/fastai/data/load.py in <listcomp>(.0)
     44     b = t[0]
     45     return (default_collate(t) if isinstance(b, _collate_types)
---> 46             else type(t[0])([fa_collate(s) for s in zip(*t)]) if isinstance(b, Sequence)
     47             else default_collate(t))
     48 

~/git/fastai/fastai/data/load.py in fa_collate(t)
     43 def fa_collate(t):
     44     b = t[0]
---> 45     return (default_collate(t) if isinstance(b, _collate_types)
     46             else type(t[0])([fa_collate(s) for s in zip(*t)]) if isinstance(b, Sequence)
     47             else default_collate(t))

~/anaconda3/lib/python3.7/site-packages/torch/utils/data/_utils/collate.py in default_collate(batch)
     53             storage = elem.storage()._new_shared(numel)
     54             out = elem.new(storage)
---> 55         return torch.stack(batch, 0, out=out)
     56     elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
     57             and elem_type.__name__ != 'string_':

RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 0. Got 414 and 375 in dimension 2 at /opt/conda/conda-bld/pytorch_1579022060824/work/aten/src/TH/generic/THTensor.cpp:612
```

now we can train it

```py
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(2)
```
