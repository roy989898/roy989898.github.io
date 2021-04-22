+++
title = "Ai Tutorial 2"
date = "2021-04-22T15:30:42+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch"]

description = ""
showFullContent = false
+++
# Build your Bear reconigize model

## Download image
we use the Azure [bing image search API](https://www.microsoft.com/en-us/bing/apis/bing-image-search-api)

you need to apply the key for free

```python
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

from fastbook import *
from fastai.vision.widgets import *

key = 'secret_key_from_bing'

import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
def search_images_bing_min(search_term):
  search_url = "https://api.bing.microsoft.com/v7.0/images/search"
  headers = {"Ocp-Apim-Subscription-Key" : key}
  # search_term = "grizzly bear"
  params  = {"q": search_term, "license": "public", "imageType": "photo","count":'150'}
  response = requests.get(search_url, headers=headers, params=params)
  response.raise_for_status()
  search_results = response.json()
  # ims=[img["thumbnailUrl"] for img in search_results["value"]]
  ims=search_results["value"]
  return ims
```

we can use the abovce function to down the image

```python
ims=search_images_bing_min("grizzly bear")
len(ims)

# ims is a string array

```


now we can download the iomage to the google drive
```python
bear_types = 'grizzly','black','teddy'
path = Path('bearss')
if not path.exists():
    path.mkdir()
for o in bear_types:
  print(o)
  dest = (path/o)
  dest.mkdir(exist_ok=True)
  results = search_images_bing_min(f'{o} bear')
  
  # print(results)
  ims=[img["contentUrl"] for img in results]
  # print(ims)
  download_images(dest, urls=ims)
```

clean the image,remove the fail image
```python
fns = get_image_files(path)
fns
failed = verify_images(fns)
failed
# if fail,remove it
failed.map(Path.unlink);
```

## intro to create the model
datablock is the templat of a dataloader
data loader tell fastai 4 thing:
1. what is the type of the inf
2. how to get the items
3. how to tag the items
4. How to create the validation set


blocks=(ImageBlock, CategoryBlock), mean use the image to predict,  
#CategoryBlock mean target is the category  
get_items=get_image_files ,how to get the image,from files  
splitter mean how to get the validation set  
get_y mean how to get the Category  
parent_label mean use the parent folder as a category tag  
need t oresize all the image to same size,Resize(128)  
```python
# datablock is the templat of a dataloader
# data loader tell fastai 4 thing:
# 1.what is the type of the inf
# 2.how to get the items
# 3.how to tag the items
# 4.How to create the validation set


# blocks=(ImageBlock, CategoryBlock), bean use the image to predict,
#CategoryBlock mean target is the category
# get_items=get_image_files ,how to get the image,from files
# splitter mean how to get the validation set
# get_y mean how to get the Category
# parent_label mean use the parent folder as a category tag
# need t oresize all the image to same size,Resize(128)
bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))

```
create the dataloaders,path is the image path,it validate and train the dataloader

```python
# create the dataloaders,path is the image path,it validate and train the dataloader
dls = bears.dataloaders(path)

```
see some item in the dataLoader
```python
dls.valid.show_batch(max_n=4, nrows=1)

```
we can see ,at defaukt ,the fastai crop the image to the size 128

![default](/img/ai_t/t1/default.PNG)

we also can Squish the image
```python
bears = bears.new(item_tfms=Resize(128, ResizeMethod.Squish))
dls = bears.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)
```
![Squish](/img/ai_t/t1/squizh.PNG)

or pad them
```python
bears = bears.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'))
dls = bears.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)

```
![pad](/img/ai_t/t1/pad.PNG)


or randomly choose a part to crop the image 
```python
# unique=True mean use the sam picture
# RandomResizedCrop  Crop  different part of the same picture,we can have more data to train,Data Augmentation
# min_scale=0.3, select the % of the picture to crop,30%
bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=4, nrows=1, unique=True)

```
![pad](/img/ai_t/t1/RandomResizedCrop.PNG)

another way to Data Augmentation(資料增強,種通過讓有限的資料產生更多的等價資料來人工擴充套件訓練資料集的技術),not crop,just example, to show  rotation, flipping, perspective warping, brightness changes and contrast changes  
batch_tfms apply the aug_transforms to batch ,not only items
```python
# another way to Data Augmentation,not crop,just example, to show  rotation, flipping, perspective warping, brightness changes and contrast changes
# production use crop + aug_transforms
bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)
```
![pad](/img/ai_t/t1/aug_t.PNG)

## Start create the model

```python
bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())

dls = bears.dataloaders(path)


learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)
```
```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```
![square](/img/ai_t/t1/square.PNG)
this is the result box,,to see how many item is worng predict
we can see some grizzly,put in black


```python
interp.plot_top_losses(5, nrows=1)
# we can see some grizzly,put in black
# the number loss,mean the predict is right, but not conficdent,or the answer is wrong,this number will high
```
the number loss,mean the predict is right, but not conficdent,or the answer is wrong,this number will high
![pi](/img/ai_t/t1/pl.PNG)

```python
# ues the fastai GUI to clean the data,remove or re tag
cleaner = ImageClassifierCleaner(learn)
cleaner
```
![gui](/img/ai_t/t1/gui.png)


after change the action to the pait(bear colormgroup(valid,train))  
run below to move and dlete the items
```python
print(cleaner.delete())
print(cleaner.change())
# delete the delete marked photo
for idx in cleaner.delete(): cleaner.fns[idx].unlink()
# move the photo to the right folder /tag
for idx,cat in cleaner.change(): 
  try:
    shutil.move(str(cleaner.fns[idx]), path/cat)
  except:
    cleaner.fns[idx].unlink()
```
after this process, we can retrain again,run 

## Use it
```python
uploader = widgets.FileUpload()
uploader
```

```python
img = PILImage.create(uploader.data[0])
bear_type,_,probs=learn.predict(img)

print(f"bear type: {bear_type}.")
print(f"Probability : {probs[1]}")

```