+++
title = "Ai Notes 1.1"
date = "2021-04-20T14:46:19+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch"]
keywords = ["", ""]
description = ""
showFullContent = false
+++


Your First Model on fastai framework
```python

!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()



from fastbook import *


#id first_training
#caption Results from the first training

from fastai.vision.all import *
# get the cat do images path
path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper()
# because we are stuing the image so use ImageDataLoaders
# valid_pct=0.2 meankeep 20 % photo image not use on training,for testing
# label_func=is_cat mean get the tag of the imag,to detect
# is the photos is a cat or dog,if photo name start by Upper case
# ,than tha pohot is a cat
# item_tfms=Resize(224) resize the photo to 224
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)


# use to create a upload button

uploader = widgets.FileUpload()
uploader


# use the model to detect is your photo is a cat
img = PILImage.create(uploader.data[0])
is_cat,_,probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")
```