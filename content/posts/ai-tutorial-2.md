+++
title = "Ai Tutorial 2"
date = "2021-04-22T15:30:42+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["", ""]
keywords = ["", ""]
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
  subscription_key = "158558d991224dd9930aa6397b1f0b35"
  search_url = "https://api.bing.microsoft.com/v7.0/images/search"
  headers = {"Ocp-Apim-Subscription-Key" : subscription_key}
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