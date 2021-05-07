+++
title = "Ai Tutorial 5.2 Image Classification >2 types Cross-entropy loss 1"
date = "2021-05-06T16:55:28+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch","cross-entropy loss","Log Likelihood","softmax"]
keywords = ["", ""]
description = ""
showFullContent = false
+++
[My Code
](https://colab.research.google.com/drive/1Rqum2194iz5nXH26PPoBMpKM71wQ4eYI?usp=sharing)

[Source Code
](https://colab.research.google.com/github/fastai/fastbook/blob/master/05_pet_breeds.ipynb#scrollTo=YOTrrdP7BuWd)
# Cross-entropy loss

fastai will choose the loss based on what kind of data and model you are using. In this case we have image data and a categorical outcome, so fastai will default to using cross-entropy loss.

Cross-entropy loss can use for more than 2 category

## Viewing Activations and Labels

```py
x,y = dls.one_batch()

```

```py
x.shape
# torch.Size([64, 3, 224, 224])
```

our batch isze is 64,so we can see the list is 64 item.0-36,37 type

```py
y
# TensorCategory([ 7,  1,  0, 14, 19,  9,  2, 35, 12,  0, 26, 34, 18, 21,  5,  8,  0, 35,  8,  8, 28, 35, 17, 34, 21,  3, 17, 19, 18, 22,  9, 12, 34, 10, 35, 25, 13, 18, 32, 36, 20, 26,  5, 18, 31,  6,  7,  9,
#          3,  1,  0, 30,  2,  4, 12, 24, 30,  1, 30, 20, 30, 21,  3, 12], device='cuda:0')
```

see the predict

```py
preds,target = learn.get_preds(dl=[(x,y)])
```

```py
target
# TensorCategory([ 7,  1,  0, 14, 19,  9,  2, 35, 12,  0, 26, 34, 18, 21,  5,  8,  0, 35,  8,  8, 28, 35, 17, 34, 21,  3, 17, 19, 18, 22,  9, 12, 34, 10, 35, 25, 13, 18, 32, 36, 20, 26,  5, 18, 31,  6,  7,  9,
#          3,  1,  0, 30,  2,  4, 12, 24, 30,  1, 30, 20, 30, 21,  3, 12])
```

```py
# preds containe 64 pred, becasue beatch size is 64,probilitiesof 37 type ,because it contain 37 type
preds.shape
# torch.Size([64, 37])
```

```py
# between 0-1,
preds[0]
# tensor([2.7509e-08, 4.1222e-08, 3.7762e-06, 4.6692e-07, 6.6490e-06, 1.6953e-08, 2.9940e-05, 9.9975e-01, 1.9381e-04, 2.9978e-09, 1.0564e-08, 1.0974e-07, 3.9340e-07, 1.0617e-08, 7.8258e-09, 4.8307e-08,
#         2.9032e-07, 8.0013e-09, 2.2539e-08, 5.3139e-07, 1.7915e-08, 1.0556e-07, 3.6633e-06, 5.3050e-06, 1.2096e-07, 6.5162e-08, 4.3347e-09, 9.6756e-08, 5.2215e-06, 2.0169e-07, 1.5412e-07, 8.8911e-07,
#         2.2806e-07, 1.2523e-07, 6.1131e-09, 6.0672e-08, 3.3345e-07])
```

```py
# add them all is 1
len(preds[0]),preds[0].sum()
# (37, tensor(1.))
```

## Softmax

```py
# if we have 6 picture,and 2 type
acts = torch.randn((6,2))*2
acts
# first column is confident of the 3 ,second is the column of the 7
# tensor([[-0.9916, -2.2545],
#         [ 0.1560, -1.9368],
#         [-0.6164,  1.1047],
#         [-2.0798, -2.1778],
#         [ 1.6429, -3.7728],
#         [-1.2445, -2.9512]])
```

```py
acts.sigmoid()
# we can not direct use sigmoid,because c1+c2!=1, we hope the probaility of 7 and 3 sum is 1
```

we can calculate the relative of the 7 and 3

```py
acts[:,0]
# get the first column
# tensor([-0.9916,  0.1560, -0.6164, -2.0798,  1.6429, -1.2445])
```

```py

# this is first column
f_c=(acts[:,0]-acts[:,1]).sigmoid()
f_c
# second column is 1- first column softmax do this thing
```

```py
s_c=1-f_c
```

softmax do this thing

```py

def softmax(x): return exp(x) / exp(x).sum(dim=1, keepdim=True)
# exp is e**8 ,e is 2.718
```

```py
sm_acts = torch.softmax(acts, dim=1)
sm_acts
# tensor([[0.7795, 0.2205],
#         [0.8902, 0.1098],
#         [0.1517, 0.8483],
#         [0.5245, 0.4755],
#         [0.9956, 0.0044],
#         [0.8464, 0.1536]])
```

softmax is the multi-category equivalent of sigmoid—we have to use it any time we have more than two categories and the probabilities of the categories must add to 1, and we often use it even when there are just two categories, just to make things a bit more consistent.  
Taking the exponential ensures all our numbers are positive, and then dividing by the sum ensures we are going to have a bunch of numbers that add up to 1. The exponential also has a nice property: if one of the numbers in our activations x is slightly bigger than the others, the exponential will amplify this (since it grows, well... exponentially), which means that in the softmax, that number will be closer to 1.

Intuitively, the softmax function really wants to pick one class among the others, so it's ideal for training a classifier when we know each picture has a definite label. (Note that it may be less ideal during inference, as you might want your model to sometimes tell you it doesn't recognize any of the classes that it has seen during training, and not pick a class because it has a slightly bigger activation score. In this case, it might be better to train a model using multiple binary output columns, each using a sigmoid activation.)

Softmax is the first part of the cross-entropy loss—the second part is log likelihood.

取指數可確保我們所有的數字都是正數，然後除以和可確保我們將擁有一堆加起來為1的數字。指數也具有很好的屬性：如果x中的數字之一比其他稍大一些,放大（因為它會以指數形式增長）（這是指數增長），這意味著在softmax中，該數字將接近於1。

直觀上，softmax函數確實希望從其他類別中選擇一個類別，因此當我們知道每張圖片都有一個確定的標籤時，訓練分類器是理想的選擇。 （請注意，在推理過程中它可能不太理想，因為您可能希望模型有時告訴您，它無法識別訓練中看到的任何課程，並且不選一個課程，因為它的激活分數稍高在這種情況下，最好使用多個二進制輸出列訓練模型，每個輸出列都使用S型激活。）

## Log Likelihood

```py
# old
def mnist_loss(inputs, targets):
    inputs = inputs.sigmoid()
    return torch.where(targets==1, 1-inputs, inputs).mean()
```

```py
# tag
# 0 is7, 1 is3?????
targ = tensor([0,1,0,1,1,0])
```

```py
# these are the softmax activations:
# left is 3,rightis 7 probility
sm_acts

# tensor([[0.7795, 0.2205],
#         [0.8902, 0.1098],
#         [0.1517, 0.8483],
#         [0.5245, 0.4755],
#         [0.9956, 0.0044],
#         [0.8464, 0.1536]])
```

```py
# get the taged probility
idx = range(6)
sm_acts[idx, targ]
# tensor([0.7795, 0.1098, 0.1517, 0.4755, 0.0044, 0.8464])

```

```py
# is 3? 
# 0 is3, 1 is 7

#hide_input
from IPython.display import HTML
df = pd.DataFrame(sm_acts, columns=["3","7"])
df['targ'] = targ
df['idx'] = idx
df['loss'] = sm_acts[range(6), targ]
t = df.style.hide_index()
#To have html code compatible with our script
html = t._repr_html_().split('</style>')[1]
html = re.sub(r'<table id="([^"]+)"\s*>', r'<table >', html)
display(HTML(html))


# 3 7 targ idx loss
# 0.779514 0.220486 0 0 0.779514
# 0.890204 0.109796 1 1 0.109796
# 0.151727 0.848273 0 2 0.151727
# 0.524483 0.475517 1 3 0.475517
# 0.995573 0.004427 1 4 0.004427
# 0.846414 0.153586 0 5 0.846414

```

the above is log likehold

Pytorch have a function that do the samething with the sm_acts[],but it recive negative number nll_loss

```py

# do the same thing of sm_acts[range(n), targ],except it takes the negative, because when applying the log afterward, we will have negative numbers
F.nll_loss(sm_acts, targ, reduction='none')
# tensor([-0.7795, -0.1098, -0.1517, -0.4755, -0.0044, -0.8464])
```


we can see thta the Log Likelihood get the number is bigger when the distance is close,but we want when the distance is close,the loss number is close, we handle this problem later
