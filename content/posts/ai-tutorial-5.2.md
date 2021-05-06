+++
title = "Ai Tutorial 5.2  Cross-entropy loss"
date = "2021-05-06T16:55:28+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch","cross-entropy loss"]
keywords = ["", ""]
description = ""
showFullContent = false
+++
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
