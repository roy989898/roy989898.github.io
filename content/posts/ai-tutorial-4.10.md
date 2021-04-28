+++
title = "Ai Tutorial 4.10 Put it all together"
date = "2021-04-28T16:52:44+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch","SGD"]
keywords = ["", ""]
description = ""
showFullContent = false
+++
# Put it all together

 each epoch is like this

```py
# basic example
# for x,y in dl:
#     pred = model(x)
#     loss = loss_func(pred, y)
#     loss.backward()
#     parameters -= parameters.grad * lr
```
re-initialize our parameters:
```py

weights = init_params((28*28,1))
bias = init_params(1)
weights.shape
# torch.Size([784, 1])
```

create DataLoader of train data  from [Dataset]({{< ref "posts/ai-tutorial-4.8.md#prepare-the-pytorch-need-format" >}} "Dataset")
```py
dl = DataLoader(dset, batch_size=256)
xb,yb = first(dl)
xb.shape,yb.shape
(torch.Size([784]), tensor([1]))
# (torch.Size([784]), tensor([1]))
```

create DataLoader of valid data [valid data]({{< ref "posts/ai-tutorial-4.8.md#prepare-the-valid-data" >}} "valid data")
```
valid_dl = DataLoader(valid_dset, batch_size=256)
```