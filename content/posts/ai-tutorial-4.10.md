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