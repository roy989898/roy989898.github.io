+++
title = "Ai Tutorial 6.3 Other Computer Vision Problems-Multi-Label Binary Cross-Entropy"
date = "2021-05-16T16:59:46+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch","Multi-Label Classification","Cross-Entropy"]
keywords = ["", ""]
description = ""
showFullContent = false
+++
# Binary Cross-Entropy

a Learner object contains four main things: the model, a DataLoaders object, an Optimizer, and the loss function to use.
we use resnet models (teach later),we know howto build SGD optimizer(lesson 4) and the dataloader,so we look focus on the _loss function_.

```py
learn = cnn_learner(dls, resnet18)
```
seeone batch
```py
x,y = to_cpu(dls.train.one_batch())
x[0].shape

# torch.Size([3, 128, 128])
```

```py
# pass the independernt vairable to the model,to gte the activs 
activs = learn.model(x)
activs.shape
# torch.Size([64, 20])
```
why is this shape???torch.Size([64, 20]), because the match size is 64,and we have 20 categories,the activs, is for each image,the probability of each of 20 categories

```py
activs
# tensor([[ 0.7476, -1.1988,  4.5421,  ...,  0.7063, -1.3358, -0.3715],
#         [-0.9919, -0.4608, -0.4424,  ..., -1.4165, -2.9962,  0.5873],
#         [ 2.1179, -0.0294,  0.7001,  ...,  2.2310,  1.1888, -0.0595],
#         ...,
#         [-0.3535,  3.0212,  0.4811,  ...,  1.8732,  1.2486, -3.3234],
#         [-1.4724, -2.8740, -1.2860,  ..., -2.7895, -1.8632, -0.1557],
#         [-1.6487,  1.5647,  1.0682,  ..., -0.6979, -1.5629, -1.7217]], grad_fn=<MmBackward>)
```