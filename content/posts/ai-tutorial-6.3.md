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

# some python basic

partial

```py
def say_hello(name, say_what="Hello"): return f"{say_what} {name}."
say_hello('Jeremy'),say_hello('Jeremy', 'Ahoy!')
# ('Hello Jeremy.', 'Ahoy! Jeremy.')

```

```py
f = partial(say_hello, say_what="Bonjour")
f("Jeremy"),f("Sylvain")
# ('Bonjour Jeremy.', 'Bonjour Sylvain.')
```

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

we can see that the number still not between 0 and 1,but we can use the the loss function learn in lesson 4(mist_loss,because have sigmoid) and add log

```py
def binary_cross_entropy(inputs, targets):
    inputs = inputs.sigmoid()
    return -torch.where(targets==1, 1-inputs, inputs).log().mean()
```

why we do not use the nll_loss or softmax thta lear in lesson 5????becuase it use for one image one tag,but ther is one imagfe maybe >1 tag or 0 tag

* **softmax**, as we saw, requires that all predictions sum to 1, and tends to push one activation to be much larger than the others (due to the use of exp); however, we may well have multiple objects that we're confident appear in an image, so restricting the maximum sum of activations to 1 is not a good idea. By the same reasoning, we may want the sum to be less than 1, if we don't think any of the categories appear in an image.
* **nll_loss**, as we saw, returns the value of just one activation: the single activation corresponding with the single label for an item. This doesn't make sense when we have multiple labels.

pytorch already provide binary_cross_entropy

```py
loss_func = nn.BCEWithLogitsLoss()
loss = loss_func(activs, y)
loss

# TensorMultiCategory(1.0342, grad_fn=<AliasBackward>)
```

However ,we do not need to require fastai use this loss function,!!! Becasue if fastai dataloaders know have multi categories ta a image, default use nn.BCEWithLogitsLoss

we need to change the metric too,compare to the lesson 5

```py
# orginal one
def accuracy(inp, targ, axis=-1):
    "Compute accuracy with `targ` when `pred` is bs * n_classes"
    # select the mots hight valu one.but know we have multi category for a image
    pred = inp.argmax(dim=axis)
    return (pred == targ).float().mean()
```

```py
# suitable one
# we need to set a value:thresh,to decide which is 1,whis is 0
def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Compute accuracy when `inp` and `targ` are the same size."
    if sigmoid: inp = inp.sigmoid()
    return ((inp>thresh)==targ.bool()).float().mean()
```
