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

now ,use the new metric start to train

```py
# start to train
learn = cnn_learner(dls, resnet50, metrics=partial(accuracy_multi, thresh=0.2))
learn.fine_tune(3, base_lr=3e-3, freeze_epochs=4)

# epoch train_loss valid_loss accuracy_multi time
# 0 0.942215 0.698972 0.239303 00:26
# 1 0.824776 0.551198 0.290996 00:26
# 2 0.607759 0.198789 0.827131 00:26
# 3 0.361537 0.125557 0.943287 00:26
# epoch train_loss valid_loss accuracy_multi time
# 0 0.134416 0.125471 0.934343 00:27
# 1 0.118428 0.105183 0.949880 00:27
# 2 0.097109 0.102836 0.950040 00:27
```

after train we can chnage the metrics with different value of thresh If you pick a threshold that's too low, you'll often be failing to select correctly labeled objects

```py
learn.metrics = partial(accuracy_multi, thresh=0.1)
learn.validate()
# validation loss and metrics

# [0.10283613950014114,0.9265138506889343]
```

you'll only be selecting the objects for which your model is very confident with a high thresh:

```py
learn.metrics = partial(accuracy_multi, thresh=0.99)
learn.validate()
# [0.10283613950014114,0.9433467388153076]
```

we calculate one pred to test different thresh value

```py
# we can getpre and 
# the get_preds get all the valid data ,to calculate their pred,and the target
preds,targs = learn.get_preds()

# (tensor([[1.3728e-03, 3.1368e-03, 4.9623e-04,  ..., 4.6060e-01, 1.1935e-03, 9.1202e-02],
#          [3.3482e-04, 1.2069e-02, 1.0969e-03,  ..., 1.7647e-02, 1.6421e-03, 9.6689e-04],
#          [3.8831e-03, 1.3268e-02, 4.4939e-03,  ..., 1.1680e-02, 1.0579e-03, 2.7399e-03],
#          ...,
#          [3.9797e-03, 5.1892e-03, 7.0612e-04,  ..., 3.0286e-03, 1.7749e-03, 7.1625e-03],
#          [8.4477e-03, 7.8008e-03, 1.7175e-03,  ..., 2.0243e-03, 2.5596e-02, 1.8781e-03],
#          [6.2252e-04, 9.4245e-01, 4.3180e-03,  ..., 8.6979e-03, 8.4691e-04, 5.2906e-03]]),
#  TensorMultiCategory([[0., 0., 0.,  ..., 0., 0., 0.],
#          [0., 0., 0.,  ..., 0., 0., 0.],
#          [0., 0., 0.,  ..., 0., 0., 0.],
#          ...,
#          [0., 0., 0.,  ..., 0., 0., 0.],
#          [0., 0., 0.,  ..., 0., 0., 0.],
#          [0., 1., 0.,  ..., 0., 0., 0.]]))
```

```py
# at default ,get_preds applies the output activation function (sigmoid, in this case) for us, so we'll need to tell accuracy_multi to not apply it:
accuracy_multi(preds, targs, thresh=0.9, sigmoid=False)
```

use this way to find the best thresh value
```
xs = torch.linspace(0.05,0.95,29)
accs = [accuracy_multi(preds, targs, thresh=i, sigmoid=False) for i in xs]
plt.plot(xs,accs);
```

![graph_tresh](/img/ai_t/t1/graph_tresh.PNG)

In this case, we're using the validation set to pick a hyperparameter (the threshold), which is the purpose of the validation set. Sometimes students have expressed their concern that we might be overfitting to the validation set, since we're trying lots of values to see which is the best. However, as you see in the plot, changing the threshold in this case results in a smooth curve, so we're clearly not picking some inappropriate outlier. This is a good example of where you have to be careful of the difference between theory (don't try lots of hyperparameter values or you might overfit the validation set) versus practice (if the relationship is smooth, then it's fine to do this).

在這種情況下，我們使用驗證集來選擇一個超參數（閾值），這是驗證集的目的。 有時，學生表達了他們對我們可能過度適合驗證集的擔憂，因為我們正在嘗試大量的值以查看哪種值最好。 但是，正如您在圖中所看到的，在這種情況下，更改閾值會產生平滑的曲線，因此我們顯然不會選擇一些不合適的離群值。 這是一個很好的例子，說明您必須注意理論（不要嘗試過多的超參數值，否則可能會過度擬合驗證集）與實踐之間的差異（如果關係是平滑的，則可以這樣做） 。