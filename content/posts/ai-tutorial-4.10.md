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
```py
valid_dl = DataLoader(valid_dset, batch_size=256)
```
create a 4 size batch for test
```py
batch = train_x[:4]
batch.shape
# torch.Size([4, 784])
```
alcaulate the predict
[linear1]({{< ref "posts/ai-tutorial-4.8.md#predict--multi-image" >}} "linear1")
```py
preds = linear1(batch)
preds
''' tensor([[-4.5725],
        [ 0.2557],
        [-5.5496],
        [ 3.6488]], grad_fn=<AddBackward0>) '''
```

calculate a loss 
```py
loss = mnist_loss(preds, train_y[:4])
loss
# tensor(0.6119, grad_fn=<MeanBackward0>)
```
Now we can calculate the gradients:
```py
loss.backward()
weights.grad.shape,weights.grad.mean(),bias.grad
# (torch.Size([784, 1]), tensor(-0.0103), tensor([-0.0712]))
```

## put above together to create calc_grad functions
```py
def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()
```

test
```py
calc_grad(batch, train_y[:4], linear1)
weights.grad.mean(),bias.grad
# (tensor(-0.0207), tensor([-0.1423]))
```

run again
```py
calc_grad(batch, train_y[:4], linear1)
weights.grad.mean(),bias.grad
# (tensor(-0.0310), tensor([-0.2135]))
```
have probelm!!!!! we expect the grad should be the same ,becasue all the parameter of the calc_grad is same,but not!!!
becasue because the loss.backward add the gradients of loss to any gradients that are currently stored. So, we have to set the current gradients to 0 first:
```py
weights.grad.zero_()
bias.grad.zero_();
```

```py
# each epoch function
# params already use in model
def train_epoch(model, lr, params):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad*lr
            p.grad.zero_()
```

That gives us this function to calculate our validation accuracy:

```py
def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()
```
We can check it works:
linear1 calculate the prediction
```py
batch_accuracy(linear1(batch), train_y[:4])
```