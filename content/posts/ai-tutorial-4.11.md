+++
title = "Ai Tutorial 4.11 Creating an Optimizer"
date = "2021-04-29T11:23:16+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch","SGD"]
keywords = ["", ""]
description = ""
showFullContent = false
+++
[My Code](https://colab.research.google.com/drive/1rMfM4H92wklMLDydjnChmJMHoJ3OS6SL?usp=sharing)
[Source Code](https://colab.research.google.com/github/fastai/fastbook/blob/master/04_mnist_basics.ipynb)
# Creating an Optimizer

## we can make the above code more general to use

```py

# use nn.Linear to replace the linear1
# it do the same thing with the linear1 and init_params
linear_model = nn.Linear(28*28,1)
```

```py
#  we can get the paramater, weight,basic
w,b = linear_model.parameters()
w.shape,b.shape,b

# (torch.Size([1, 784]), torch.Size([1]), Parameter containing:
#  tensor([-0.0180], requires_grad=True))
```

```py
class BasicOptim:
    def __init__(self,params,lr): self.params,self.lr = list(params),lr

    def step(self, *args, **kwargs):
        for p in self.params: p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params: p.grad = None
```

```py
opt = BasicOptim(linear_model.parameters(), lr)
```

simplfy the trainb loop

```py
# use it
def train_epoch(model):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()
```

[calc_grad]({{< ref "posts/ai-tutorial-4.10.md#put-above-together-to-create-calc_grad-functions" >}} "calc_grad")

```py
def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model), end=' ')
```

```py

train_model(linear_model, 20)
# same with above code
```

## Actually ,fastai already have the same thing

to replace the BasicOptim

```py
linear_model = nn.Linear(28*28,1)
opt = SGD(linear_model.parameters(), lr)
train_model(linear_model, 20)

```

to replace the train train_model

```py
dls = DataLoaders(dl, valid_dl)
```

```py

learn = Learner(dls, nn.Linear(28*28,1), opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)
```

nn.Linear:how to predict the value  
opt_func:howw to change the weight  
loss_func:how to calculate the loss  
metrics:how to calculate the metrics  

[mnist_loss]({{< ref "posts/ai-tutorial-4.8.md#better-loss-finction" >}} "mnist_loss")
[batch_accuracy]({{< ref "posts/ai-tutorial-4.10.md#put-above-together-to-create-calc_grad-functions" >}} "batch_accuracy")

```py
learn.fit(10, lr=lr)
```

```py
# epoch train_loss valid_loss batch_accuracy time
# 0 0.636991 0.503566 0.495584 00:00
# 1 0.553366 0.176069 0.857704 00:00
# 2 0.202398 0.188561 0.829244 00:00
# 3 0.088171 0.108241 0.912169 00:00
# 4 0.046019 0.078468 0.932287 00:00
# 5 0.029606 0.062658 0.947498 00:00
# 6 0.022893 0.052850 0.955839 00:00
# 7 0.019928 0.046356 0.962218 00:00
# 8 0.018433 0.041814 0.966143 00:00
# 9 0.017540 0.038480 0.968106 00:00

```
