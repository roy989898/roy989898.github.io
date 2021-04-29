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


we can make the above code more general to use

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
