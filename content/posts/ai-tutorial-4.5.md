+++
title = "Ai Tutorial 4.5 Gredient"
date = "2021-04-28T11:15:32+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch","Gradients"]
keywords = ["", ""]
description = ""
showFullContent = false
+++
[My Code](https://colab.research.google.com/drive/1rMfM4H92wklMLDydjnChmJMHoJ3OS6SL?usp=sharing)
[Source Code](https://colab.research.google.com/github/fastai/fastbook/blob/master/04_mnist_basics.ipynb)
# _Gradients_

[explain for the Gredient
](https://www.khanacademy.org/math/differential-calculus/dc-diff-intro)

## calculate for the gradient in program

```py

def f(x): return x**2

```

 [the function in 4.4]({{< ref "posts/ai-tutorial-4.4.md" >}} "the function in 4.4")

```python
 # select a tensor to calculate the grad
xt = tensor(3.).requires_grad_()
xt
```

```python
yt = f(xt)
yt

```

```py
# calculate the gradients
yt.backward()

```

``` python
# see the grad,answer is 6
xt.grad
# tensor(6.)
```

## another example

```py

xt = tensor([3.,4.,10.]).requires_grad_()

def f(x): return (x**2).sum()

yt = f(xt)

yt.backward()
xt.grad
```
