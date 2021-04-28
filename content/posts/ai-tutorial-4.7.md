+++
title = "Ai Tutorial 4.7 An End-to-End SGD Example"
date = "2021-04-28T11:57:33+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch","LR","stepping","SGD"]
keywords = ["", ""]
description = ""
showFullContent = false
+++


# _An End-to-End SGD Example_
we want to find the smallest value


## Some useful function
craete a 0-19 torch array 
```py
time = torch.arange(0,20).float(); time
# tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.])
```

create randome number
```py

# 返回一個張量，包含了從標準正態分佈（均值為0，方差為1，即高斯白噪聲）中抽取的一組隨機數。張量的形狀由參數sizes定義。
num=20
t=torch.randn(num)
time_f = torch.arange(0,num).float(); time
plt.scatter(time_f,t);
t
```
![rt](/img/ai_t/t1/rt.PNG)

simulate a car speed
```py
# simulate a car speed
# torch.randn(20)*3 is some random noise
time = torch.arange(0,20).float(); time
speed = torch.randn(20)*3 + 0.75*(time-9.5)**2 + 1
plt.scatter(time,speed);
```
![car_speed](/img/ai_t/t1/car_speed.PNG)

## use SGD to find the smallest value for the loss
### Step 0 gues the functions
we nedd to find the a,b,c that make the loss is the lowset
(time**2)+(b*time)+c

```py
def f(t, params):
    a,b,c = params
    return a*(t**2) + (b*t) + c
```

### Step 0.1 define the meaning of best
we use a loss function to define the best, which will return a value based on a prediction and a target, where lower values of the function correspond to "better" predictions. For continuous data, it's common to use mean squared error:

```py
def mse(preds, targets): return ((preds-targets)**2).mean().sqrt()
```

### Step 1 set the apramter as a randome value
```py
params=None
params = torch.randn(3).requires_grad_()
orig_params = params.clone()
params
```

### Step 2 calculate the predict

```py
preds = f(time, params)

def show_preds(preds, ax=None):
    if ax is None: ax=plt.subplots()[1]
    ax.scatter(time, speed)
    # to_npconvert tensor to numpy arry
    ax.scatter(time, to_np(preds), color='red')
    ax.set_ylim(-300,100)

show_preds(preds)
```

![pred1](/img/ai_t/t1/pred1.PNG)

### Step 3 calculate the losses

```py
loss = mse(preds, speed)
loss
# tensor(25.1871, grad_fn=<SqrtBackward>)
```

### Step 4  know the gradients

```py
loss.backward()
params.grad
# the a b c gradients
# tensor([-3.1634, -0.2709, -0.3931])
```

### Step 5  Step the weights

```py
lr = 1e-5
# assign the chnaged parameter to the params
params.data -= lr * params.grad
params.grad = None
```

Let's see if the loss has improved:

```py

# Let's see if the loss has improved:
preds = f(time,params)
mse(preds, speed)
show_preds(preds)
# improve a little bit
```
![pred1](/img/ai_t/t1/ip.PNG)


### step 6 , repeat it
# we use a for loop to do multi time
```py
def apply_step(params, prn=True):
    preds = f(time, params)
    loss = mse(preds, speed)
    loss.backward()
    params.data -= lr * params.grad.data
    params.grad = None
    if prn: print(loss.item())
    return preds
```

```py
for i in range(10): apply_step(params)

# 160.42279052734375
# 160.14772033691406
# 159.87269592285156
# 159.59768676757812
# 159.3227081298828
# 159.04774475097656
# 158.7728271484375
# 158.4979248046875
# 158.22305297851562
# 157.9481964111328
```

```py
_,axs = plt.subplots(1,4,figsize=(12,3))
for ax in axs: show_preds(apply_step(params, False), ax)
plt.tight_layout()
```
![4p](/img/ai_t/t1/4p.PNG)

### Step7 stop
we do 10 round ,than stop**