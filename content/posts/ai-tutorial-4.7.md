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