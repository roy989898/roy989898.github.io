+++
title = "Ai Tutorial 4.6 Stepping With a Learning Rate"
date = "2021-04-28T11:30:02+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch","LR","stepping"]
keywords = ["", ""]
description = ""
showFullContent = false
+++

# _Stepping With a Learning Rate_

when we get the gradient,we cau use it calculate the new paramter . Nearly all approaches start with the basic idea of multiplying the gradient by some small number, called the learning rate (LR). The learning rate is often a number between 0.001 and 0.1, although it could be anything Often, people select a learning rate just by trying a few, and finding which results in the best model after training (we'll show you a better approach later in this book, called the learning rate finder). Once you've picked a learning rate, you can adjust your parameters using this simple function:
w -= gradient(w) * lr  
This is known as *stepping* your parameters, using an *optimizer step*.
當我們得到梯度時，我們就用它來計算新的參數。 幾乎所有方法都始於將梯度乘以一個稱為學習率（LR）的小數的基本思想。 學習率通常是0.001到0.1之間的數字，儘管可以是任意數。通常，人們僅通過嘗試一些就可以選擇學習率，並在訓練後發現哪種模式可以得到最佳模型（我們將在稍後向您展示一種更好的方法 在這本書中，稱為學習率查找器）。 選擇學習速度後，您可以使用以下簡單功能調整參數：
w -= gradient(w) * lr  
使用“優化步”，這稱為“步進”你的參數。

if your Lr too small,maybe too slow,

![sgd_step](/img/ai_t/t1/step_small.PNG)


if LR too big,it can actually result in the loss getting worse,

![sgd_step](/img/ai_t/t1/strp_big1.PNG)

![sgd_step](/img/ai_t/t1/steo_big2.PNG)