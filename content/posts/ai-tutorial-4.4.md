+++
title = "Ai Tutorial 4.4 Stochastic Gradient Descent 隨機梯度下降 (SGD)"
date = "2021-04-27T19:56:41+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch","sgd","SGD"]
keywords = ["", ""]
description = ""
showFullContent = false
+++
Instead of trying to find the similarity between an image and an "ideal image," we could instead look at each individual pixel and come up with a set of weights for each one, such that the highest weights are associated with those pixels most likely to be black for a particular category. For instance, pixels toward the bottom right are not very likely to be activated for a 7, so they should have a low weight for a 7, but they are likely to be activated for an 8, so they should have a high weight for an 8. This can be represented as a function and set of weight values for each possible category—for instance the probability of being the number 8:    

與其嘗試查找圖像與“理想圖像”之間的相似性，不如查看每個單獨的像素並為每個像素提出一組權重，以使最高的權重與最有可能與之相關的那些像素相關聯。 對於特定類別為黑色。 例如，朝右下角移動的像素不太可能為7激活，因此對於7,像素應該具有較低的權重，但對於8,像素應該很容易被激活，因此對於8,像素應該具有較高的權重.這可以表示為每個可能類別的一個函數和一組權重值，例如，成為數字8的概率：  
`def pr_eight(x,w): return (x*w).sum()`  
x is the image, represented as a vector—in other words, with all of the rows stacked up end to end into a single long line. And we are assuming that the weights are a vector w. If we have this function, then we just need some way to update the weights to make them a little bit better. With such an approach, we can repeat that step a number of times, making the weights better and better, until they are as good as we can make them.  

x是表示為矢量的圖像，換句話說，所有行首尾相連地排成一條長線。 並且我們假設權重是向量w。 如果我們具有此功能，那麼我們只需要一些方法來更新權重即可使它們更好一點。 通過這種方法，我們可以重複該步驟多次，使權重越來越好，直到權重達到我們所能達到的程度為止。

want to find the specific values for the vector w that causes the result of our function to be high for those images that are actually 8s, and low for those images that are not. Searching for the best vector w is a way to search for the best function for recognising 8s.  

想要找到向量w的特定值，該值導致函數的結果對於那些實際上是8s的圖像來說較高，而對於那些不是8s的圖像來說較低。 搜索最佳向量w是搜索識別8s的最佳函數的一種方式。


1. Initialize the weights.初始化權重。
2. For each image, use these weights to predict whether it appears to be a 3 or a 7.對於每個圖像，使用這些權重來預測它是3還是7。
3. Based on these predictions, calculate how good the model is (its loss).根據這些預測，計算模型的好壞（損失）。
4. Calculate the gradient, which measures for each weight, how changing that weight would change the loss.計算坡度，該坡度針對每個權重進行度量，更改該權重將如何改變損耗
5. Step (that is, change) all the weights based on that calculation.根據該計算步進（即更改）所有權重。
6. Go back to the step 2, and repeat the process.返回到步驟2，並重複該過程。
7. Iterate until you decide to stop the training process (for instance, because the model is good enough or you don't want to wait any longer).重複進行直到您決定停止訓練過程為止（例如，因為模型足夠好或者您不想再等待了）。

   
![sgd_step](/img/ai_t/t1/sgd_step.PNG)