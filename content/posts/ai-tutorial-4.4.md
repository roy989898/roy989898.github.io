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
# _SGD_
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

There are many different ways to do each of these seven steps

* Initialize:: initialize the parameters to random values. This may sound surprising. There are certainly other choices we could make, such as initializing them to the percentage of times that pixel is activated for that category—but since we already know that we have a routine to improve these weights, it turns out that just starting with random weights works perfectly well.
 將參數初始化為隨機值。 這聽起來可能令人驚訝。 當然，我們還可以做出其他選擇，例如將其初始化為該類別的像素被激活的次數的百分比-但由於我們已經知道我們有一個例程可以改善這些權重，因此事實證明，只是從隨機權重開始 效果很好。
* Loss:: when testing the effectiveness of any current weight assignment in terms of actual performance. We need some function that will return a number that is small if the performance of the model is good (the standard approach is to treat a small loss as good, and a large loss as bad, although this is just a convention).
  在實際性能方面測試任何當前重量分配的有效性時。 如果模型的性能良好，我們需要一些函數返回一個較小的數字（標準方法是將小的損失視為好，將大損失視為壞，儘管這只是一個慣例）。
* Step:: A simple way to figure out whether a weight should be increased a bit, or decreased a bit, would be just to try it: increase the weight by a small amount, and see if the loss goes up or down. Once you find the correct direction, you could then change that amount by a bit more, and a bit less, until you find an amount that works well. However, this is slow! As we will see, the magic of calculus allows us to directly figure out in which direction, and by roughly how much, to change each weight, without having to try all these small changes. The way to do this is by calculating gradients. This is just a performance optimization, we would get exactly the same results by using the slower manual process as well.
  一種簡單的判斷重量是否應該增加還是減少的簡單方法就是嘗試：將重量增加一點，然後看看損失是增加還是減少。 找到正確的方向後，您可以再多一點，少一點地更改該金額，直到找到一個行之有效的金額。 但是，這很慢！ 就像我們將看到的那樣，微積分的神奇之處使我們能夠直接弄清楚改變每個權重的方向和大致幅度，而不必嘗試所有這些小的改變。 做到這一點的方法是通過計算梯度。 這只是性能優化，通過使用較慢的手動過程，我們也將獲得完全相同的結果。
* Stop:: Once we've decided how many epochs to train the model for (a few suggestions for this were given in the earlier list), we apply that decision. This is where that decision is applied. For our digit classifier, we would keep training until the accuracy of the model started getting worse, or we ran out of time.
  一旦我們確定了訓練模型的時間（在前面的列表中給出了一些建議），我們就會應用該決定。 這就是應用該決定的地方。 對於我們的數字分類器，我們將繼續訓練直到模型的準確性開始變差或用完為止。

## simple example of SGD
```py
def f(x): return x**2
```

```py
plot_function(f, 'x', 'x**2')
```

![sgd_step](/img/ai_t/t1/x2p.PNG)