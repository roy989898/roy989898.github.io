+++
title = "Ai Tutorial 4.3 Metric"
date = "2021-04-27T18:18:28+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch"]
keywords = ["", ""]
description = ""
showFullContent = false
+++

# _Computing Metrics Using Broadcasting_

#### Metric

 a metric is a number that is calculated based on the predictions of our model, and the correct labels in our dataset, in order to tell us how good our model is.  
 we want to calculate our metric over a validation set. This is so that we don't inadvertently overfit—that is, train a model to work well only on our training data  
 指標是根據我們的模型預測和數據集中的正確標籤計算出的數字，目的是告訴我們我們的模型有多好。  
  我們要根據驗證集計算指標。 這樣一來，我們就不會無意間過度擬合-也就是說，訓練模型只能在訓練數據上有效地工作

get the data

```py
valid_3_tens = torch.stack([tensor(Image.open(o)) 
                            for o in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float()/255
valid_7_tens = torch.stack([tensor(Image.open(o)) 
                            for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens.float()/255
valid_3_tens.shape,valid_7_tens.shape

# (torch.Size([1010, 28, 28]), torch.Size([1028, 28, 28]))
```

## Computing Metrics Using Broadcasting

write a function that canculate the distance

```py
def mnist_distance(a,b): return (a-b).abs().mean((-1,-2))
mnist_distance(a_3, mean3)
# tensor(0.1114)
```

for every image ,we do not need to write a loop ,we use Broadcasting

```py

valid_3_dist = mnist_distance(valid_3_tens, mean3)
valid_3_dist, valid_3_dist.shape
# (tensor([0.1329, 0.1555, 0.1107,  ..., 0.1359, 0.1526, 0.1126]),
#  torch.Size([1010]))
```

它沒有抱怨形狀不匹配，而是將每個單個圖像的距離作為長度為1,010（我們的驗證集中的3的數量）的向量（即1級張量）返回

當PyTorch嘗試在不同等級的兩個張量之間執行簡單的減法運算時，它將使用廣播。 也就是說，它將自動擴展具有較小等級的張量，使其具有與具有較大等級的張量相同的大小。 廣播是一項重要功能，可使張量代碼更易於編寫。

Instead of complaining about shapes not matching, it returned the distance for every single image as a vector (i.e., a rank-1 tensor) of length 1,010 (the number of 3s in our validation set).
 PyTorch, when it tries to perform a simple subtraction operation between two tensors of different ranks, will use broadcasting. That is, it will automatically expand the tensor with the smaller rank to have the same size as the one with the larger rank. Broadcasting is an important capability that makes tensor code much easier to write.

#### More Brodcast example

```py
# brodcast example
tensor([[1,2,3],[1,2,3]]) + tensor(1)
# tensor([[2, 3, 4],
#         [2, 3, 4]])
```

```py
(valid_3_tens-mean3).shape

# torch.Size([1010, 28, 28])
```

## is_3

```py
def is_3(x): return mnist_distance(x,mean3) < mnist_distance(x,mean7)
```

```py
is_3(a_3), is_3(a_3).float()
# (tensor(True), tensor(1.))
```

```py
is_3(valid_3_tens)
# tensor([True, True, True,  ..., True, True, True])
```

Now we can calculate the accuracy for each of the 3s and 7s by taking the average of that function for all 3s and its inverse for all 7s:

```py
accuracy_3s =      is_3(valid_3_tens).float() .mean()
accuracy_7s = (1 - is_3(valid_7_tens).float()).mean()

accuracy_3s,accuracy_7s,(accuracy_3s+accuracy_7s)/2

# (tensor(0.9168), tensor(0.9854), tensor(0.9511))
```

 over 90% accuracy on both 3s and 7s!!!!
