+++
title = "Ai Tutorial 4.12 Adding a Nonlinearity"
date = "2021-04-29T14:45:20+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch","Nonlinearity"]
keywords = ["", ""]
description = ""
showFullContent = false
+++

# Adding a Nonlinearity

```py
def simple_net(xb): 
    res = xb@w1 + b1
    res = res.max(tensor(0.0))
    res = res@w2 + b2
    return res
```

```py
# init the w and b just like we did in the previous section:
w1 = init_params((28*28,30))
b1 = init_params(30)
w2 = init_params((30,1))
b2 = init_params(1)
```

why w1 = init_params((28*28,30)) is 30???? That means that the first layer can construct 30 different features, each representing some different mix of pixels. You can change that 30 to anything you like, to make the model more or less complex.

w2 neeed to match w1,so 30 too

## rectified linear unit ,整流線性單元,RelU

what is res.max(tensor(0.0))???rectified linear unit ,整流線性單元,RelU,in other words, replace every negative number with a zero. This tiny function is also available in PyTorch as F.relu:

Why ????? The basic idea is that by using more linear layers,  can have our model do more computation, and therefore model more complex functions. But because when we multiply things together and then add them up multiple times, that could be replaced by multiplying different things together and adding them up just once! That is to say, a series of any number of linear layers in a row can be replaced with a single linear layer with a different set of parameters.

But if we put a nonlinear function between them, such as max, then this is no longer true. Now each linear layer is actually somewhat decoupled from the other ones, and can do its own useful work. The max function is particularly interesting, because it operates as a simple if statement.

為什麼 ？？？？？ 基本思想是，通過使用更多的線性層，可以使我們的模型進行更多的計算，從而為更複雜的函數建模。 但是，因為當我們將事物相乘然後多次相加時，可以通過將不同事物相乘並僅相加一次來代替！ 也就是說，可以將一行中任意數量的線性層中的一系列序列替換為具有不同參數集的單個線性層。

但是，如果我們在它們之間放置一個非線性函數（例如max），則不再適用。 現在，每個線性層實際上都已與其他線性層解耦，並且可以做自己有用的工作。 max函數特別有趣，因為它作為簡單的if語句運行。

Amazingly enough, it can be mathematically proven that this little function can solve any computable problem to an arbitrarily high level of accuracy, if you can find the right parameters for w1 and w2 and if you make these matrices big enough. For any arbitrarily wiggly function, we can approximate it as a bunch of lines joined together; to make it closer to the wiggly function, we just have to use shorter lines. This is known as the universal approximation theorem. The three lines of code that we have here are known as layers. The first and third are known as linear layers, and the second line of code is known variously as a nonlinearity, or activation function.

Just like in the previous section, we can replace this code with something a bit simpler, by taking advantage of PyTorch:

足夠令人驚訝的是，如果可以找到w1和w2的正確參數，並且使這些矩陣足夠大，則可以用數學方式證明此小函數可以以任意高的精度解決任何可計算的問題。 對於任何任意擺動的函數，我們可以將其近似為一束連接在一起的線。 為了使其更接近任意擺動的函數，我們只需要使用較短的線即可。 這被稱為通用近似定理。 我們在這裡擁有的三行代碼稱為層。 第一和第三層被稱為線性層，第二行代碼被不同地稱為非線性或激活函數。

就像在上一節中一樣，我們可以利用PyTorch將代碼替換為更簡單的代碼：

```py
# floow the sequence Linear ->ReLU->Linear
simple_net = nn.Sequential(
    nn.Linear(28*28,30),
    nn.ReLU(),
    nn.Linear(30,1)
)
```

```py
learn = Learner(dls, simple_net, opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)
```

```py
# lr  =0.1
# eopch num 40
learn.fit(40, 0.1)
```

```py
plt.plot(L(learn.recorder.values).itemgot(2));
```

![sl](/img/ai_t/t1/sl.PNG)
we can see that, 1.A function that can solve any problem to any level of accuracy (the neural network) given the correct set of parameters
2.A way to find the best set of parameters for any function (stochastic gradient descent)

# Going Deeper

if this can approximate any function with a single nonlinearity with two linear layers,why we nee to go deeper???? because performance With a deeper model (that is, one with more layers) we do not need to use as many parameters; it turns out that we can use smaller matrices with more layers, and get better results than we would get with larger matrices, and few layers.

that mean we can train the mode lquicky,smaller memory

```py
# 18 layer,only one epoch,90%!!!
dls = ImageDataLoaders.from_folder(path)
learn = cnn_learner(dls, resnet18, pretrained=False,
                    loss_func=F.cross_entropy, metrics=accuracy)
learn.fit_one_cycle(1, 0.1)
```

# Some term

**Activations**:: Numbers that are calculated (both by linear and nonlinear layers)
**Parameters**:: Numbers that are randomly initialized, and optimized (that is, the numbers that define the model)

Our activations and parameters are all contained in tensors. These are simply regularly shaped arrays—for example, a matrix. Matrices have rows and columns; we call these the axes or dimensions. The number of dimensions of a tensor is its rank. There are some special tensors:

Rank zero: scalar Rank one: vector Rank two: matrix

A neural network contains a number of layers. Each layer is either linear or nonlinear. We generally alternate between these two kinds of layers in a neural network. Sometimes people refer to both a linear layer and its subsequent nonlinearity together as a single layer. Yes, this is confusing. Sometimes a nonlinearity is referred to as an **activation function**.

## Deep learning vocabulary

| Term | Meaning|
| ---- | ---- |
|ReLU | Function that returns 0 for negative numbers and doesn't change positive numbers.|
|Mini-batch | A small group of inputs and labels gathered together in two arrays. A gradient descent step is updated on this batch (rather than a whole epoch).|
|Forward pass | Applying the model to some input and computing the predictions.|
|Loss | A value that represents how well (or badly) our model is doing.|
|Gradient | The derivative of the loss with respect to some parameter of the model.|
|Backward pass | Computing the gradients of the loss with respect to all model parameters.|
|Gradient descent | Taking a step in the directions opposite to the gradients to make the model parameters a little bit better.|
|Learning rate | The size of the step we take when applying SGD to update the parameters of the model.
