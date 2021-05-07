+++
title = "Ai Tutorial 5.3 Image Classification >2 types Cross-entropy loss 2"
date = "2021-05-07T11:46:30+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch","cross-entropy loss","log"]
keywords = ["", ""]
description = ""
showFullContent = false
+++

[My Code
](https://colab.research.google.com/drive/1Rqum2194iz5nXH26PPoBMpKM71wQ4eYI?usp=sharing)

[Source Code
](https://colab.research.google.com/github/fastai/fastbook/blob/master/05_pet_breeds.ipynb#scrollTo=YOTrrdP7BuWd)

# Cross-entropy loss 2

although softmax+ log Likelihood look like very suitable as a loss function.But the problem is we are using probabilities, 1>=p>=0.That mean when the model see 0.99 and 0.999, they are very close,but in another sense, 0.999 is 10 times more confident than 0.99. So, we want to transform our numbers between 0 and 1 to instead be between negative infinity and 0.Log!!!!!

## Taking the Log

```py
plot_function(torch.log, min=-5,max=4)
```

![log](/img/ai_t/t1/log.PNG)

log in python

```py
y = b**a
a = log(y,b)
```

```py
log(a*b) = log(a)+log(b)
```

at default, the pYtorch use e=2.718 as the log basic

in the Pytorch,nll_loss awsume you get the log of the softmax,so it do not the log.
softmax+log+nll_loss==log_softmax+nll_loss==nn.CrossEntropyLoss()

```py
# log_softmax ->nll_loss,cross-entropy loss!!!!!
loss_func = nn.CrossEntropyLoss()

```

```py
loss_func(acts, targ)
# tensor(1.7790)
```

same

```py
# same
F.cross_entropy(acts, targ)
# tensor(1.7790)
```

```py
# at default,will take all the loss mean
# reduction='none' disable
nn.CrossEntropyLoss(reduction='none')(acts, targ)
```

we do some testing to prove `softmax+log+nll_loss==log_softmax+nll_loss==nn.CrossEntropyLoss()`

```py
sm_acts2 = torch.softmax(acts, dim=1)
sm_acts2
# tensor([[0.7795, 0.2205],
#         [0.8902, 0.1098],
#         [0.1517, 0.8483],
#         [0.5245, 0.4755],
#         [0.9956, 0.0044],
#         [0.8464, 0.1536]])
```

```py
torch.log(sm_acts2)
# tensor([[-2.4908e-01, -1.5119e+00],
#         [-1.1630e-01, -2.2091e+00],
#         [-1.8857e+00, -1.6455e-01],
#         [-6.4534e-01, -7.4335e-01],
#         [-4.4367e-03, -5.4201e+00],
#         [-1.6675e-01, -1.8735e+00]])
```

```py
# test ,equal the above code,softmax + log
sfm=torch.log_softmax(acts, dim=1)
sfm
# tensor([[-2.4908e-01, -1.5119e+00],
#         [-1.1630e-01, -2.2091e+00],
#         [-1.8857e+00, -1.6455e-01],
#         [-6.4534e-01, -7.4335e-01],
#         [-4.4366e-03, -5.4201e+00],
#         [-1.6675e-01, -1.8735e+00]])
```

```py
F.nll_loss(sfm, targ, reduction='none')
# tensor([0.2491, 2.2091, 1.8857, 0.7434, 5.4201, 0.1667])
```

# Model Interpretation

use a confusion matrix to see where our model is doing well, and where it's doing badly:

```py
#width 600
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
```

![pet_matrix](/img/ai_t/t1/pet_matrix.PNG)

too diccfcult to read

```py
# only show the most bad 
# min_val 5 mean the wrong at least is 5
interp.most_confused(min_val=5)
# [('Ragdoll', 'Birman', 9),
#  ('american_pit_bull_terrier', 'staffordshire_bull_terrier', 8),
#  ('Bengal', 'Egyptian_Mau', 6)]
# actual Egyptian_Mau ,but predict Bengal is9

# search at google, we can foind the they really need to classifer by a humanexpert, so it is ok

```

new we have a good model,how can we make it better?
