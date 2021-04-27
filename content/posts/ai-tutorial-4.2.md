+++
title = "Ai Tutorial 4.2"
date = "2021-04-27T18:14:43+08:00"
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ai", "fastai","pytorch","寫給程式設計師的深度學習：使用fastai和PyTorch"]
keywords = ["", ""]
description = ""
showFullContent = false
+++
# _NumPy Arrays and PyTorch Tensors
they almost the same but  NumPy Arrays not support GPU
```py
data = [[1,2,3],[4,5,6]]
arr = array (data)
tns = tensor(data)
```

```py
arr  # numpy
# array([[1, 2, 3],
#        [4, 5, 6]])
```

```py
tns  # pytorch

# tensor([[1, 2, 3],
#         [4, 5, 6]])
```

```py
tns[1]
# get index 1
# tensor([4, 5, 6])

tns[:,1]
# all first axis,index 1 at ssecond axis
# tensor([2, 5])


tns[1,1:3]
# first axis :1,sendoc axis 1-3(exclude)
# tensor([5, 6])

tns+1
# tensor([[2, 3, 4],
#         [5, 6, 7]])

tns*1.5
# tensor([[1.5000, 3.0000, 4.5000],
#         [6.0000, 7.5000, 9.0000]])
```