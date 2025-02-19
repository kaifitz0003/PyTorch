"""
The below code is a simple example of MaxPool2d.

According to Alex Krizhevsky in 2012, we generally observe during training that models with overlapping
pooling find it slightly more difficult to overfit.

https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
https://en.wikipedia.org/wiki/AlexNet
"""

import torch
from torch import nn
x = torch.tensor([[[[1,2,3],[4,5,6],[7,8,9]],   [[10,11,12],[13,14,15],[16,17,18]]]], dtype = torch.float32)
m = nn.MaxPool2d((2,2), stride = (1,1)) # Stride shows how much to move the kernel over. 

y = m(x)
y.shape
y

'''
Answer:
tensor([[[[ 5.,  6.],
          [ 8.,  9.]],

         [[14., 15.],
          [17., 18.]]]])
'''
