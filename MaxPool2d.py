import torch
from torch import nn
x = torch.tensor([[[[1,2,3],[4,5,6],[7,8,9]],   [[10,11,12],[13,14,15],[16,17,18]]]], dtype = torch.float32)
m = nn.MaxPool2d((2,2), stride = (1,1))
y = m(x)
y.shape
y

'''
Anwser:
tensor([[[[ 5.,  6.],
          [ 8.,  9.]],

         [[14., 15.],
          [17., 18.]]]])
'''
