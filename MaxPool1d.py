import torch
from torch import nn
x = torch.tensor([[[1,2,3],[4,5,6]],
                  [[7,8,9],[10,11,12]],
                  [[13,14,15],[16,17,18]],
                  [[19,20,21],[22,23,24]]], dtype = torch.float32)
m = nn.MaxPool1d(2, stride = 1)
y = m(x)
y.shape
y

'''
Answer:
tensor([[[ 2.,  3.],
         [ 5.,  6.]],

        [[ 8.,  9.],
         [11., 12.]],

        [[14., 15.],
         [17., 18.]],

        [[20., 21.],
         [23., 24.]]])
'''
