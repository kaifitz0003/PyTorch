import torch
from torch import nn

x = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype = torch.float32)
torch.flatten(x) # tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.])

y = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]], dtype = torch.float32)
torch.flatten(y) # tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.])
