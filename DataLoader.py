"""
We play with only DataLoader to see how they impact the data.
You can use TensorDataset and DataLoader
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
# Supervised learning
X = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
y = torch.tensor([16,17,18,19,20])

out = DataLoader(X, shuffle=False,batch_size = 1)
for i in out:
  print(i)
  print('--------')
