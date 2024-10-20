import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
# Supervised learning
X = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
y = torch.tensor([16,17,18,19,20])
dataset = TensorDataset(X,y)

out = DataLoader(dataset, shuffle=True,batch_size = 1)
for i,j in out:
  print(i)
  print(j)
  print('--------')
