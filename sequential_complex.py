import torch
import torch.nn as nn

X = torch.tensor([[[1]],[[0]],[[1]],[[2]],[[3]]], dtype = torch.float32) # 5x1x1
y = torch.tensor([0,0,0,1,1])
in_channels = 1
out_channels = 1
kernel_size = 1
Conv1d = nn.Conv1d(in_channels, out_channels,kernel_size)
max_pool = nn.MaxPool1d(1)
Linear = nn.Linear(1,1)
sig = nn.Sigmoid()

model = nn.Sequential(Conv1d, max_pool, Linear, sig)
model(X)
