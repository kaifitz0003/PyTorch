import torch
import torch.nn as nn

# Data
X = torch.tensor([[[1,2,3,4],[5,6,7,8],[9,10,11,12]]],dtype = torch.float32)
N,Cin,Lin = X.shape

# Model
Cout = 4
K = 3
model = nn.Sequential(
    nn.Conv1d(in_channels=Cin,out_channels=Cout,kernel_size=K)) 

# Runs X through model to compute output
model(X)
