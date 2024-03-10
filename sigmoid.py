import torch
import torch.nn as nn

X = torch.tensor([1,2,3])
m = nn.Sigmoid() # 1/(1+np.exp(-X))
y = m(X)
y
