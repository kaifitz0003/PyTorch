import torch
import torch.nn as nn
# 1d Data
X_1d = torch.tensor([1,2,3], dtype=torch.float32)

# Torch's Layernorm
norm = nn.LayerNorm(3)
X_norm = norm(X_1d)

# Manual Layernorm
mean = X_1d.mean()
variance = X_1d.var(unbiased = False) # Find the mean, and average the squared differences. ((1-2)^2 + (2-2)^2 + (3-2)^2)/3
norm_manual = (X_1d-mean)/(torch.sqrt(variance))
torch.allclose(X_norm,norm_manual)

# 2d Data
X_2d = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32)

# Torch's Layernorm
norm = nn.LayerNorm(3) # LayerNorm needs the row dimensions
X_norm = norm(X_2d)

# Compute mean and variance row wise.
mean = X_2d.mean(dim=1).unsqueeze(dim=1)
variance = X_2d.var(dim=1, unbiased = False).unsqueeze(dim=1)
norm_manual = (X_2d-mean)/(torch.sqrt(variance))
torch.allclose(X_norm,norm_manual) 
