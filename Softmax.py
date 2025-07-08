"""
Softmax rescales the data so it adds to 1 and each data point is between 0 and 1.

The scaled numbers are not proportional. However, the relative sizes remain the same. 
For example, the biggest number stays the biggest, the smallest stays the smallest.

Dimension means which way Softmax looks at the data. For example, if the data is 2d and dim=0, Softmax will go over the rows. 
Instead, if dim = 1, it will go over the columns.

"""
import numpy as np
import torch
import torch.nn as nn

logits = torch.tensor([1.,4.,-1.])
model = nn.Softmax(dim = 0) # Goes over the rows. Since the data is 1d, dim is not required. 
model(logits)

e = np.exp(1) # 2.718281, special number like pi
e**1/(e**1+e**4+e**-1) # Does Softmax for 1. The output is 0.0471
e**4/(e**1+e**4+e**-1) # Does Softmax for 4. The output is 0.9465
e**-1/(e**1+e**4+e**-1) # Does Softmax for -1. The output is 0.0064
