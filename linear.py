import torch
import torch.nn as nn

# y = xA^t+b where x is the input, A^t is the transposed A, and b is the bias.
# Transposed is when the rows become columns and columns become rows.
# nn.linear multiplies the first row of x with all the columns of the weights, and then the 2nd row of x with all the columns, and repeats for all the rows of x.
# The # of columns of m and the # of rows of the weight must be the same.

x = torch.tensor([[1,2],[3,4]], dtype = torch.float32) # 2x2
m = nn.Linear(2,3) # 2x3
m.weight.data = torch.tensor([[5,8],[6,9],[7,0]], dtype = torch.float32) # 3x2
m.bias.data = torch.tensor([0,0,0], dtype = torch.float32) # Size of 3
y = m(x)
y
