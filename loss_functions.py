import torch
from torch import nn


a = torch.tensor([1,2,3], dtype = torch.float32)
b = torch.tensor([4,7,5], dtype = torch.float32)

# MAE (mean absolute error) is computed by finding the absolute value of the corresponding values, and then adding them and finding the mean.
# MAE is also known as L1 loss
L1loss_fn = nn.L1Loss()
L1loss = L1loss_fn(a, b) # (|4-1|+|7-2|+|5-3|)/3 = (3+5+2)/3 = (10)/3 = 3.33333

# MSE (Mean Squared Error) is computed by finding the difference of the corresponing values and squaring the result
# Find the sum of those numbers and find the mean. MSE is also called L2 Loss
L2loss_fn = nn.MSELoss()
L2loss = L2loss_fn(a, b) # ( (4-1)^2 + (7-2)^2 + (5-3)^2 )/3 = (3^2 + 5^2 + 2^2)/3 = (9+25+4)/3 = 38/3 = 12.6667

print(L1loss) # 3.33333
print(L2loss) # 12.6667
