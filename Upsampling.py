"""
In the below code, we expirement with Upsample. 

https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html
Upsampling doesnt have any machine learning, it just carries out some predefined methods. 
In the example below, the answer is:
tensor([[[1.0000, 1.2500, 1.7500, 2.2500, 2.7500, 3.2500, 3.7500, 4.2500,
          4.7500, 5.0000]]])
"""
import torch.nn as nn
import torch
X = torch.tensor([[[1,2,3,4,5]]], dtype = torch.float32) # Upsampling requires the data to be 3d, 4d, or 5d. If you have 1d date, you still have to make it 3d. 

model = nn.Upsample(scale_factor=2, mode='linear')
model(X)
