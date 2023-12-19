'''
The code below performs 1d Convolution.

A kernel is a short signal that either you pick or the ML learns it.
Then you convolve the kernel with the input to get the output.
Convolution is a sliding dot product, where you slide the kernel over the input.
Then, you multiply each point of the kernel with the corresponding point and you add the result. This is the dot product.
'''

## NUMPY

import numpy as np

X = np.array([1, 3, 0, 2, 10, 1, 2,0]) # Input signal
w = np.array([1, 2, 3]) # Weights of the Kernel
y_length = len(X)-len(w)+1 # Finds how many times you need to slide the kernel over the input signal
y1 = np.zeros(y_length) # Place Holder for Output Signal


for i in range (y_length): # len(X)-len(w)+1 represents how many times the the kernel slides over the input signal
  y1[i] = np.dot(X[i:i+len(w)], w) # Manually slides the kernel by taking the parts of the data with len(w)

print(y1)

## PYTORCH

import torch
x_torch = torch.tensor([[1.,3,0,2,10,1,2,0]])

algo = torch.nn.Conv1d(in_channels=1, out_channels = 1, kernel_size = 3)
algo.weight.data = torch.tensor([[[1.0,2,3]]])
algo.bias.data = torch.tensor([0.]) # Bias is how much you add the the convolution.
y2 = algo(x_torch)

#print(y2)

## COMPARE
n = y2.detach().numpy()
n = n[0] # Gets rid of extra brackets
print(n)
np.array_equal(y1,n)
