'''
We are using BCE (Binary Cross Entropy Loss) Loss to see how far away the algorithm was from the correct answer (y).

https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss
'''

import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

def plot(y): 
  '''
  This function plots the BCE Loss between prediction and the actual answer (y), when y = 0 and 1.

  '''
  loss_list = []
  pred_list = []
  Y = torch.tensor([y], dtype = torch.float32)
  loss_fn = nn.BCELoss()

  # Every time the loop runs, it produces a number pred, and runs a loss function comparing pred to Y.
  # Then, it appends loss to loss_list and pred to pred_list. 
  # The point of this is to compare the loss of the algorithm when Y = 0, all the way up to when Y = 1. One plot is when Y = 0, the other plot is when Y = 1
  
  for pred in np.linspace (0,1,1001): # The 0,1,1001 means that it will start at 0, stop at 1, and produce 1001 numbers.
  # We put in 1001 numbers so on the graph, we can see how steep the increase is.
    loss = (loss_fn(torch.tensor([pred], dtype = torch.float32),Y))
    loss_list.append(loss)
    pred_list.append(pred)

  plt.scatter(pred_list, loss_list, s = 3, label = y)
  plt.legend()
  plt.xlabel('Prediction')
  plt.ylabel('BCE Loss between pred and y')
  plt.legend()



# We set y to 0 or 1 because y is a parameter of the function. Since y is a parameter it is inside the function and can be changed when calling the function.
plot(0) # y = 0
plot(1) # y = 1
