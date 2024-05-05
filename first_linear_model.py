'''
The code below trains a model to predict y when given X by giving it training data and its results.

We create the training data and the testing data. We feed the training data to the y = mx + b model, and try to find the best m and b through this equation. 
We give the model its error so it can learn and become more accurate. For example, when the model receives 1 for an input, it should return 2 because they correspond in the training data.
Since the data is simple, we picked a simpler model. If the data was more complex, we would have picked a different model. 
Neural Networks work very well with many types of data which is why the're so common.
Convolutional Neural Networks work well with images.
Transformers (type of neural network) work very well with text data.
Graph Neural Networks work very well with social media data and medicine data.
'''
import torch
import torch.nn as nn

# Data
X_train = torch.tensor([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]], dtype = torch.float32)
y_train = torch.tensor([[2],[4],[6],[8],[10],[12],[14],[16],[18],[20]], dtype = torch.float32)
X_test = torch.tensor([[1.5],[14],[3.5],[2.9]])

# Model
model = nn.Linear(1,1) # Creates a nn.linear model with 1 input and 1 output. Its the equation y = mx + b

# Training
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
for epoch in range (200):
  for i in range(len(X_train)):
    pred = model(X_train)
    loss = loss_fn(pred, y_train)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Testing
model(X_test) 
