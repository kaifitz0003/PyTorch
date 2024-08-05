import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns

df = pd.read_csv('mushroom_cleaned.csv')
X = df.drop('class', axis = 1).values
y = df['class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
model = nn.Sequential(nn.Linear(8,1),nn.Sigmoid(),nn.Linear(1,1),nn.Sigmoid())
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
for epoch in range(20):
  for i in range(len(X_train)):
    out = model(X_train[i])
    loss = loss_fn(out[0], y_train[i])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
  print(epoch)
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
y_pred = model(X_test)
y_pred
accuracy_score(y_test, y_pred.detach().numpy().round())
