import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns

# Data
readings = 8
half_year = int(366/2)
days_year = 366

temp = np.concatenate([np.random.rand(half_year*readings)*10+70,np.random.rand(half_year*readings)*10+90]) # Good & Bad Tempature
humidity = np.concatenate([np.random.rand(half_year*readings)*10+30,np.random.rand(half_year*readings)*10+60]) # Good & Bad Humidity

#weather = np.concatenate([np.ones(half_year*readings,dtype = np.int32),np.zeros(half_year*readings,dtype = np.int32)]) # Good & Bad Weather
idx = pd.date_range("2024-01-01T12:00AM", periods = readings*days_year, freq='3h')

df = pd.DataFrame(range(readings*days_year), index = idx, columns = ['Tempature'])
df['Humidity'] = humidity # Good Tempature
df['Tempature'] = temp # Bad Tempature

# We reshaped the data so each day had its own section.
N = 366
C = 2 # 2 channels which are tempature and humitidy
L = 8 # 8 readings a day
weather = df.values.reshape(N,L,C)
X = weather.transpose(0,2,1) # X has a shape of NxCxL or 366x2x8
X = torch.from_numpy(X.astype(np.float32))

y = np.concatenate([np.ones(half_year,dtype = np.int32),np.zeros(half_year,dtype = np.int32)]) # Combining Good & Bad Weather
y = torch.from_numpy(y.astype(np.float32))
X_train, X_test, y_train, y_test = train_test_split(X, y)


# Algo

model = nn.Sequential(nn.Conv1d(in_channels = 2, out_channels = 5, kernel_size = 3),
                      nn.MaxPool1d(kernel_size = 2),
                      nn.Conv1d(),
                      nn.Flatten(start_dim = 0),
                      nn.Linear(15,1),
                      nn.Sigmoid())

# Training

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.005)
for epoch in range(100):
  for i in range(len(X_train)):
    out = model(X_train[i])
    loss = loss_fn(out[0], y_train[i])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(epoch,loss, y_train[i], out)

