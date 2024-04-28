import torch
import torch.nn as nn

# Data
X_train = torch.tensor([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]], dtype = torch.float32)
y_train = torch.tensor([[2],[4],[6],[8],[10],[12],[14],[16],[18],[20]], dtype = torch.float32)

# Model
model = nn.Linear(1,1)

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

X_test = torch.tensor([[1.5],[14],[3.5],[2.9]])
model(X_test) 
