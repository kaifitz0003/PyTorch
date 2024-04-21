'''
Using nn.Linear to compute y = 2x. 

The below code in the doc string is without nn.Linear. This is a manual way of doing it. 
x = torch.tensor([2], dtype = torch.float32)
y = 2*x
y
'''
x = torch.tensor([5], dtype = torch.float32)
model = nn.Linear(1,1) 

model.weight.data = torch.tensor([[2.]]) # m in y = mx+b
model.bias.data = torch.tensor([0.]) # b in y = mx+b
y = model(x)
y
