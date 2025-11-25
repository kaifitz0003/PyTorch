import torch
import torch.nn
import torch.nn.functional as F
L=3
Hin=2
d_model=6
X = torch.rand(L,Hin)
WQ = torch.rand(d_model,Hin)
Q = X@WQ.T
WK = torch.rand(d_model,Hin)
K = X@WK.T
WV = torch.rand(d_model,Hin)
V = X@WV.T

sdpa_reference=F.scaled_dot_product_attention(Q,K,V)
scores = (Q@K.T)/(d_model**0.5)
weights = scores.softmax(axis=1)
sdpa_manual = weights@V
torch.allclose(sdpa_reference,sdpa_manual)
