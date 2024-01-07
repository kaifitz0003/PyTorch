import torch
# N = batch size of input (2)
# L = length of input (4)
# Cin = in_channels (3)
# Cout = out_channels (6)
# K = kernel_size (2)
# Lout = length of output signal (L-K+1)
# Input dimensions are Cin x L
# Weight dimensions are Cout x Cin x K and K <= L
# Output dimensions is 2d and is Cout x Lout

# INPUT
x = torch.tensor([[[ 1, 2, 3, 4],[ 5, 6, 7, 8],[ 9,10,11,12]],
                  [[13,14,15,16],[17,18,19,20],[21,22,23,24]],
                  [[25,26,27,28],[29,30,31,32],[33,34,35,36]],
                  [[37,38,39,40],[41,42,43,44],[45,46,47,48]],
                  [[49,50,51,52],[53,54,55,56],[57,58,59,60]]], dtype = torch.float32)
N, Cin, L = x.shape

# PROCESS
Cout = 6
K = 2
# Cout changes the number of 2d arrays, Cin changes the number of rows,  and kernel changes the number of columns
algo = torch.nn.Conv1d(in_channels=Cin, out_channels = Cout, kernel_size = K) # Picks the weights and bias at random
algo.weight.data = torch.tensor([[[1,2],[3,4],[5,6]], # Optional. Set your own weights
                                 [[7,8],[9,10],[11,12]],
                                 [[13,14],[15,16],[17,18]],
                                 [[19,20],[21,22],[23,24]],
                                 [[25,26],[27,28],[29,30]],
                                 [[31,32],[33,34],[35,36]]], dtype = torch.float32)
algo.bias.data = torch.tensor([1,2,3,4,5,6], dtype = torch.float32) # Bias is how much you add the the convolution Optional. Set your own bias

# OUTPUT
y = algo(x)
y
