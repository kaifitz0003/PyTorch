import torch

# Input Dimensions (N x Cin x H x W)
# N = batch size of input (1)
# Cin = in_channels (3)
# W = width of input image(4)
# H = height of input image(4)

# Weight Dimensions (Cout x Cin x K x K)
# Cout = out_channels (1)
# K = kernel_size (2)
# K is both the height and the width of the kernel

# Output Dimensions
# Lout = length of output signal (L-K+1)
# Output dimensions is 2d and is N x Cout x Lout



# INPUT
x = torch.tensor([[[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]],
                [[17,18,19,20],[21,22,23,24],[25,26,27,28],[29,30,31,32]],
                [[33,34,35,36],[37,38,39,40],[41,42,43,44],[45,46,47,48]]]], dtype = torch.float32)

N, Cin, H, W = x.shape

# PROCESS
Cout = 1
K = (2,2)
algo = torch.nn.Conv2d(in_channels = Cin, out_channels = Cout, kernel_size = K)
algo.weight.data = torch.tensor([[[[1,2],[3,4]],
                                 [[5,6],[7,8]],
                             [[9,10],[11,12]]]], dtype = torch.float32)
algo.bias.data = torch.tensor([1], dtype=torch.float32)
# Cout changes the number of 2d arrays, Cin changes the number of rows,  and kernel changes the number of columns.

out = algo(x)
out

