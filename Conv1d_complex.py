import torch
# N = batch size of input (2)
# L = length of input (4)
# Cin = in_channels (3)
# Cout = out_channels (6)
# K = kernel_size (2)
# Lout = length of output signal (L-K+1)
# The Input is a 3d tensor and its dimensions are N x Cin x L
# The Weight is a 3d tensor and its dimensions are Cout x Cin x K, where K <= L
# The Bias is a 1d tensor and its dimensions are Cout
# The Output is a 3d tensor and its dimensions are N x Cout x Lout, where Lout is (L - K + 1)


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
'''
Answer: 
tensor([[[  150.,   171.,   192.],
         [  349.,   406.,   463.],
         [  548.,   641.,   734.],
         [  747.,   876.,  1005.],
         [  946.,  1111.,  1276.],
         [ 1145.,  1346.,  1547.]],

        [[  402.,   423.,   444.],
         [ 1033.,  1090.,  1147.],
         [ 1664.,  1757.,  1850.],
         [ 2295.,  2424.,  2553.],
         [ 2926.,  3091.,  3256.],
         [ 3557.,  3758.,  3959.]],

        [[  654.,   675.,   696.],
         [ 1717.,  1774.,  1831.],
         [ 2780.,  2873.,  2966.],
         [ 3843.,  3972.,  4101.],
         [ 4906.,  5071.,  5236.],
         [ 5969.,  6170.,  6371.]],

        [[  906.,   927.,   948.],
         [ 2401.,  2458.,  2515.],
         [ 3896.,  3989.,  4082.],
         [ 5391.,  5520.,  5649.],
         [ 6886.,  7051.,  7216.],
         [ 8381.,  8582.,  8783.]],

        [[ 1158.,  1179.,  1200.],
         [ 3085.,  3142.,  3199.],
         [ 5012.,  5105.,  5198.],
         [ 6939.,  7068.,  7197.],
         [ 8866.,  9031.,  9196.],
         [10793., 10994., 11195.]]], grad_fn=<ConvolutionBackward0>)
         '''
