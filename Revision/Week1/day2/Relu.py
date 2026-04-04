#Relu Function:
#if any value in the tensor is lesser than 0 we make it 0 and >0 we leave them alone
# Relu function mainly used in neural networks for to make non linear relationships between weights/features



import torch


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    output[:]=torch.clamp(input,min=0)