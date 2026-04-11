import torch


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    output[:]=input[0:N//2]*(1.0/(1.0+torch.exp(-input[0:N//2])))*input[N//2:N]
