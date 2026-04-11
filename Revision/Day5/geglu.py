import torch


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    output[:] = 0.5 * input[N//2:N] * (1.0 + torch.erf(input[N//2:N] / 2.0**0.5)) * input[0:N//2]
