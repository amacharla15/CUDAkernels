import torch

def solve(A: torch.Tensor, B: torch.Tensor, result: torch.Tensor, N: int):
    result[0] = torch.sum(A[:N] * B[:N])