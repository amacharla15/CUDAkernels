
#vector Addition
import torch
# A, B, C are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    torch.add(A,B, out=C)

#Matrix Addition
import torch
# A, B, C are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    torch.add(A,B,out=C)


Key things:
1. In-place vs rebinding
C = A + B creates a brand new tensor in a brand new memory location. The variable C now points there.  
C[:] = A + B and torch.add(A, B, out=C) write directly into the memory that C already occupies. No new allocation. 

2. Element-wise operations and dimensionality:
PyTorch doesn't care about dimensionality for element-wise operations. + applies to every corresponding element regardless of shape.
A 1D tensor, 2D tensor, 5D tensor — same operator.

3.Device mismatch
Every tensor lives on a specific device — cpu or cuda:0 (first GPU), cuda:1 (second GPU), etc. 
If A is on CPU and B is on GPU, A + B throws a RuntimeError. PyTorch does NOT automatically move tensors between devices because that transfer is 
expensive and it wants us to be intentional about it. We move tensors with .to('cuda'), .cuda(), or .to(device). 

4.Broadcasting rules
Rules — dimensions are compared right to left:
If both dimensions are equal → fine
If one of them is 1 → it gets stretched to match the other
If they're different and neither is 1 → error