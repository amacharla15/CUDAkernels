GPU Problems in PyTorch, CUDA & Triton

Every problem solved from scratch on an NVIDIA A100.

## Progress

| Day | Problems | LeetGPU | PyTorch | CUDA | Triton | Key Concepts |
|-----|----------|---------|---------|------|--------|--------------|
| 1 | Vector Addition, Matrix Addition | ✅ | ✅ | ✅ | — | Thread-block mapping, 2D grid/block indexing, broadcasting |
| 2 | ReLU, Leaky ReLU, Sigmoid | ✅ | ✅ | ✅ | — | Element-wise kernels, custom autograd (fwd + bwd), numerically stable sigmoid, MNIST MLP from scratch (raw training loop) |
| 3 | Matrix Multiplication, Matrix Transpose | ✅ | ✅ | ✅ | — | Naive vs tiled shared-memory matmul, custom nn.Linear from scratch, replaced MLP layers with custom implementation, Nsight Systems profiling on A100 |