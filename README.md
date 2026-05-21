For every important kernel, the required flow is:

PyTorch reference -> CUDA implementation -> Triton implementation -> correctness test -> benchmark -> profiling note

### Standard repo layout

```text
pytorch_refs/       PyTorch reference implementations
cuda/               Clean CUDA kernel implementations
triton/             Triton implementations
benchmarks/         Benchmark scripts and CSV results
benchmarks/results/ Saved benchmark outputs
reports/            Written kernel analysis reports
profiling/          Nsight / PyTorch profiler notes
legacy/             Older practice code kept for reference
scripts/            Build/run helper scripts
# CUDA Kernels

CUDA kernels implemented from scratch while practicing GPU programming, parallel reductions, shared memory, warp-level primitives, and ML kernel patterns.

Every problem is written in CUDA C/C++ with a focus on understanding how GPU kernels actually work.

## Progress

| Day / Topic | Problems | Status | Key Concepts |
|------------|----------|--------|--------------|
| Day 1 | Vector Addition, Matrix Addition | Done | Thread/block indexing, 1D mapping, linear memory layout |
| Day 2 | ReLU, Leaky ReLU, Sigmoid | Done | Element-wise kernels, activation functions, boundary checks |
| Day 3 | Matrix Multiplication, Matrix Transpose | Done | 2D grid/block indexing, naive matmul, shared-memory tiled matmul |
| Day 4 | Matrix Copy, Reverse Array | Done | Simple memory access patterns, indexing practice |
| Day 5 | SiLU, SwiGLU, GeGLU | Done | ML activation kernels, gated activations |
| Day 6 | Reduction, Dot Product | Done | Shared-memory reduction, atomicAdd, multi-stage reduction |
| Day 7 | MSE Loss, Categorical Cross Entropy Loss | Done | Parallel loss computation, reductions, numerical stability |
| Day 8 | Softmax | Done | Stable softmax, max reduction, sum reduction, normalization |
| LeetGPU | Count Elements Equal to K | Done | Warp-level reduction, constraint-based optimization, leaderboard runtime |
| Prefix Sum | Prefix Sum / Scan | Host-side loop pending | Block-level scan, shared memory, multi-block scan structure |
| Top K Elements | Top K Selection using Bitonic Sort | Host-side loop pending | Bitonic sort, shared memory sorting, selecting k largest values |
| Attention | Softmax Attention | In Progress | QKᵀ computation, softmax, weighted value aggregation |
| Attention | Multi-Head Attention | In Progress | Multiple attention heads, head-wise parallelism, transformer kernel structure |
| WIP | RMS Normalization | In Progress | Normalization kernels, reduction patterns |

## Highlights

- Implemented CUDA kernels from scratch
- Used shared memory for tiled matrix multiplication and reductions
- Used warp-level primitives like `__shfl_down_sync`
- Benchmarked matrix multiplication using CUDA events
- Compared naive matmul vs shared-memory matmul
- Practiced ML-style kernels such as Softmax, SwiGLU, GeGLU, MSE, and Cross Entropy
- Implemented Top K selection using bitonic sort
- Used problem constraints to avoid unnecessary kernel launches in LeetGPU problems
- Currently working toward attention and transformer-style CUDA kernels

## Matrix Multiplication Benchmark

Naive matmul and shared-memory tiled matmul were benchmarked on square matrix sizes from `128` to `1024`.

| Size | Naive GFLOPS | Shared GFLOPS | Speedup |
|------|-------------:|--------------:|--------:|
| 128 | 375.78 | 425.83 | 1.13x |
| 256 | 1479.37 | 1785.57 | 1.20x |
| 512 | 2589.08 | 3785.47 | 1.46x |
| 1024 | 3086.77 | 4556.12 | 1.47x |

## Repo Structure

```text
Revision/
  Day1/   Vector and matrix addition
  day2/   ReLU, Leaky ReLU, Sigmoid
  Day3/   Matmul, transpose, benchmarking
  Day4/   Matrix copy, reverse array
  Day5/   SiLU, SwiGLU, GeGLU
  Day6/   Reductions, dot product

day7/     Loss kernels
day8/     Softmax

countarrayelementsequaltok/   LeetGPU optimized solution
prefix_sum/                  Prefix sum, host-side loop pending
topk/                        Top K selection using bitonic sort, host-side loop pending
attention/                   Softmax attention in progress
multihead_attention/          Multi-head attention in progress
RMS_Normalization/            RMS normalization in progress