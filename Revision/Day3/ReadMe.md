# CUDA Matrix Multiplication: Naive vs Shared Memory

This project compares two CUDA matrix multiplication kernels:


For each matrix size:

- generates random input matrices
- computes a CPU reference result
- runs both CUDA kernels
- verifies correctness against the CPU result
- measures kernel-only execution time using CUDA events
- averages timings across multiple runs
- computes GFLOPS

## Tested sizes

The benchmark currently tests square matrices of size:

- `128 x 128 x 128`
- `256 x 256 x 256`
- `512 x 512 x 512`
- `1024 x 1024 x 1024`

Each kernel is warmed up once and then timed over 20 runs.

## Kernels

### 1. Naive kernel
Each thread computes one output element directly from global memory.

### 2. Shared-memory kernel
Each block loads tiles of `A` and `B` into shared memory and reuses them during the dot-product computation.

## Build

```bash
nvcc -arch=sm_80 matmul_perf.cu -o matmul

Run
./matmul
Benchmark Results
Size	Naive Avg Time (ms)	Shared Avg Time (ms)	Naive GFLOPS	Shared GFLOPS	Speedup
128	0.0111616	0.0098496	375.78	425.835	1.1332x
256	0.0226816	0.0187920	1479.37	1785.57	1.2070x
512	0.1036800	0.0709120	2589.08	3785.47	1.4621x
1024	0.6957060	0.4713410	3086.77	4556.12	1.4760x
Profiling Summary

Nsight Systems kernel summary:

naive_matmul: 59.7% of total GPU kernel time
shared_matmul: 40.3% of total GPU kernel time

This confirms the benchmark result that the shared-memory kernel is faster than the naive kernel on the tested workloads.

Notes
Timing is kernel-only timing using CUDA events.
Host-to-device and device-to-host copy time is not included in the reported kernel timings.
Correctness is checked against a CPU reference implementation.
Nsight Compute (ncu) profiling could not be completed on the shared machine because GPU performance counters were restricted.