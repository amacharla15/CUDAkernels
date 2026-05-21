# Benchmark Result Schema

Every important kernel must report results using the same format.

## Required fields

| Field | Meaning |
|---|---|
| kernel_name | Name of kernel being tested |
| shape | Input/output tensor shape |
| dtype | Data type: fp32, fp16, int8, etc. |
| implementation | pytorch_eager, torch_compile, cuda_naive, cuda_optimized, triton |
| time_ms | Average runtime in milliseconds |
| bandwidth_gb_s | Estimated memory bandwidth when applicable |
| gflops | Estimated compute throughput when applicable |
| speedup_vs_pytorch | PyTorch time / implementation time |
| correctness | pass/fail |
| max_abs_error | Maximum absolute error vs PyTorch |
| notes | Memory behavior, bottleneck, or limitation |

## Required benchmark flow

1. Create input tensor.
2. Run PyTorch reference.
3. Run CUDA version.
4. Run Triton version if available.
5. Compare correctness against PyTorch.
6. Time with CUDA events.
7. Save result to CSV.
8. Write a short profiling/memory note.

## Standard CSV columns

kernel_name,shape,dtype,implementation,time_ms,bandwidth_gb_s,gflops,speedup_vs_pytorch,correctness,max_abs_error,notes