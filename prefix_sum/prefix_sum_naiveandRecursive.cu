#include <cuda_runtime.h>


__global__ void prefix_sum(
    const float* input,
    float* output,
    float* block_sums,
    int N
) {
    __shared__ float arr[256];
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    int lid=threadIdx.x;
    if (gid<N) {
        arr[lid]=input[gid];
    } else {
        arr[lid]=0.0f;
    }
    __syncthreads();
    float temp=arr[lid];
    for (int i=1;i<=128;i=i*2){
        if (lid>=i) {
            temp=temp+arr[lid-i];
        }
        __syncthreads();
        arr[lid] = temp;
        __syncthreads();
    }
    if (gid<N) {
        output[gid]=arr[lid];
    }
    if (lid==255&&block_sums!=nullptr) {
        block_sums[blockIdx.x]=arr[lid];
    }
}
// Phase 3: Add preceding blocks' sums
__global__ void add_offsets(
    float* output,
    const float* scanned_sums,
    int N
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    if (gid < N && bid > 0) {
        output[gid] += scanned_sums[bid - 1];
    }
}

// Phase 2: Recursive scan of block totals
void recursive_scan(
    const float* input,
    float* output,
    int N
) {
    if (N <= 0) return;
    int threadsperblock = 256;
    int blockspergrid = (N - 1) / threadsperblock + 1;
    // Base case
    if (blockspergrid == 1) {
        prefix_sum<<<1, 256>>>(input, output, nullptr, N);
        return;
    }
    // Allocate GPU memory
    float* block_sums = nullptr;
    float* scanned_sums = nullptr;
    cudaMalloc((void**)&block_sums,
               blockspergrid * sizeof(float));
    cudaMalloc((void**)&scanned_sums,
               blockspergrid * sizeof(float));
    // Scan each block independently
    prefix_sum<<<blockspergrid, threadsperblock>>>(
        input, output, block_sums, N
    );
    // Recursively scan the block totals
    recursive_scan(
        block_sums, scanned_sums, blockspergrid
    );
    // Add offsets to the original output
    add_offsets<<<blockspergrid, threadsperblock>>>(
        output, scanned_sums, N
    );
    // Release temporary GPU memory
    cudaFree(block_sums);
    cudaFree(scanned_sums);
}
extern "C" void solve(
    const float* input,
    float* output,
    int N
) {
    recursive_scan(input, output, N);
}