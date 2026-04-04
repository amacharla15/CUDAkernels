#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* X, float* Y, int N) {
    int global_id=blockIdx.x*blockDim.x+threadIdx.x;
    if (global_id<N){
        Y[global_id]=1.0f/(1.0f+exp(-X[global_id]));
    }
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}
