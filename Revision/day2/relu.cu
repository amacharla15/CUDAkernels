#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {

    int global_id=blockIdx.x*blockDim.x+threadIdx.x;

    if(global_id<N){
        output[global_id]=max(0.0f,input[global_id]);
    }
    
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
