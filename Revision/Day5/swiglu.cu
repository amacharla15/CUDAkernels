#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    if (gid<halfN){
        output[gid]=input[gid]*(1.0f/(1.0f+expf(-input[gid])))*input[gid+halfN];
    }
    
    
    
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
