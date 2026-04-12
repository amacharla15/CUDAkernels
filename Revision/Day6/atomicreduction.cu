#include <cuda_runtime.h>

__global__ void atomicreduction(const float*input, float *output, int N){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    if(gid<N){
        atomicAdd(output, input[gid]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsperblock=256;
    int blockspergrid=(N+threadsperblock-1)/threadsperblock;
    cudaMemset(output, 0, sizeof(float)); //key things to remember, to set output value as zero because we may sum garbage values
    atomicreduction<<<blockspergrid,threadsperblock>>>(input,output,N);
    cudaDeviceSynchronize();
}
