#include <cuda_runtime.h>

__global__ void highperformancereduction(const float*input, float *partial_sums, int N){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    int lid=threadIdx.x; 
    __shared__ float arr[256];
    if(gid<N){
        arr[lid]=input[gid];
    }else{
        arr[lid]=0;
    }
    __syncthreads();
    int temp=128;
    while(temp>0){
        if(lid<temp){
            arr[lid]=arr[lid]+arr[lid+temp];
        }
        temp=temp/2;
        __syncthreads();
    }
    if(lid==0){
        partial_sums[blockIdx.x]=arr[0];
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsperblock=256;
    int blockspergrid=(N+threadsperblock-1)/threadsperblock;
    float* partial_sums;
    cudaMalloc(&partial_sums, blockspergrid * sizeof(float));
    cudaMemset(output, 0, sizeof(float));
    highperformancereduction<<<blockspergrid,threadsperblock>>>(input,partial_sums,N);
    int secondN = blockspergrid;
    int secondBlocks = (secondN + threadsperblock - 1) / threadsperblock;

    highperformancereduction<<<secondBlocks, threadsperblock>>>(partial_sums, output, secondN);
    cudaDeviceSynchronize();
    cudaFree(partial_sums);
}
