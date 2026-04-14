#include <cuda_runtime.h>

__global__ void highperformancereduction(const float*input, float *output, int N){
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
        atomicAdd(output,arr[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsperblock=256;
    int blockspergrid=(N+threadsperblock-1)/threadsperblock;
    cudaMemset(output, 0, sizeof(float));
    highperformancereduction<<<blockspergrid,threadsperblock>>>(input,output,N);
    cudaDeviceSynchronize();
}
