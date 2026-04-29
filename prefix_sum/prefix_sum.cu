#include <cuda_runtime.h>
__global__ void prefixsum(const float * input, float * output, float * block_sums, int N){
    __shared__ float arr[256];
    int gid = blockIdx.x*blockDim.x+threadIdx.x;
    int lid = threadIdx.x;
    if(gid<N){
        arr[lid]=input[gid];
    }else{
        arr[lid]=0;
    }
    __syncthreads();
    int temp=0;
    while(1 << temp<256){
        float localsum=0.0f;
        if(lid>=1 << temp){
            localsum=arr[lid-(1 << temp)];
        }
        __syncthreads();
        if(lid>=1 << temp){
            arr[lid]+=localsum;
        }
        __syncthreads();
        temp=temp+1;
        if()
    }
    if(gid<N){
        output[gid]=arr[lid];
    }
    if(gid<N){

    }
}
// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsperblock = 256;
    int blockspergrid = (N+threadsperblock-1)/threadsperblock;
    void scan_helper(const float *input,float *output, int N){
        int threadsperblock = 256;
        int blockspergrid = (N+threadsperblock-1)/threadsperblock;
        float * block_sums;
        cudaMalloc(block_sums, blockspergrid* sizeof(float));
        prefixsum<<<blockspergrid,threadsperblock>>>(input,output,block_sums,N);
        if(blockspergrid>1){
            
        }
    }
    cudaDeviceSynchronize();
}
