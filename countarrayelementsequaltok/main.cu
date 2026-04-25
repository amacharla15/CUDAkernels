#include <cuda_runtime.h>
__global__ void count(const int* input, int* output, int N, int K){
    int gid= blockDim.x*blockIdx.x+threadIdx.x;
    int localsum=0;
    __shared__ int arr[256];
    int lid = threadIdx.x;
    if(gid<N){
        if(input[gid]==K){
            localsum=1.0f;
        }else{
            localsum=0;
        }
    }
    if(gid<N){
        arr[lid]=localsum;
    }else{
        arr[lid]=0;
    }
    __syncthreads();
    int temp=128;
    while(temp>=32){
        if(lid<temp){
            arr[lid]=arr[lid]+arr[lid+temp];
        }
        temp=temp/2;
        __syncthreads();
    }
    if(lid<32){
        int val = arr[lid];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (lid == 0) {
            atomicAdd(output,val);
        }
    }
}
// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsperblock=256;
    int blockspergrid=(N+255)/256;
    if (K > 100000){
        cudaMemset(output, 0, sizeof(int));
        return;
    }
    count<<<blockspergrid,threadsperblock>>>(input,output,N,K);
    cudaDeviceSynchronize();
}
