#include <cuda_runtime.h>

__global__ void dot_product(const float* A, const float* B, float* result, int N){
    int gid= blockIdx.x*blockDim.x+threadIdx.x;
    __shared__ float arr[256];
    int lid=threadIdx.x;
    if(gid<N){
        arr[lid]=A[gid]*B[gid];
    }else{
        arr[lid]=0.0f;
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
        atomicAdd(result, arr[0]);
    }
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threadsperblock=256;
    int blockspergrid=(N+threadsperblock-1)/threadsperblock;
    cudaMemset(result, 0, sizeof(float));
    dot_product<<<blockspergrid,threadsperblock>>>(A,B,result,N);
    cudaDeviceSynchronize();
}
