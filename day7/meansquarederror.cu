#include <cuda_runtime.h>

__global__ void meansquarederror(const float* predictions, const float* targets, float* mse, int N){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    __shared__ float arr[256];
    int lid=threadIdx.x;
    if(gid<N){
        arr[lid]=(predictions[gid]-targets[gid])*(predictions[gid]-targets[gid]);
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
        atomicAdd(mse,arr[lid]/(float)N);
    }
}
// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {
    int threadsperblock=256;
    int blockspergrid=(N+threadsperblock-1)/threadsperblock;
    cudaMemset(mse, 0, sizeof(float));
    meansquarederror<<<blockspergrid,threadsperblock>>>(predictions,targets,mse,N);
    cudaDeviceSynchronize();
}
