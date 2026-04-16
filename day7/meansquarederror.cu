#include <cuda_runtime.h>

__global__ void meansquarederror(const float* predictions, const float* targets, float* mse, int N){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    __shared__ float arr[256];
    int stride= blockDim.x * gridDim.x;
    int lid=threadIdx.x;
    if(gid<N){
        float local_sum=0.0f;
        for (int idx = gid; idx < N; idx += stride){
            local_sum= local_sum+(predictions[idx]-targets[idx])*(predictions[idx]-targets[idx]);
        }
        arr[lid]=local_sum;
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
    int fullblockspergrid=(N+threadsperblock-1)/threadsperblock;
    int strideblockspergrid=fullblockspergrid/4;
    cudaMemset(mse, 0, sizeof(float));
    if(strideblockspergrid>=1){
        meansquarederror<<<strideblockspergrid,threadsperblock>>>(predictions,targets,mse,N);
    }
    else{
        meansquarederror<<<fullblockspergrid,threadsperblock>>>(predictions,targets,mse,N);
    }
    
    cudaDeviceSynchronize();
}
