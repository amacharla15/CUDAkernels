#include <cuda_runtime.h>
__global__ void prefix_sum(const float* input , float* output , int N){
    __shared__ float arr[256];
    int gid = blockIdx.x*blockDim.x+threadIdx.x;
    int lid = threadIdx.x;
    if(gid<N){
        arr[lid]=input[gid];
    }else{
        arr[lid]=0.0f;
    }
    __syncthreads();
    float temp = arr[lid];
    for (int i = 1;i<=128;i=i*2){
        if (lid>=i){
            temp=temp+arr[lid-i];
        }
        __syncthreads();
            arr[lid]=temp;
        __syncthreads();
    }
    if(gid<N){
        output[gid]=arr[lid];
    }

}




extern "C" void solve(const float* input, float* output, int N) {
    int threadsperblock=256;
    int blockspergrid=(N+threadsperblock-1)/threadsperblock;
    prefix_sum<<<blockspergrid,threadsperblock>>>(input,output,N);
}
