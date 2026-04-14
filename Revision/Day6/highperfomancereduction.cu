#include <cuda_runtime.h>

__global__ void highperformancereduction(const float*input, float *output, int N){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    __shared__ float arr[256];
    int lid=threadIdx.x; 
    if(gid<N){
        if(lid<N){
            arr[lid]=input[lid];
        }else{
            arr[lid]=0;
        }
        
        __syncthreads();
        int temp=128;
        while(temp>0){
            int dup=temp+lid;
            if (lid<temp){
                arr[lid]=arr[lid]+arr[dup];
            }
            temp=temp/2;
            __syncthreads();
        }
        if (lid<0){
            output[0]=arr[0];
        }
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
