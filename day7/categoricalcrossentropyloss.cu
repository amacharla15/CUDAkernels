#include <cuda_runtime.h>
__global__ void CCEL(const float* logits, const int* true_labels, float* loss, float* arr, int N, int C){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    //finding maximum
    //exponentiating and storing sums for each thread in array 
    if(gid<N){
        float sum1=0.0f;
        int row = gid;
        int k=0;
        float maxi=logits[row*C+0];
        while(k<C){
            if(logits[row*C+k]>maxi){
                maxi=logits[row*C+k];
            }
            k=k+1;
        }
        k=0;
        while(k<C){
            sum1=sum1+__expf(logits[row*C + k] - maxi);
            k=k+1;
        }
        sum1=__logf(sum1);
        arr[row]=maxi+sum1-logits[row*C+true_labels[row]];
    }
}
__global__ void averageloss(float *arr, float * loss, int N){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    __shared__ float red[256];
    int lid =threadIdx.x;
    if(gid<N){
        red[lid]=arr[gid];
    }else{
        red[lid]=0.0f;
    }
    __syncthreads();
    int temp=128;
    while(temp>0){
        if(lid<temp){
            red[lid]=red[lid]+red[lid+temp];
        }
        temp=temp/2;
        __syncthreads();
    }
    if(lid==0){
        atomicAdd(loss,red[lid]);
    }
}
__global__ void finalloss( float * loss, int N){
    if (blockIdx.x == 0 && threadIdx.x == 0){
        *loss=*loss/N;
    }
    
}
// logits, true_labels, loss are device pointers
extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    int threadsperblock=256;
    int blockspergrid=(N+threadsperblock-1)/threadsperblock;

    float *arr;
    cudaMalloc((void**)&arr, N*sizeof(float));
    cudaMemset(loss, 0, sizeof(float));
    CCEL<<<blockspergrid,threadsperblock>>>(logits,true_labels,loss,arr,N,C);
    averageloss<<<blockspergrid,threadsperblock>>>(arr, loss, N);
    finalloss<<<1,1>>>(loss, N);
    cudaDeviceSynchronize();
    cudaFree(arr);
}
