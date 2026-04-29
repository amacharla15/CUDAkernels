#include <cuda_runtime.h>
__global__ void topkselection(const float* input, float* output, int N, int k){
    int gid = blockIdx.x*blockDim.x+threadIdx.x;
    __shared__ float arr[256];
    int lid=threadIdx.x;
    if(gid<N){
        arr[lid]=input[gid];
    }else{
        arr[lid]=-Float_MAX;
    }
    __syncthreads();

    for (int size = 2; size <= 256; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            int partner = lid ^ stride;

            if (partner > lid) {
                bool descending = ((lid & size) == 0);

                float a = arr[lid];
                float b = arr[partner];

                if ((descending && a < b) || (!descending && a > b)) {
                    arr[lid] = b;
                    arr[partner] = a;
                }
            }

            __syncthreads();
        }
    }

}


// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {

    int threadsperblock=256;
    int blockspergrid= (N+255)/256;

    topkselection<<<blockspergrid,threadsperblock>>>(input,output,N,K);
    cudaDeviceSynchronize();
}
