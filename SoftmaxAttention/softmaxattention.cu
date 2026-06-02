#include <cuda_runtime.h>
__global__ void softmaxattention(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d){
                        int row = blockIdx.x*blockDim.x+threadIdx.x;
                        int col = blockIdx.y*blockDim.y+threadIdx.y;
                        int temp=0;
                        float localsum=0.0f;
                        __shared__ float arr[16][16];
                        arr[row*d+col]=K[row*d+col];
                        while(temp<d){
                            localsum=localsum+Q[row*temp+col]*arr[row*temp+col];
                        }
                      }

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d) {
                        int threadsperblock=256;
                        int blockspergrid=(16,16);
                        softmaxattention<<<blockspergrid,threadsperblock>>>(Q,K,V,output,M,N,d);
                        cudaDeviceSynchronize();
                      }
