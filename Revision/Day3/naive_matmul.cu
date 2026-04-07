#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
                                                int col=blockIdx.x*blockDim.x+threadIdx.x;
                                                int row=blockIdx.y*blockDim.y+threadIdx.y;
                                                int temp=0;
                                                float sum1=0.0f;
                                                if(row<M and col<K){
                                                    while(temp<N){
                                                        sum1=sum1+A[row*N+temp]*B[temp*K+col];
                                                        temp=temp+1;
                                                    }
                                                    C[row*K+col]=sum1;
                                                }
                                             }
// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
