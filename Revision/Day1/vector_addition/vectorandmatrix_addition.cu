//Vector_Addition 
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int global_id=blockIdx.x*blockDim.x+threadIdx.x;
    if(global_id<N){
        C[global_id]=A[global_id]+B[global_id];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}


//Matrix addition 
// in gpu memory 2d is stored as 1dimension so this works
#include <cuda_runtime.h>
__global__ void matrix_add(const float* A, const float* B, float* C, int N) {
    int global_id=blockIdx.x*blockDim.x+threadIdx.x;
    if (global_id<N*N){
        C[global_id]=A[global_id]+B[global_id];
    }
    
}
// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
