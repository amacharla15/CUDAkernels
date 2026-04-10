//the kernel is launched as 1D here so single gid is enough 
//if the kernel is launched as 2D we can use formula [rows*number of columns+cols]
//if the kernel is launched as 1D we can use gid and compute row col , gid /k , gid %k for row and col 

#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    if(gid<N*N){
        B[gid]=A[gid];
    }


}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    cudaDeviceSynchronize();
}
