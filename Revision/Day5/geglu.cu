#include <cuda_runtime.h>

__global__ void geglu_kernel(const float* input, float* output, int halfN) {
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    if(gid<halfN){
        float x1 = input[gid];
        float x2 = input[gid + halfN];
        output[gid] = x1 * (0.5f * x2 * (1.0f + erff(x2 / sqrtf(2.0f))));
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
