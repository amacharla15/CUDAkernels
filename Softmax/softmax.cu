#include <cuda_runtime.h>

__global__ void maxi_kernel(const float* input, float* maxi, int N) {
    //finding max to handle overflows of exponentiation and using strides and 
    //local registers for optimization
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float arr[256];
    int lid = threadIdx.x;
    float localmax = -FLT_MAX;
    int stride = blockDim.x * gridDim.x;
    int temp = gid;
    while (temp < N) {
        if (localmax < input[temp]) {
            localmax = input[temp];
        }
        temp += stride;
    }
    arr[lid] = localmax;
    __syncthreads();
    int div = blockDim.x / 2;
    while (div >= 64) {
        if (lid < div) {
            if (arr[lid] < arr[lid + div]) {
                arr[lid] = arr[lid + div];
            }
        }
        __syncthreads();
        div = div / 2;
    }
    if (lid < 32) {
        float val = arr[lid];
        if (val < arr[lid + 32]) val = arr[lid + 32];

        float other = __shfl_down_sync(0xffffffff, val, 16);
        if (val < other) val = other;
        other = __shfl_down_sync(0xffffffff, val, 8);
        if (val < other) val = other;
        other = __shfl_down_sync(0xffffffff, val, 4);
        if (val < other) val = other;
        other = __shfl_down_sync(0xffffffff, val, 2);
        if (val < other) val = other;
        other = __shfl_down_sync(0xffffffff, val, 1);
        if (val < other) val = other;

        if (lid == 0) {
            maxi[blockIdx.x] = val;
        }
    }

}

__global__ void single_maxi(const float* maxi, float* single_maxi, int maxi_len) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float arr[256];
    float val = -FLT_MAX;
    if (gid < maxi_len) {
        val = maxi[gid];
    }
    if (gid + 256 < maxi_len and maxi[gid + 256] > val) {
        val = maxi[gid + 256];
    }
    arr[gid] = val;
    __syncthreads();
    int div = blockDim.x / 2;
    while (div >= 64) {
        if (gid < div) {
            if (arr[gid] < arr[gid + div]) {
                arr[gid] = arr[gid + div];
            }
        }
        __syncthreads();
        div = div / 2;
    }
    if (gid < 32) {
        val = arr[gid];
        if (val < arr[gid + 32]) val = arr[gid + 32];

        float other = __shfl_down_sync(0xffffffff, val, 16);
        if (other > val) {
            val = other;
        }
        other = __shfl_down_sync(0xffffffff, val, 8);
        if (other > val) {
            val = other;
        }
        other = __shfl_down_sync(0xffffffff, val, 4);
        if (other > val) {
            val = other;
        }
        other = __shfl_down_sync(0xffffffff, val, 2);
        if (other > val) {
            val = other;
        }
        other = __shfl_down_sync(0xffffffff, val, 1);
        if (other > val) {
            val = other;
        }
        if (gid == 0) {
            *single_maxi = val;
        }
    }
}

__global__ void fused_softmax(const float *input, float*output, float *sum1, float* single_maxi_ptr, int N ){
    int gid = blockIdx.x*blockDim.x+threadIdx.x;
    __shared__ float arr[256];
    int lid=threadIdx.x;
    int temp=0;
    int stride=blockDim.x*gridDim.x;
    float local_max=0.0f;
    while (gid+stride*temp < N) {
        int idx = gid + temp * stride;
        float val = __expf(input[idx] - *single_maxi_ptr);
        output[idx] = val;
        local_max += val;
        temp += 1;
    }
    arr[lid] = local_max;
    //shared_memory sums 
    __syncthreads();
    int div = blockDim.x / 2;
    while (div >= 32) {
        if (lid < div) {
            arr[lid]+=arr[lid+div];
        }
        __syncthreads();
        div = div / 2;
    }
    if(lid<32){
        float val = arr[lid];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (lid == 0) {
            sum1[blockIdx.x] = val;
        }
    }
}

__global__ void single_sum(const float * sum1, float *single_sum1, int N){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    int lid=threadIdx.x;
    float local_sum=0.0f;
    __shared__ float arr[256];
    if (gid<N) {
        local_sum = sum1[gid];
    }
    if (gid+256< N) {
        local_sum += sum1[gid+256];
    }
    arr[lid] = local_sum;
    __syncthreads();
    int div=blockDim.x/2;
    while (div >= 64) {
        if (lid<div) {
            arr[lid] += arr[lid+div];
        }
        __syncthreads();
        div = div/2;
    }
    if (lid < 32) {
        float val = arr[lid];
        val += arr[lid + 32];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (gid == 0) {
            *single_sum1 = val;
        }
    }
}

__global__ void normalize_kernel(float *output, float *single_sum1, int N) {
    int gid = blockIdx.x*blockDim.x+threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    int temp =gid;
    while (temp<N) {
        output[temp] =output[temp]/(*single_sum1);
        temp+=stride;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int fullblocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int strideblockspergrid = fullblocksPerGrid / 4;
    int launchedBlocks;

    if (strideblockspergrid >= 1) {
        launchedBlocks = strideblockspergrid;
    } else {
        launchedBlocks = fullblocksPerGrid;
    }

    //finding maxi array
    float* maxi;
    cudaMalloc(&maxi, launchedBlocks * sizeof(float));
    maxi_kernel<<<launchedBlocks, threadsPerBlock>>>(input, maxi, N);

    //finding single maxi with the reduction
    // in general we need to do a strided version with launching multiple kernels if N is very large to find single maximum
    //but the problem statement gave 50000 as N , input size and since we are launching ' 4elements per stride" the size of maxi array 
    // is just 49 so we can simple launch two warps and compute single maximum

    float* single_maxi_ptr;
    cudaMalloc(&single_maxi_ptr, sizeof(float));
    single_maxi<<<1, 256>>>(maxi, single_maxi_ptr, launchedBlocks);

    float * sum1;
    cudaMalloc(&sum1,launchedBlocks *sizeof(float));
    fused_softmax<<<launchedBlocks, threadsPerBlock>>>(input, output, sum1, single_maxi_ptr, N);


    float * single_sum1;
    cudaMalloc(&single_sum1, sizeof(float));
    single_sum<<<1, 256>>>(sum1, single_sum1, launchedBlocks);

    normalize_kernel<<<launchedBlocks, threadsPerBlock>>>(output, single_sum1, N);

    cudaDeviceSynchronize();

    cudaFree(maxi);
    cudaFree(single_maxi_ptr);
    cudaFree(sum1);
    cudaFree(single_sum1);
}