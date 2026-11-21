#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// S = Q K^T * scale, one thread per score
__global__ void scores_kernel(const __nv_bfloat16* q, const __nv_bfloat16* k, __nv_bfloat16* s, int n, int d, float scale) {
    int h = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n) return;
    const __nv_bfloat16* qi = q + ((size_t)h * n + i) * d;
    const __nv_bfloat16* kj = k + ((size_t)h * n + j) * d;
    float acc = 0.0f;
    for (int x = 0; x < d; x++) acc += __bfloat162float(qi[x]) * __bfloat162float(kj[x]);
    s[((size_t)h * n + i) * n + j] = __float2bfloat16(acc * scale);
}

// P = softmax(S), one block per row
__global__ void softmax_kernel(const __nv_bfloat16* s, __nv_bfloat16* p, int n) {
    __shared__ float buf[256];
    const __nv_bfloat16* row = s + (size_t)blockIdx.x * n;
    __nv_bfloat16* prow = p + (size_t)blockIdx.x * n;
    int t = threadIdx.x;
    float m = -INFINITY;
    for (int j = t; j < n; j += 256) m = fmaxf(m, __bfloat162float(row[j]));
    buf[t] = m;
    __syncthreads();
    for (int st = 128; st > 0; st /= 2) {
        if (t < st) buf[t] = fmaxf(buf[t], buf[t + st]);
        __syncthreads();
    }
    m = buf[0];
    __syncthreads();
    float sum = 0.0f;
    for (int j = t; j < n; j += 256) sum += expf(__bfloat162float(row[j]) - m);
    buf[t] = sum;
    __syncthreads();
    for (int st = 128; st > 0; st /= 2) {
        if (t < st) buf[t] += buf[t + st];
        __syncthreads();
    }
    sum = buf[0];
    for (int j = t; j < n; j += 256) prow[j] = __float2bfloat16(expf(__bfloat162float(row[j]) - m) / sum);
}

// O = P V, one thread per output element
__global__ void pv_kernel(const __nv_bfloat16* p, const __nv_bfloat16* v, __nv_bfloat16* o, int n, int d) {
    int h = blockIdx.y;
    int i = blockIdx.x;
    int c = threadIdx.x;
    if (c >= d) return;
    const __nv_bfloat16* pi = p + ((size_t)h * n + i) * n;
    const __nv_bfloat16* vh = v + (size_t)h * n * d;
    float acc = 0.0f;
    for (int j = 0; j < n; j++) acc += __bfloat162float(pi[j]) * __bfloat162float(vh[(size_t)j * d + c]);
    o[((size_t)h * n + i) * d + c] = __float2bfloat16(acc);
}

cudaError_t naive_attention_launch(const void* q, const void* k, const void* v, void* s, void* p, void* o, int h, int n, int d, cudaStream_t stream) {
    float scale = 1.0f / sqrtf((float)d);
    dim3 b1(16, 16);
    dim3 g1((n + 15) / 16, (n + 15) / 16, h);
    scores_kernel<<<g1, b1, 0, stream>>>((const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (__nv_bfloat16*)s, n, d, scale);
    softmax_kernel<<<h * n, 256, 0, stream>>>((const __nv_bfloat16*)s, (__nv_bfloat16*)p, n);
    pv_kernel<<<dim3(n, h), d, 0, stream>>>((const __nv_bfloat16*)p, (const __nv_bfloat16*)v, (__nv_bfloat16*)o, n, d);
    return cudaGetLastError();
}