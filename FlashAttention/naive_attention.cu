// nvcc -O3 -arch=sm_90 naive_attention.cu -o naive_attention && ./naive_attention 4096
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("cuda error: %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

// S = Q K^T * scale, one thread per score
__global__ void scores_kernel(const __nv_bfloat16* q, const __nv_bfloat16* k, float* s, int n, int d, float scale) {
    int h = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n) return;
    const __nv_bfloat16* qi = q + ((size_t)h * n + i) * d;
    const __nv_bfloat16* kj = k + ((size_t)h * n + j) * d;
    float acc = 0.0f;
    for (int x = 0; x < d; x++) acc += __bfloat162float(qi[x]) * __bfloat162float(kj[x]);
    s[((size_t)h * n + i) * n + j] = acc * scale;
}

// softmax in place, one block per row
__global__ void softmax_kernel(float* s, int n) {
    extern __shared__ float buf[];
    float* row = s + (size_t)blockIdx.x * n;
    int t = threadIdx.x;
    float m = -INFINITY;
    for (int j = t; j < n; j += blockDim.x) m = fmaxf(m, row[j]);
    buf[t] = m;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st /= 2) {
        if (t < st) buf[t] = fmaxf(buf[t], buf[t + st]);
        __syncthreads();
    }
    m = buf[0];
    __syncthreads();
    float sum = 0.0f;
    for (int j = t; j < n; j += blockDim.x) {
        float e = __expf(row[j] - m);
        row[j] = e;
        sum += e;
    }
    buf[t] = sum;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st /= 2) {
        if (t < st) buf[t] += buf[t + st];
        __syncthreads();
    }
    sum = buf[0];
    for (int j = t; j < n; j += blockDim.x) row[j] /= sum;
}

// O = P V, one thread per output element
__global__ void pv_kernel(const float* p, const __nv_bfloat16* v, __nv_bfloat16* o, int n, int d) {
    int h = blockIdx.y;
    int i = blockIdx.x;
    int c = threadIdx.x;
    const float* pi = p + ((size_t)h * n + i) * n;
    const __nv_bfloat16* vh = v + (size_t)h * n * d;
    float acc = 0.0f;
    for (int j = 0; j < n; j++) acc += pi[j] * __bfloat162float(vh[(size_t)j * d + c]);
    o[((size_t)h * n + i) * d + c] = __float2bfloat16(acc);
}

void naive_attention(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, float* s, __nv_bfloat16* o, int h, int n, int d) {
    float scale = 1.0f / sqrtf((float)d);
    dim3 b1(16, 16);
    dim3 g1((n + 15) / 16, (n + 15) / 16, h);
    scores_kernel<<<g1, b1>>>(q, k, s, n, d, scale);
    softmax_kernel<<<h * n, 256, 256 * sizeof(float)>>>(s, n);
    pv_kernel<<<dim3(n, h), d>>>(s, v, o, n, d);
}

// reference for one row (head h, query i) on the cpu
void cpu_row(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, int n, int d, int h, int i, float* out) {
    std::vector<float> s(n);
    float scale = 1.0f / sqrtf((float)d);
    float m = -INFINITY;
    for (int j = 0; j < n; j++) {
        float acc = 0.0f;
        for (int x = 0; x < d; x++) acc += q[((size_t)h * n + i) * d + x] * k[((size_t)h * n + j) * d + x];
        s[j] = acc * scale;
        m = fmaxf(m, s[j]);
    }
    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
        s[j] = expf(s[j] - m);
        sum += s[j];
    }
    for (int c = 0; c < d; c++) {
        float acc = 0.0f;
        for (int j = 0; j < n; j++) acc += s[j] * v[((size_t)h * n + j) * d + c];
        out[c] = acc / sum;
    }
}

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 4096;
    int h = 32, d = 128;
    size_t count = (size_t)h * n * d;

    std::vector<float> qf(count), kf(count), vf(count);
    std::vector<__nv_bfloat16> qb(count), kb(count), vb(count), ob(count);
    srand(0);
    for (size_t x = 0; x < count; x++) {
        qb[x] = __float2bfloat16(4.0f * rand() / RAND_MAX - 2.0f);
        kb[x] = __float2bfloat16(4.0f * rand() / RAND_MAX - 2.0f);
        vb[x] = __float2bfloat16(4.0f * rand() / RAND_MAX - 2.0f);
        qf[x] = __bfloat162float(qb[x]);
        kf[x] = __bfloat162float(kb[x]);
        vf[x] = __bfloat162float(vb[x]);
    }

    __nv_bfloat16 *q, *k, *v, *o;
    float* s;
    size_t bytes = count * sizeof(__nv_bfloat16);
    CHECK(cudaMalloc(&q, bytes));
    CHECK(cudaMalloc(&k, bytes));
    CHECK(cudaMalloc(&v, bytes));
    CHECK(cudaMalloc(&o, bytes));
    CHECK(cudaMalloc(&s, (size_t)h * n * n * sizeof(float)));
    CHECK(cudaMemcpy(q, qb.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(k, kb.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(v, vb.data(), bytes, cudaMemcpyHostToDevice));

    naive_attention(q, k, v, s, o, h, n, d);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    int runs = 10;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int r = 0; r < runs; r++) naive_attention(q, k, v, s, o, h, n, d);
    cudaEventRecord(stop);
    CHECK(cudaEventSynchronize(stop));
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= runs;
    double flops = 4.0 * h * (double)n * n * d;
    double tflops = flops / (ms * 1e-3) / 1e12;

    CHECK(cudaMemcpy(ob.data(), o, bytes, cudaMemcpyDeviceToHost));
    float max_err = 0.0f;
    std::vector<float> ref(d);
    for (int t = 0; t < 64; t++) {
        int hh = rand() % h, i = rand() % n;
        cpu_row(qf, kf, vf, n, d, hh, i, ref.data());
        for (int c = 0; c < d; c++) {
            float got = __bfloat162float(ob[((size_t)hh * n + i) * d + c]);
            max_err = fmaxf(max_err, fabsf(got - ref[c]));
        }
    }

    printf("n %d  time %.3f ms  %.2f TFLOPS  max err %.4f\n", n, ms, tflops, max_err);

    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o); cudaFree(s);
    return 0;
}
