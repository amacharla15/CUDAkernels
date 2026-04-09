#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

__global__ void naive_matmul(const float* A, const float* B, float* C, int M, int N, int K) { //naive matmul kernel 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < K) {
        float sum1 = 0.0f; //K is dimension of the matrix and small k is running index could be replaced with temp but 
        // I usually use k as running index
        for (int k = 0; k < N; k++) {
            sum1 = sum1 + A[row * N + k] * B[k * K + col];
        }
        C[row * K + col] = sum1;
    }
}

__global__ void shared_matmul(const float*A, const float*B, float*C, int M, int N, int K){
    int row= blockIdx.y * blockDim.y + threadIdx.y;
    int col= blockIdx.x * blockDim.x + threadIdx.x;

    int local_row= threadIdx.y;
    int local_col= threadIdx.x; // blocklevel for shared memory
    
    const int tile_size = 16;

    __shared__ float tile_A[16][16];
    __shared__ float tile_B[16][16];

    float sum1 = 0.0f;

    int num_phases=(N+tile_size-1)/tile_size;

    for (int phases=0;phases<num_phases;phases++){ // phases for each block
        int a_global_row = row;
        int a_global_col = phases * tile_size + local_col;
                                                        // each tile loads values according to this formula/way
        int b_global_row = phases * tile_size + local_row;
        int b_global_col = col;

        //shared memory loading  for A and B 
        // 2D is stored as 1D in memory so " a_global_row * N + a_global_col" 
        if (a_global_row < M && a_global_col < N) {
            tile_A[local_row][local_col] = A[a_global_row * N + a_global_col];
        } else {
            tile_A[local_row][local_col] = 0.0f;
        }
        // 2D is stored as 1D in memory so " b_global_row * K + b_global_col" 
        if (b_global_row < N && b_global_col < K) {
            tile_B[local_row][local_col] = B[b_global_row * K + b_global_col];
        } else {
            tile_B[local_row][local_col] = 0.0f;
        }
        // all threads need to sync 
        __syncthreads();

        // actual dot product 
        for (int t = 0; t < tile_size; t++) {
            sum1 = sum1 + tile_A[local_row][t] * tile_B[t][local_col];
        }

        //this is because we create one shared-memory array for the block not different tiles , such as 4 tiles with size of 2
        // if matrices are of 4X4 size , so reuse of arrays will occur and other threads may rewrite one arrays values when
        //computation is going on 
        __syncthreads();

        
    }
    if (row < M && col < K) {
        C[row * K + col] = sum1;
    }


}
 // cpu version of matmul just to check if the kernels output matches our local matmul works for any matrix
void reference_matmul(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float sum1 = 0.0f;
            for (int k = 0; k < N; k++) {
                sum1 = sum1 + A[i * N + k] * B[k * K + j];
            }
            C[i * K + j] = sum1;
        }
    }
}
// debug function used to print our matrices
void print_matrix(float* X, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << X[i * cols + j] << " ";
        }
        cout << endl;
    }
}
// comparison matrix for simpler way of printing if the matrices matches or not 
bool compare_matrix(float* X, float* Y, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fabs(X[i * cols + j] - Y[i * cols + j]) > 1e-5) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    // sizes of our matrics , can be replaced with any valid dimension numbers 
    srand(time(0));
    int sizes[4] = {128, 256, 512, 1024};
    int repeats = 20;

    for (int s = 0; s < 4; s++) {
        // checking if the user inputs are corret or niot 
        int M = sizes[s];
        int N = sizes[s];
        int K = sizes[s];

        if (M <= 0 || N <= 0 || K <= 0) {
            cout << "Invalid matrix sizes" << endl;
            return 0;
        }

        //matrices
        float* A = new float[M * N];
        float* B = new float[N * K];
        float* C_ref = new float[M * K];
        float* C_naive = new float[M * K];
        float* C_shared = new float[M * K];

        // random values initialziation 
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = (rand() % 10) + 1;
            }
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                B[i * K + j] = (rand() % 10) + 1;
            }
        }

        reference_matmul(A, B, C_ref, M, N, K);

        //pointers to our gpu memory 
        float* d_A;
        float* d_B;
        float* d_C; 

        //memory allocation
        cudaMalloc((void**)&d_A, M * N * sizeof(float));
        cudaMalloc((void**)&d_B, N * K * sizeof(float));
        cudaMalloc((void**)&d_C, M * K * sizeof(float));

        //copying the memory to gpu
        cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, N * K * sizeof(float), cudaMemcpyHostToDevice);

        //cuda timers for kernels
        cudaEvent_t start_naive, stop_naive, start_shared, stop_shared;
        cudaEventCreate(&start_naive);
        cudaEventCreate(&stop_naive);
        cudaEventCreate(&start_shared);
        cudaEventCreate(&stop_shared);

        float naive_time_ms = 0.0f;
        float shared_time_ms = 0.0f;
        double naive_total_ms = 0.0;
        double shared_total_ms = 0.0;

        //block dimensions and threads per block 
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // kernel launch 

        // timers and Flops check with both CPU and CUDA timers

        //warmup
        naive_matmul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();

        shared_matmul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();

        for (int r = 0; r < repeats; r++) {
            cudaEventRecord(start_naive);
            naive_matmul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
            cudaEventRecord(stop_naive);
            cudaEventSynchronize(stop_naive);
            cudaEventElapsedTime(&naive_time_ms, start_naive, stop_naive);
            naive_total_ms += naive_time_ms;
        }

        cudaMemcpy(C_naive, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost); // copy back our results

        // shared_timer
        for (int r = 0; r < repeats; r++) {
            cudaEventRecord(start_shared);
            shared_matmul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
            cudaEventRecord(stop_shared);
            cudaEventSynchronize(stop_shared);
            cudaEventElapsedTime(&shared_time_ms, start_shared, stop_shared);
            shared_total_ms += shared_time_ms;
        }

        cudaMemcpy(C_shared, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost); // copy back our results

        double naive_avg_ms = naive_total_ms / repeats;
        double shared_avg_ms = shared_total_ms / repeats;

        double flops = 2.0 * M * N * K;
        double naive_gflops = flops / (naive_avg_ms / 1000.0) / 1e9;
        double shared_gflops = flops / (shared_avg_ms / 1000.0) / 1e9;
        double speedup = naive_avg_ms / shared_avg_ms;

        bool naive_ok = compare_matrix(C_ref, C_naive, M, K);
        bool shared_ok = compare_matrix(C_ref, C_shared, M, K);

        cout << "Size: " << M << "x" << N << "x" << K << endl;
        cout << "Naive avg time: " << naive_avg_ms << " ms" << endl;
        cout << "Shared avg time: " << shared_avg_ms << " ms" << endl;
        cout << "Naive GFLOPS: " << naive_gflops << endl;
        cout << "Shared GFLOPS: " << shared_gflops << endl;
        cout << "Speedup (naive/shared): " << speedup << "x" << endl;

        cout << endl;
        if (naive_ok) {
            cout << "CPU and GPU Naive results match" << endl;
        } else {
            cout << "CPU and GPU Naive results do not match" << endl;
        }

        cout << endl;
        if (shared_ok) {
            cout << "CPU and GPU Shared results match" << endl;
        } else {
            cout << "CPU and GPU Shared results do not match" << endl;
        }

        cout << endl;
        if (compare_matrix(C_naive, C_shared, M, K)) {
            cout << "GPU Naive and GPU Shared results match" << endl;
        } else {
            cout << "GPU Naive and GPU Shared results do not match" << endl;
        }

        cout << "-----------------------------" << endl;

        //freeing the memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        delete[] A;
        delete[] B;
        delete[] C_ref;
        delete[] C_naive;
        delete[] C_shared;

        cudaEventDestroy(start_naive);
        cudaEventDestroy(stop_naive);
        cudaEventDestroy(start_shared);
        cudaEventDestroy(stop_shared);
    }

    return 0;
}