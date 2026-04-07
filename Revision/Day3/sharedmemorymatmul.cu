#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
using namespace std;

__global__ void naive_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < K) {
        float sum1 = 0.0f;
        for (int k = 0; k < N; k++) {
            sum1 = sum1 + A[row * N + k] * B[k * K + col];
        }
        C[row * K + col] = sum1;
    }
}

void reference_matmul(float A[][4], float B[][4], float C[][4], int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float sum1 = 0.0f;
            for (int k = 0; k < N; k++) {
                sum1 = sum1 + A[i][k] * B[k][j];
            }
            C[i][j] = sum1;
        }
    }
}

void print_matrix_4x4(float X[][4], int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << X[i][j] << " ";
        }
        cout << endl;
    }
}

bool compare_matrix_4x4(float X[][4], float Y[][4], int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fabs(X[i][j] - Y[i][j]) > 1e-5) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    int M = 4;
    int N = 4;
    int K = 4;

    float A[4][4];
    float B[4][4];
    float C_ref[4][4];
    float C_gpu[4][4];

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = 1.0f;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            B[i][j] = 1.0f;
        }
    }

    reference_matmul(A, B, C_ref, M, N, K);

    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc((void**)&d_A, M * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * K * sizeof(float));
    cudaMalloc((void**)&d_C, M * K * sizeof(float));

    cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    naive_matmul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(C_gpu, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Matrix A:" << endl;
    print_matrix_4x4(A, M, N);

    cout << endl;
    cout << "Matrix B:" << endl;
    print_matrix_4x4(B, N, K);

    cout << endl;
    cout << "CPU Reference C:" << endl;
    print_matrix_4x4(C_ref, M, K);

    cout << endl;
    cout << "GPU Naive C:" << endl;
    print_matrix_4x4(C_gpu, M, K);

    cout << endl;
    if (compare_matrix_4x4(C_ref, C_gpu, M, K)) {
        cout << "CPU and GPU results match" << endl;
    } else {
        cout << "CPU and GPU results do not match" << endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}