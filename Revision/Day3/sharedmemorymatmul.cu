#include <iostream>
using namespace std;

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

int main() {
    int M = 4;
    int N = 4;
    int K = 4;

    float A[4][4];
    float B[4][4];
    float C[4][4];

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

    reference_matmul(A, B, C, M, N, K);

    cout << "Matrix A:" << endl;
    print_matrix_4x4(A, M, N);

    cout << endl;
    cout << "Matrix B:" << endl;
    print_matrix_4x4(B, N, K);

    cout << endl;
    cout << "Matrix C = A x B:" << endl;
    print_matrix_4x4(C, M, K);

    return 0;
}