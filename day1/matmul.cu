#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 420
#define M 69
#define K 420

__global__ void matmul(float *A, float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < M) {
    C[row * M + col] = 0;
    for (int k = 0; k < K; k++) {
    C[row * M + col] += A[row * K + k] * B[k * M + col];
}


}

}

void initialize_random_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX; 
    }
}

void print_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    float *A, *B, *C;
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(N * K * sizeof(float));
    h_B = (float*)malloc(K * M * sizeof(float));
    h_C = (float*)malloc(N * M * sizeof(float));
    srand(time(NULL)); // Seed the random number generator
    initialize_random_matrix(h_A, N, K);
    initialize_random_matrix(h_B, K, M);
    cudaMalloc(&A, N * K * sizeof(float));
    cudaMalloc(&B, K * M * sizeof(float));
    cudaMalloc(&C, N * M * sizeof(float));
    cudaMemcpy(A, h_A, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_B, K * M * sizeof(float), cudaMemcpyHostToDevice);
    dim3 blockDim(16, 16);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);
    matmul<<<gridDim, blockDim>>>(A, B, C);

    cudaMemcpy(h_C, C, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    print_matrix(h_C, 5, 5);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
