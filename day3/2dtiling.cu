#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void tiledMatMul(float* C, float* A, float* B, int M, int N, int K) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];
    
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_WIDTH - 1)/TILE_WIDTH; ++t) {
        int tiledX = t * TILE_WIDTH + threadIdx.x;
        int tiledY = t * TILE_WIDTH + threadIdx.y;
        
        if (row < M && tiledX < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + tiledX];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && tiledY < K)
            sB[threadIdx.y][threadIdx.x] = B[tiledY * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

int main() {
    int M = 2, N = 2, K = 2;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float h_A[] = {1.0f, 2.0f,   
                   3.0f, 4.0f};
    float h_B[] = {5.0f, 6.0f,   
                   7.0f, 8.0f};
    float h_C[4] = {0};

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1)/TILE_WIDTH, 
                 (M + TILE_WIDTH - 1)/TILE_WIDTH);

    tiledMatMul<<<gridDim, blockDim>>>(d_C, d_A, d_B, M, N, K);

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    printf("Matrix C (Result):\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%8.2f", h_C[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

