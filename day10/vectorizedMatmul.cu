#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) do {                                    \
    cudaError_t err = call;                                      \
    if(err != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err));    \
        exit(EXIT_FAILURE);                                      \
    }                                                            \
} while (0)

__global__ void matmul_vectorized(const float * __restrict__ A,
                                  const float * __restrict__ B,
                                  float * __restrict__ C,
                                  int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        int vecK = K / 4;

        const float4* A_vec = reinterpret_cast<const float4*>(A);
        const float4* B_vec = reinterpret_cast<const float4*>(B);

        int A_offset = row * vecK;
        int B_offset = col * vecK;

        for (int i = 0; i < vecK; i++) {
            float4 a_val = A_vec[A_offset + i];
            float4 b_val = B_vec[B_offset + i];
            sum += a_val.x * b_val.x +
                   a_val.y * b_val.y +
                   a_val.z * b_val.z +
                   a_val.w * b_val.w;
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int M = 4;  
    int K = 8;  
    int N = 3;  

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            h_A[i * K + j] = (float)(i * K + j + 1);
        }
    }

    for (int col = 0; col < N; col++) {
        for (int row = 0; row < K; row++) {
            h_B[col * K + row] = 1.0f;
        }
    }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    matmul_vectorized<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    printf("Matrix A (%d x %d) [row-major]:\n", M, K);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            printf("%6.1f ", h_A[i * K + j]);
        }
        printf("\n");
    }
    printf("\n");

    printf("Matrix B (%d x %d) [column-major]:\n", K, N);
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6.1f ", h_B[i + j * K]);
        }
        printf("\n");
    }
    printf("\n");

    printf("Result Matrix C (%d x %d) = A * B:\n", M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6.1f ", h_C[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");

    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}

