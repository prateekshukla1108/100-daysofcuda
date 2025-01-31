#include <iostream>

#include <cstdlib>

#include <cstdio>

#include <cuda_runtime.h>


#define BLOCK_SIZE 16

#define EPSILON 1e-3


// CUDA Kernel for 2D Tiled Matrix Multiplication

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {

    // Calculate row and column indices for this thread

    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int col = blockIdx.x * blockDim.x + threadIdx.x;


    // Shared memory tiles

    __shared__ float A_tile[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ float B_tile[BLOCK_SIZE][BLOCK_SIZE];


    float sum = 0.0f;


    // Loop over tiles along K dimension

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {

        // Load elements into shared memory

        int A_col = t * BLOCK_SIZE + threadIdx.x;

        int B_row = t * BLOCK_SIZE + threadIdx.y;


        // Load A tile element

        if (row < M && A_col < K) {

            A_tile[threadIdx.y][threadIdx.x] = A[row * K + A_col];

        } else {

            A_tile[threadIdx.y][threadIdx.x] = 0.0f;

        }


        // Load B tile element

        if (B_row < K && col < N) {

            B_tile[threadIdx.y][threadIdx.x] = B[B_row * N + col];

        } else {

            B_tile[threadIdx.y][threadIdx.x] = 0.0f;

        }


        __syncthreads();


        // Compute partial product

        for (int k = 0; k < BLOCK_SIZE; ++k) {

            sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];

        }


        __syncthreads();

    }


    // Write result to global memory

    if (row < M && col < N) {

        C[row * N + col] = sum;

    }

}


// CPU reference implementation

void matmul_cpu(float *A, float *B, float *C, int M, int N, int K) {

    for (int i = 0; i < M; ++i) {

        for (int j = 0; j < N; ++j) {

            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {

                sum += A[i * K + k] * B[k * N + j];

            }

            C[i * N + j] = sum;

        }

    }

}


// Function to compare GPU and CPU results

bool verify_results(float *cpu, float *gpu, int size) {

    for (int i = 0; i < size; ++i) {

        if (fabs(cpu[i] - gpu[i]) > EPSILON) {

            printf("Mismatch at index %d: CPU=%.5f, GPU=%.5f\n", i, cpu[i], gpu[i]);

            return false;

        }

    }

    return true;

}


int main() {

    // Matrix dimensions

    const int M = 512;

    const int N = 512;

    const int K = 512;


    // Allocate host memory

    float *h_A = new float[M * K];

    float *h_B = new float[K * N];

    float *h_C = new float[M * N];

    float *h_C_ref = new float[M * N];


    // Initialize matrices with random values

    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;

    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;


    // Allocate device memory

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, M * K * sizeof(float));

    cudaMalloc(&d_B, K * N * sizeof(float));

    cudaMalloc(&d_C, M * N * sizeof(float));


    // Copy data to device

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);


    // Launch kernel

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);


    // Copy result back to host

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);


    // Compute reference on CPU

    matmul_cpu(h_A, h_B, h_C_ref, M, N, K);


    // Verify results

    bool correct = verify_results(h_C_ref, h_C, M * N);

    if (correct) {

        std::cout << "Results verified successfully!\n";

    } else {

        std::cerr << "Results verification failed!\n";

    }


    // Cleanup

    delete[] h_A;

    delete[] h_B;

    delete[] h_C;

    delete[] h_C_ref;

    cudaFree(d_A);

    cudaFree(d_B);

    cudaFree(d_C);


    return 0;

}
