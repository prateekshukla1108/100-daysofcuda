#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Tile sizes for the kernel
#define TILE_K 32
#define TILE_N 32
#define BLOCK_SIZE 32

// Error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Kernel
__global__ void matmul_1d_tiling(float *C, const float *A, const float *B, int M, int N, int K) {
    const int row = blockIdx.x;  
    const int j_start = blockIdx.y * TILE_N; 
    const int tx = threadIdx.x;
    const int j = j_start + tx;  

    __shared__ float A_shared[TILE_K];
    __shared__ float B_shared[TILE_K][TILE_N];
    
    float sum = 0.0f;
    
    for (int k_tile_start = 0; k_tile_start < K; k_tile_start += TILE_K) {
        const int k_a = k_tile_start + tx;
        if (k_a < K) {
            A_shared[tx] = A[row * K + k_a];
        } else {
            A_shared[tx] = 0.0f;
        }
        
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            const int k_b = k_tile_start + k;
            if (k_b < K && j < N) {
                B_shared[k][tx] = B[k_b * N + j];
            } else {
                B_shared[k][tx] = 0.0f;
            }
        }
        
        __syncthreads(); 
        
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            sum += A_shared[k] * B_shared[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && j < N) {
        C[row * N + j] = sum;
    }
}


void matmul_gpu(float *h_C, const float *h_A, const float *h_B, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate GPU memory
    cudaCheckError(cudaMalloc(&d_A, size_A));
    cudaCheckError(cudaMalloc(&d_B, size_B));
    cudaCheckError(cudaMalloc(&d_C, size_C));


    cudaCheckError(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + TILE_N - 1) / TILE_N);
    dim3 block(TILE_N);

    // Launch kernel
    matmul_1d_tiling<<<grid, block>>>(d_C, d_A, d_B, M, N, K);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());


    cudaCheckError(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(d_A));
    cudaCheckError(cudaFree(d_B));
    cudaCheckError(cudaFree(d_C));
}

void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
     mat[i] = (float)(rand()) / RAND_MAX;
    }
}

int main() {
    const int M = 1024; 
    const int N = 1024; 
    const int K = 1024;  

    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));

    // Initialize input matrices
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // Perform matrix multiplication on GPU
    matmul_gpu(h_C, h_A, h_B, M, N, K);

    // show the result
    printf("GPU Results:\n");
    for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      printf("%.4f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // goodbye cruel world
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
