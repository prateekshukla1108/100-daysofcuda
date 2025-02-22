#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <iostream>

#define N 4096  // Matrix size (adjust as needed)
#define BLOCK_SIZE 32  // CUDA block size

__global__ void matmul_kernel(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void cuda_matmul(float* d_A, float* d_B, float* d_C, int n) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(n / threads.x, n / threads.y);
    matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Initialize NCCL
    ncclComm_t nccl_comm;
    ncclUniqueId nccl_id;

    if (world_rank == 0) ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&nccl_comm, world_size, nccl_id, world_rank);

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    // Simulate distributed matrix chunk (each GPU gets a portion)
    cuda_matmul(d_A, d_B, d_C, N / world_size);

    // NCCL AllReduce: Aggregate results across GPUs
    ncclAllReduce(d_C, d_C, (N * N) / world_size, ncclFloat, ncclSum, nccl_comm, cudaStreamDefault);
    
    // Cleanup
    ncclCommDestroy(nccl_comm);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    MPI_Finalize();

    return 0;
}

