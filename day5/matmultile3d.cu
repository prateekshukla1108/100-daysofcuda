#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>

// More descriptive error checking macro
#define CUDA_ERROR_CHECK(call) do { \
    cudaError_t cudaError = (call); \
    if (cudaError != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(cudaError) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// Kernel configuration constants
constexpr int MATRIX_TILE_HEIGHT = 16;
constexpr int MATRIX_TILE_WIDTH = 16;
constexpr int MATRIX_TILE_DEPTH = 16;

__global__ void BatchedMatrixMultiplyKernel(
    const float* __restrict__ inputMatrixA, 
    const float* __restrict__ inputMatrixB, 
    float* __restrict__ outputMatrixC, 
    int numRows, int numColumns, int sharedDimension, 
    int numBatches) 
{
    // Shared memory for tiled matrix multiplication
    __shared__ float sharedA[MATRIX_TILE_HEIGHT][MATRIX_TILE_DEPTH];
    __shared__ float sharedB[MATRIX_TILE_DEPTH][MATRIX_TILE_WIDTH];

    const int batchIndex = blockIdx.z;
    const int rowIndex = blockIdx.x * MATRIX_TILE_HEIGHT + threadIdx.x;
    const int colIndex = blockIdx.y * MATRIX_TILE_WIDTH + threadIdx.y;
    const int threadX = threadIdx.x;
    const int threadY = threadIdx.y;

    float accumulatedSum = 0.0f;

    // Iterate through matrix tiles
    for (int tileOffset = 0; tileOffset < sharedDimension; tileOffset += MATRIX_TILE_DEPTH) {
        // Load tiles into shared memory
        if (rowIndex < numRows && (tileOffset + threadY) < sharedDimension) {
            sharedA[threadX][threadY] = inputMatrixA[
                (rowIndex * sharedDimension + (tileOffset + threadY)) * numBatches + batchIndex
            ];
        } else {
            sharedA[threadX][threadY] = 0.0f;
        }

        if ((tileOffset + threadX) < sharedDimension && colIndex < numColumns) {
            sharedB[threadX][threadY] = inputMatrixB[
                ((tileOffset + threadX) * numColumns + colIndex) * numBatches + batchIndex
            ];
        } else {
            sharedB[threadX][threadY] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum within the tile
        for (int k = 0; k < MATRIX_TILE_DEPTH; k++) {
            accumulatedSum += sharedA[threadX][k] * sharedB[k][threadY];
        }

        __syncthreads();
    }

    // Store final result
    if (rowIndex < numRows && colIndex < numColumns) {
        outputMatrixC[(rowIndex * numColumns + colIndex) * numBatches + batchIndex] = accumulatedSum;
    }
}

// CPU reference implementation for verification
void ReferenceMatrixMultiply(
    const float* inputA, 
    const float* inputB, 
    float* outputC, 
    int numRows, 
    int numColumns, 
    int sharedDimension, 
    int numBatches) 
{
    for (int batch = 0; batch < numBatches; batch++) {
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numColumns; j++) {
                float sum = 0.0f;
                for (int k = 0; k < sharedDimension; k++) {
                    int indexA = (i * sharedDimension + k) * numBatches + batch;
                    int indexB = (k * numColumns + j) * numBatches + batch;
                    sum += inputA[indexA] * inputB[indexB];
                }
                int indexC = (i * numColumns + j) * numBatches + batch;
                outputC[indexC] = sum;
            }
        }
    }
}

int main() {
    // Matrix and batch configuration
    const int numRows = 64, numColumns = 64, sharedDimension = 64;
    const int numBatches = 16;

    // Memory allocation sizes
    const size_t matrixASize = numRows * sharedDimension * numBatches * sizeof(float);
    const size_t matrixBSize = sharedDimension * numColumns * numBatches * sizeof(float);
    const size_t matrixCSize = numRows * numColumns * numBatches * sizeof(float);

    // Allocate host memory
    float *hostMatrixA = new float[numRows * sharedDimension * numBatches];
    float *hostMatrixB = new float[sharedDimension * numColumns * numBatches];
    float *hostMatrixC = new float[numRows * numColumns * numBatches];
    float *hostMatrixCReference = new float[numRows * numColumns * numBatches];

    // Initialize input matrices
    for (int batch = 0; batch < numBatches; batch++) {
        for (int i = 0; i < numRows; i++) {
            for (int k = 0; k < sharedDimension; k++) {
                int indexA = (i * sharedDimension + k) * numBatches + batch;
                hostMatrixA[indexA] = static_cast<float>(i + k + batch);
            }
        }
        for (int k = 0; k < sharedDimension; k++) {
            for (int j = 0; j < numColumns; j++) {
                int indexB = (k * numColumns + j) * numBatches + batch;
                hostMatrixB[indexB] = static_cast<float>(k - j + batch);
            }
        }
    }

    // Allocate device memory
    float *deviceMatrixA, *deviceMatrixB, *deviceMatrixC;
    CUDA_ERROR_CHECK(cudaMalloc(&deviceMatrixA, matrixASize));
    CUDA_ERROR_CHECK(cudaMalloc(&deviceMatrixB, matrixBSize));
    CUDA_ERROR_CHECK(cudaMalloc(&deviceMatrixC, matrixCSize));

    // Copy input data to device
    CUDA_ERROR_CHECK(cudaMemcpy(deviceMatrixA, hostMatrixA, matrixASize, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(deviceMatrixB, hostMatrixB, matrixBSize, cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    dim3 gridDimension(
        (numRows + MATRIX_TILE_HEIGHT - 1) / MATRIX_TILE_HEIGHT,
        (numColumns + MATRIX_TILE_WIDTH - 1) / MATRIX_TILE_WIDTH, 
        numBatches
    );
    dim3 blockDimension(MATRIX_TILE_HEIGHT, MATRIX_TILE_WIDTH);
    
    // Launch CUDA kernel
    BatchedMatrixMultiplyKernel<<<gridDimension, blockDimension>>>(
        deviceMatrixA, deviceMatrixB, deviceMatrixC, 
        numRows, numColumns, sharedDimension, numBatches
    );
    CUDA_ERROR_CHECK(cudaGetLastError());

    // Copy results back to host
    CUDA_ERROR_CHECK(cudaMemcpy(hostMatrixC, deviceMatrixC, matrixCSize, cudaMemcpyDeviceToHost));

    // Compute reference solution on CPU
    ReferenceMatrixMultiply(
        hostMatrixA, hostMatrixB, hostMatrixCReference, 
        numRows, numColumns, sharedDimension, numBatches
    );

    // Verify results
    int errorCount = 0;
    constexpr float COMPUTATIONAL_TOLERANCE = 1e-4f;
    for (int batch = 0; batch < numBatches; batch++) {
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numColumns; j++) {
                int index = (i * numColumns + j) * numBatches + batch;
                float difference = std::abs(hostMatrixC[index] - hostMatrixCReference[index]);
                
                if (difference > COMPUTATIONAL_TOLERANCE) {
                    if (errorCount++ < 10) {
                        std::cout << "Mismatch in Batch " << batch 
                                  << ", Position [" << i << "][" << j << "]: "
                                  << hostMatrixC[index] << " vs " 
                                  << hostMatrixCReference[index] << std::endl;
                    }
                }
            }
        }
    }

    // Report results
    if (errorCount > 0) {
        std::cerr << errorCount << " computational errors detected!" << std::endl;
    } else {
        std::cout << "All results verified successfully!" << std::endl;
    }

    // Clean up resources
    delete[] hostMatrixA;
    delete[] hostMatrixB;
    delete[] hostMatrixC;
    delete[] hostMatrixCReference;
    
    CUDA_ERROR_CHECK(cudaFree(deviceMatrixA));
    CUDA_ERROR_CHECK(cudaFree(deviceMatrixB));
    CUDA_ERROR_CHECK(cudaFree(deviceMatrixC));

    return 0;
}
