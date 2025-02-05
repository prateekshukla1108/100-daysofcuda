#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

#define TILE_WIDTH 4
#define MASK_WIDTH 3
#define MASK_RADIUS (MASK_WIDTH / 2)

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

__global__ void conv2D(const float *input, float *output, const float *mask, int width, int height) {
    __shared__ float tile[TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_WIDTH + ty; 
    int col_o = blockIdx.x * TILE_WIDTH + tx; 
    int row_i = row_o - MASK_RADIUS;          
    int col_i = col_o - MASK_RADIUS;          

    // Load the main tile element into shared memory.
    if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width)
        tile[ty][tx] = input[row_i * width + col_i];
    else
        tile[ty][tx] = 0.0f;

    // Load the extra column (right halo)
    int tx_extra = tx + TILE_WIDTH;
    if (tx_extra < TILE_WIDTH + MASK_WIDTH - 1) {
        if (row_i >= 0 && row_i < height && (col_i + TILE_WIDTH) < width)
            tile[ty][tx_extra] = input[row_i * width + (col_i + TILE_WIDTH)];
        else
            tile[ty][tx_extra] = 0.0f;
    }

    
    int ty_extra = ty + TILE_WIDTH;
    if (ty_extra < TILE_WIDTH + MASK_WIDTH - 1) {
        if ((row_i + TILE_WIDTH) < height && col_i >= 0 && col_i < width)
            tile[ty_extra][tx] = input[(row_i + TILE_WIDTH) * width + col_i];
        else
            tile[ty_extra][tx] = 0.0f;
    }

    if (tx_extra < TILE_WIDTH + MASK_WIDTH - 1 && ty_extra < TILE_WIDTH + MASK_WIDTH - 1) {
        if ((row_i + TILE_WIDTH) < height && (col_i + TILE_WIDTH) < width)
            tile[ty_extra][tx_extra] = input[(row_i + TILE_WIDTH) * width + (col_i + TILE_WIDTH)];
        else
            tile[ty_extra][tx_extra] = 0.0f;
    }

    __syncthreads();

    if (row_o < height && col_o < width) {
        float sum = 0.0f;
        // Loop over the convolution mask.
        for (int i = 0; i < MASK_WIDTH; i++) {
            for (int j = 0; j < MASK_WIDTH; j++) {
                sum += mask[i * MASK_WIDTH + j] * tile[ty + i][tx + j];
            }
        }
        output[row_o * width + col_o] = sum;
    }
}

void printMatrix(const float *matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << std::setw(6) << matrix[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    const int width = 8;
    const int height = 8;
    float h_input[width * height] = {
         1, 2, 3, 4, 5, 6, 7, 8,
         8, 7, 6, 5, 4, 3, 2, 1,
         1, 3, 5, 7, 9, 7, 5, 3,
         2, 4, 6, 8, 6, 4, 2, 0,
         0, 2, 4, 6, 8, 6, 4, 2,
         1, 3, 5, 7, 9, 7, 5, 3,
         8, 6, 4, 2, 0, 2, 4, 6,
         7, 5, 3, 1, 2, 3, 4, 5
    };

    // Define a simple 3x3 averaging mask
    float h_mask[MASK_WIDTH * MASK_WIDTH] = {
        1/9.0f, 1/9.0f, 1/9.0f,
        1/9.0f, 1/9.0f, 1/9.0f,
        1/9.0f, 1/9.0f, 1/9.0f
    };

    float h_output[width * height] = {0};

    float *d_input, *d_output, *d_mask;
    size_t imageSize = width * height * sizeof(float);
    size_t maskSize = MASK_WIDTH * MASK_WIDTH * sizeof(float);

    cudaError_t err;
    err = cudaMalloc((void**)&d_input, imageSize);
    CUDA_CHECK(err);
    err = cudaMalloc((void**)&d_output, imageSize);
    CUDA_CHECK(err);
    err = cudaMalloc((void**)&d_mask, maskSize);
    CUDA_CHECK(err);

    err = cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);
    err = cudaMemcpy(d_mask, h_mask, maskSize, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);

    conv2D<<<dimGrid, dimBlock>>>(d_input, d_output, d_mask, width, height);
    cudaDeviceSynchronize();

    // Copy the output data back to the host.
    err = cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);

    std::cout << "Input Matrix:" << std::endl;
    printMatrix(h_input, width, height);

    std::cout << "\nMask:" << std::endl;
    for (int i = 0; i < MASK_WIDTH * MASK_WIDTH; i++) {
        std::cout << std::setw(6) << h_mask[i] << " ";
        if ((i + 1) % MASK_WIDTH == 0) std::cout << std::endl;
    }

    std::cout << "\nOutput Matrix (After Convolution):" << std::endl;
    printMatrix(h_output, width, height);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);

    return 0;
}

