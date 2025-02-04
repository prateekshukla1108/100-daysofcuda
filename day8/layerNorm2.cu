#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA Error: %s (line %d) in file %s\n",             \
                    cudaGetErrorString(err), __LINE__, __FILE__);                \
            exit(err);                                                           \
        }                                                                        \
    } while (0)

__global__ void LayerNormKernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                const float* __restrict__ gamma,
                                const float* __restrict__ beta,
                                int feature_size,
                                float epsilon) {
    int instance = blockIdx.x;
    int tid = threadIdx.x;

    const float* input_instance = input + instance * feature_size;
    float* output_instance = output + instance * feature_size;

    extern __shared__ float s_data[];
    float* s_input = s_data;
    const int num_warps = (blockDim.x + 31) / 32;
    float* s_sum = s_data + feature_size;
    float* s_sum_sq = s_sum + num_warps;

    // Load input into shared memory with coalesced access
    const int elements_per_thread = (feature_size + blockDim.x - 1) / blockDim.x;
    const int start = tid * elements_per_thread;
    const int end = min(start + elements_per_thread, feature_size);
    
    for (int i = start; i < end; ++i) {
        s_input[i] = input_instance[i];
    }
    __syncthreads();

    // Compute partial sums
    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = start; i < end; ++i) {
        const float val = s_input[i];
        sum += val;
        sum_sq += val * val;
    }

    // Warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Store warp sums to shared memory
    if (tid % 32 == 0) {
        const int warp_id = tid / 32;
        s_sum[warp_id] = sum;
        s_sum_sq[warp_id] = sum_sq;
    }
    __syncthreads();

    // Final reduction using first warp
    if (tid < 32) {
        float warp_sum = tid < num_warps ? s_sum[tid] : 0.0f;
        float warp_sum_sq = tid < num_warps ? s_sum_sq[tid] : 0.0f;

        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
            warp_sum_sq += __shfl_down_sync(0xffffffff, warp_sum_sq, offset);
        }

        if (tid == 0) {
            s_sum[0] = warp_sum;
            s_sum_sq[0] = warp_sum_sq;
        }
    }
    __syncthreads();

    // Compute statistics
    const float mean = s_sum[0] / feature_size;
    const float var = s_sum_sq[0] / feature_size - mean * mean;
    const float inv_std = rsqrtf(var + epsilon);

    // Apply normalization and write results
    for (int i = start; i < end; ++i) {
        const float norm_val = (s_input[i] - mean) * inv_std;
        output_instance[i] = norm_val * gamma[i] + beta[i];
    }
}

void launchLayerNorm(const float* d_input,
                     float* d_output,
                     const float* d_gamma,
                     const float* d_beta,
                     int batch_size,
                     int feature_size,
                     float epsilon) {
    int threads = 256;
    if (feature_size < threads) 
        threads = min(feature_size, 32);  // Minimum 32 threads for efficient warp usage

    const int num_warps = (threads + 31) / 32;
    const size_t sharedMemSize = feature_size * sizeof(float) 
                               + 2 * num_warps * sizeof(float);

    dim3 blocks(batch_size);
    dim3 threadsPerBlock(threads);

    LayerNormKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(
        d_input, d_output, d_gamma, d_beta, feature_size, epsilon
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main() {
    const int batch_size = 32;
    const int feature_size = 1024;
    const float epsilon = 1e-5f;

    // Calculate the data size
    size_t dataSize = batch_size * feature_size * sizeof(float);
    size_t paramSize = feature_size * sizeof(float);

    float *h_input  = (float*)malloc(dataSize);
    float *h_output = (float*)malloc(dataSize);
    float *h_gamma  = (float*)malloc(paramSize);
    float *h_beta   = (float*)malloc(paramSize);

    if (!h_input || !h_output || !h_gamma || !h_beta) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host data
    for (int i = 0; i < batch_size * feature_size; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < feature_size; ++i) {
        h_gamma[i] = 1.0f;  // or some meaningful initialization
        h_beta[i]  = 0.0f;
    }

    float *d_input, *d_output, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_input, dataSize));
    CUDA_CHECK(cudaMalloc(&d_output, dataSize));
    CUDA_CHECK(cudaMalloc(&d_gamma, paramSize));
    CUDA_CHECK(cudaMalloc(&d_beta, paramSize));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, dataSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, paramSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta, paramSize, cudaMemcpyHostToDevice));

    launchLayerNorm(d_input, d_output, d_gamma, d_beta, batch_size, feature_size, epsilon);

    // Copy the results back to the host.
    CUDA_CHECK(cudaMemcpy(h_output, d_output, dataSize, cudaMemcpyDeviceToHost));

    // Optionally, verify the output (omitted here).

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    free(h_input);
    free(h_output);
    free(h_gamma);
    free(h_beta);

    printf("Layer normalization completed successfully\n");
    return 0;
}

