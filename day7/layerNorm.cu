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

    extern __shared__ float shared[];
    float* s_sum = shared;             
    float* s_sum_sq = shared + blockDim.x;  

    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float val = input_instance[i];
        sum += val;
        sum_sq += val * val;
    }
    s_sum[tid] = sum;
    s_sum_sq[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }


    float mean = s_sum[0] / feature_size;
    float var = s_sum_sq[0] / feature_size - mean * mean;
    float inv_std = rsqrtf(var + epsilon);  

    for (int i = tid; i < feature_size; i += blockDim.x) {
        float norm_val = (input_instance[i] - mean) * inv_std;
        // Affine transformation: scale by gamma and shift by beta.
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
        threads = feature_size;

    // Each block processes one instance.
    int blocks = batch_size;

    size_t sharedMemSize = 2 * threads * sizeof(float);

    LayerNormKernel<<<blocks, threads, sharedMemSize>>>(d_input, d_output, d_gamma, d_beta, feature_size, epsilon);
    // Check for any errors 
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

