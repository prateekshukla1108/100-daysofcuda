#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>
#include <math.h>

// Helper macro to check for CUDA errors.
#define CHECK_CUDA_ERROR(call) {                                   \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error at %s:%d - %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
}

// Original forward kernels

// Fused GEMM with ReLU activation kernel.
__global__ void fusedGemmReLUKernel(const float* A, const float* B, const float* bias, float* C,
                                    int M, int N, int K);

// Simple matrix multiplication kernel.
__global__ void matmulKernel(const float* A, const float* B, float* C,
                             int M, int N, int K);

// Kernel for adding bias and residual connection.
__global__ void addBiasAndResidualKernel(float* attn, float* temp, const float* b2, int seq, int dim);

// Restructured embedding kernel for coalesced access.
__global__ void embeddingLookupKernel(const int* d_input, const float* d_embedding,
                                       float* d_output, int seq_length);

// Kernel to add positional encodings.
__global__ void addPositionalEncoding(float* d_embedded, int seq_length, int embed_dim);

// Dummy self-attention kernel.
__global__ void dummySelfAttentionKernel(const float* d_input, float* d_output, int seq_length);

// Final linear layer kernel.
__global__ void linearOutputKernel(const float* d_input, const float* d_W_out, const float* d_b_out,
                                    float* d_logits, int seq_length);

// --- Backpropagation and Gradient Descent Kernels ---

// Compute softmax and loss gradient (softmax(logits) - one_hot(target)).
// logits: [seq_length x vocab_size], target: [seq_length]
__global__ void softmaxAndLossGradientKernel(const float* logits,
                                             const int* target,
                                             float* d_logits,
                                             int seq_length,
                                             int vocab_size);

// Compute gradient for output weight matrix:
// dW_out = X^T * d_logits, where X: [seq_length x embed_dim]
__global__ void linearOutputBackwardKernel(const float* X,
                                             const float* d_logits,
                                             float* dW_out,
                                             int seq_length,
                                             int embed_dim,
                                             int vocab_size);

// Compute gradient for output bias vector by summing over sequence dimension.
__global__ void linearBiasBackwardKernel(const float* d_logits,
                                           float* db_out,
                                           int seq_length,
                                           int vocab_size);

// Generic kernel for gradient descent update:
// param[i] -= learning_rate * grad[i]
__global__ void gradientDescentUpdateKernel(float* param,
                                            const float* grad,
                                            float learning_rate,
                                            int size);

#endif // KERNELS_H

