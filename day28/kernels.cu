#include "kernels.h"
#include "transformer_model.h" // For EMBED_DIM, etc.

__global__ void fusedGemmReLUKernel(const float* A, const float* B, const float* bias, float* C,
                                    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        sum += bias[col];
        C[row * N + col] = (sum > 0.0f) ? sum : 0.0f;
    }
}

__global__ void matmulKernel(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void addBiasAndResidualKernel(float* attn, float* temp, const float* b2, int seq, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq * dim) {
        int col = idx % dim;
        temp[idx] = attn[idx] + temp[idx] + b2[col];
    }
}

__global__ void embeddingLookupKernel(const int* d_input, const float* d_embedding,
                                       float* d_output, int seq_length) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx < seq_length) {
        int token = d_input[token_idx];
        const float* src = d_embedding + token * EMBED_DIM;
        float* dst = d_output + token_idx * EMBED_DIM;
        for (int i = 0; i < EMBED_DIM; i += blockDim.y) {
            int idx = i + threadIdx.y;
            if (idx < EMBED_DIM) {
                dst[idx] = src[idx];
            }
        }
    }
}

__global__ void addPositionalEncoding(float* d_embedded, int seq_length, int embed_dim) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = threadIdx.y;
    if (pos < seq_length && dim < embed_dim) {
        float angle = pos / powf(10000.0f, 2.0f * dim / embed_dim);
        d_embedded[pos * embed_dim + dim] += (dim % 2 == 0) ? sinf(angle) : cosf(angle);
    }
}

__global__ void dummySelfAttentionKernel(const float* d_input, float* d_output, int seq_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_length * EMBED_DIM) {
        d_output[idx] = d_input[idx];
    }
}

__global__ void linearOutputKernel(const float* d_input, const float* d_W_out, const float* d_b_out,
                                   float* d_logits, int seq_length) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // sequence index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // vocabulary index
    if (row < seq_length && col < VOCAB_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < EMBED_DIM; i++) {
            sum += d_input[row * EMBED_DIM + i] * d_W_out[i * VOCAB_SIZE + col];
        }
        d_logits[row * VOCAB_SIZE + col] = sum + d_b_out[col];
    }
}

// --- Backpropagation Kernels ---

__global__ void softmaxAndLossGradientKernel(const float* logits,
                                             const int* target,
                                             float* d_logits,
                                             int seq_length,
                                             int vocab_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < seq_length) {
        // Find maximum for numerical stability.
        float max_logit = logits[row * vocab_size];
        for (int j = 1; j < vocab_size; j++) {
            float l = logits[row * vocab_size + j];
            if (l > max_logit)
                max_logit = l;
        }
        // Compute sum of exponentials.
        float sum_exp = 0.0f;
        for (int j = 0; j < vocab_size; j++) {
            sum_exp += expf(logits[row * vocab_size + j] - max_logit);
        }
        // Compute softmax and subtract one-hot target.
        for (int j = 0; j < vocab_size; j++) {
            float softmax_val = expf(logits[row * vocab_size + j] - max_logit) / sum_exp;
            int one_hot = (target[row] == j) ? 1 : 0;
            d_logits[row * vocab_size + j] = softmax_val - one_hot;
        }
    }
}

__global__ void linearOutputBackwardKernel(const float* X,
                                             const float* d_logits,
                                             float* dW_out,
                                             int seq_length,
                                             int embed_dim,
                                             int vocab_size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // over embed_dim
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // over vocab_size
    if (i < embed_dim && j < vocab_size) {
        float grad = 0.0f;
        for (int k = 0; k < seq_length; k++) {
            grad += X[k * embed_dim + i] * d_logits[k * vocab_size + j];
        }
        dW_out[i * vocab_size + j] = grad;
    }
}

__global__ void linearBiasBackwardKernel(const float* d_logits,
                                           float* db_out,
                                           int seq_length,
                                           int vocab_size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // over vocab_size
    if (j < vocab_size) {
        float grad = 0.0f;
        for (int i = 0; i < seq_length; i++) {
            grad += d_logits[i * vocab_size + j];
        }
        db_out[j] = grad;
    }
}

__global__ void gradientDescentUpdateKernel(float* param,
                                            const float* grad,
                                            float learning_rate,
                                            int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        param[idx] -= learning_rate * grad[idx];
    }
}

