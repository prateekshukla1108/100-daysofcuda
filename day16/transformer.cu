// transformer.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if(err != cudaSuccess) {                                              \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,            \
                   cudaGetErrorString(err));                                  \
            exit(err);                                                        \
        }                                                                     \
    }

// ------------------------
// Dimension definitions
// ------------------------
#define BATCH       1
#define SEQ_LEN     128
#define D_MODEL     512
#define HEADS       8
#define D_HEAD      (D_MODEL / HEADS)
#define FFN_DIM     2048

// ------------------------
// Simple matrix multiplication kernel:
// Computes C = A (MxK) * B (KxN) --> C is MxN
// ------------------------
__global__ void matMulKernel(const float* A, const float* B, float* C,
                             int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++){
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ------------------------
// Variant: multiply A (MxK) with B^T (NxK) so that C = A * B^T (M x N)
// (B is stored in row–major order; we interpret B as transposed)
// ------------------------
__global__ void matMulKernelTransposed(const float* A, const float* B, float* C,
                                       int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++){
            // B is treated as if it were transposed: element at (col, i)
            sum += A[row * K + i] * B[col * K + i];
        }
        C[row * N + col] = sum;
    }
}

// ------------------------
// Scale kernel: multiplies every element by a scalar
// ------------------------
__global__ void scaleKernel(float* data, int size, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        data[idx] *= scale;
}

// ------------------------
// Softmax kernel: applies softmax to each row of a matrix.
// The matrix "data" has dimensions (rows x cols).
// ------------------------
__global__ void softmaxKernel(float* data, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float max_val = -1e20f;
        // find max in row
        for (int j = 0; j < cols; j++){
            float val = data[row * cols + j];
            if(val > max_val) max_val = val;
        }
        float sum = 0.0f;
        // compute exponentials and sum
        for (int j = 0; j < cols; j++){
            float exp_val = expf(data[row * cols + j] - max_val);
            data[row * cols + j] = exp_val;
            sum += exp_val;
        }
        // normalize
        for (int j = 0; j < cols; j++){
            data[row * cols + j] /= sum;
        }
    }
}

// ------------------------
// Element–wise add kernel: C = A + B (in place on A)
// ------------------------
__global__ void addKernel(float* A, const float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
        A[idx] += B[idx];
}

// ------------------------
// ReLU kernel: A = max(A, 0)
// ------------------------
__global__ void reluKernel(float* A, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
        A[idx] = fmaxf(0.0f, A[idx]);
}

// ------------------------
// Layer normalization kernel:
// For each row (of length "size") in input, compute mean, variance, and normalize.
// ------------------------
__global__ void layerNormKernel(const float* input, float* output, int size)
{
    // each block processes one row (e.g., one token vector of dimension D_MODEL)
    int row = blockIdx.x;
    float mean = 0.0f;
    for (int j = 0; j < size; j++){
        mean += input[row * size + j];
    }
    mean /= size;
    float var = 0.0f;
    for (int j = 0; j < size; j++){
        float diff = input[row * size + j] - mean;
        var += diff * diff;
    }
    var /= size;
    float inv_std = rsqrtf(var + 1e-5f);
    for (int j = 0; j < size; j++){
        output[row * size + j] = (input[row * size + j] - mean) * inv_std;
    }
}

// ------------------------
// Helper: fill an array with random floats (host)
// ------------------------
void fillRandom(float* data, int size)
{
    for (int i = 0; i < size; i++){
        data[i] = ((float) rand() / (float) RAND_MAX) - 0.5f;
    }
}

// ------------------------
// Main: a simplified transformer layer.
// The layer does:
//   1. Compute Q = X*Wq, K = X*Wk, V = X*Wv   (linear projections)
//   2. For each attention head, compute attention:
//        scores = (Q_head * K_head^T) / sqrt(d_head)
//        softmax(scores)
//        head_out = scores * V_head
//   3. Concatenate heads and compute Out = concat(head_out)*Wo
//   4. Add residual & layer-norm: Y1 = LN(X + Out)
//   5. Feedforward: FFN = ReLU(Y1*W1) * W2
//   6. Add residual & layer-norm: Output = LN(Y1 + FFN)
// ------------------------
int main()
{
    srand(0);
    // Dimensions (treat batch as 1, so work with matrices of size [SEQ_LEN x D_MODEL])
    const int inputSize = SEQ_LEN * D_MODEL;
    const int projSize  = D_MODEL * D_MODEL;
    const int ffnSize1  = D_MODEL * FFN_DIM;
    const int ffnSize2  = FFN_DIM * D_MODEL;

    // Allocate host memory for input and weights
    float *h_X     = (float*) malloc(inputSize * sizeof(float)); // input X
    float *h_Wq    = (float*) malloc(projSize * sizeof(float));
    float *h_Wk    = (float*) malloc(projSize * sizeof(float));
    float *h_Wv    = (float*) malloc(projSize * sizeof(float));
    float *h_Wo    = (float*) malloc(projSize * sizeof(float));
    float *h_W1    = (float*) malloc(ffnSize1 * sizeof(float));
    float *h_W2    = (float*) malloc(ffnSize2 * sizeof(float));

    // Initialize host memory randomly
    fillRandom(h_X, inputSize);
    fillRandom(h_Wq, projSize);
    fillRandom(h_Wk, projSize);
    fillRandom(h_Wv, projSize);
    fillRandom(h_Wo, projSize);
    fillRandom(h_W1, ffnSize1);
    fillRandom(h_W2, ffnSize2);

    // Allocate device memory
    float *d_X, *d_Wq, *d_Wk, *d_Wv, *d_Wo;
    float *d_Q, *d_K, *d_V; // projections; size: [SEQ_LEN x D_MODEL]
    float *d_OutAttn;      // concatenated attention output, [SEQ_LEN x D_MODEL]
    float *d_Y1;           // output after attention residual+LN, [SEQ_LEN x D_MODEL]
    float *d_FFN, *d_Y2;   // feed-forward output and final output
    CHECK_CUDA(cudaMalloc(&d_X,     inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wq,    projSize  * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wk,    projSize  * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wv,    projSize  * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wo,    projSize  * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Q,     inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K,     inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V,     inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_OutAttn, inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Y1,    inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_FFN,   inputSize * sizeof(float))); // same shape as X
    CHECK_CUDA(cudaMalloc(&d_Y2,    inputSize * sizeof(float)));

    float *d_W1, *d_W2;
    CHECK_CUDA(cudaMalloc(&d_W1, ffnSize1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W2, ffnSize2 * sizeof(float)));

    // Copy host memory to device
    CHECK_CUDA(cudaMemcpy(d_X, h_X, inputSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wq, h_Wq, projSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wk, h_Wk, projSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv, h_Wv, projSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo, h_Wo, projSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W1, h_W1, ffnSize1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W2, h_W2, ffnSize2 * sizeof(float), cudaMemcpyHostToDevice));

    // --- Kernel launch configuration ---
    dim3 blockDim(16, 16);
    // For matrix multiplication: grid dimensions depend on matrix sizes.
    // We'll use them as needed in each call.

    // ------------------------------
    // 1. Linear projections: Q = X*Wq, K = X*Wk, V = X*Wv
    // Each: [SEQ_LEN x D_MODEL] = [SEQ_LEN x D_MODEL] * [D_MODEL x D_MODEL]
    int M = SEQ_LEN, N = D_MODEL, K_dim = D_MODEL;
    dim3 gridDim((N + blockDim.x - 1)/blockDim.x, (M + blockDim.y - 1)/blockDim.y);
    matMulKernel<<<gridDim, blockDim>>>(d_X, d_Wq, d_Q, M, N, K_dim);
    matMulKernel<<<gridDim, blockDim>>>(d_X, d_Wk, d_K, M, N, K_dim);
    matMulKernel<<<gridDim, blockDim>>>(d_X, d_Wv, d_V, M, N, K_dim);
    CHECK_CUDA(cudaDeviceSynchronize());

    // ------------------------------
    // 2. Multi-head attention: for each head, compute:
    //    scores = (Q_head * K_head^T) / sqrt(D_HEAD), softmax, then head_out = scores * V_head.
    // d_Q, d_K, d_V are stored as [SEQ_LEN x D_MODEL]. For head h, offset = h*D_HEAD.
    // Allocate temporary buffer for scores per head: [SEQ_LEN x SEQ_LEN]
    float *d_scores, *d_headOut;
    CHECK_CUDA(cudaMalloc(&d_scores, SEQ_LEN * SEQ_LEN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_headOut, SEQ_LEN * D_HEAD * sizeof(float)));

    float scale = 1.0f / sqrtf((float)D_HEAD);

    // For each head (loop on host)
    for (int h = 0; h < HEADS; h++) {
        // Compute pointers offset into Q, K, V.
        // For simplicity, treat each as a 2D matrix with [SEQ_LEN x D_HEAD]
        const float* Q_h = d_Q + h * D_HEAD; // every row: Q[i][h*D_HEAD ... h*D_HEAD+D_HEAD-1]
        const float* K_h = d_K + h * D_HEAD;
        const float* V_h = d_V + h * D_HEAD;
        // We assume row–stride = D_MODEL.

        // Compute attention scores: scores = Q_h * (K_h)^T.
        // Here Q_h: [SEQ_LEN x D_HEAD] and K_h: [SEQ_LEN x D_HEAD] (but we need K_h transposed).
        // Use our matMulKernelTransposed with: M = SEQ_LEN, N = SEQ_LEN, K = D_HEAD.
        gridDim = dim3((SEQ_LEN + blockDim.x - 1)/blockDim.x, (SEQ_LEN + blockDim.y - 1)/blockDim.y);
        matMulKernelTransposed<<<gridDim, blockDim>>>(Q_h, K_h, d_scores, SEQ_LEN, SEQ_LEN, D_HEAD);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Scale the scores.
        int totalScores = SEQ_LEN * SEQ_LEN;
        int threads = 256;
        int blocks = (totalScores + threads - 1) / threads;
        scaleKernel<<<blocks, threads>>>(d_scores, totalScores, scale);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Apply softmax along rows of scores.
        blocks = (SEQ_LEN + 255)/256;
        softmaxKernel<<<blocks, 256>>>(d_scores, SEQ_LEN, SEQ_LEN);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Compute head output: head_out = scores * V_h.
        // scores: [SEQ_LEN x SEQ_LEN], V_h: [SEQ_LEN x D_HEAD], result: [SEQ_LEN x D_HEAD].
        gridDim = dim3((D_HEAD + blockDim.x - 1)/blockDim.x, (SEQ_LEN + blockDim.y - 1)/blockDim.y);
        matMulKernel<<<gridDim, blockDim>>>(d_scores, V_h, d_headOut, SEQ_LEN, D_HEAD, SEQ_LEN);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy head output into its position in d_OutAttn.
        // d_OutAttn is [SEQ_LEN x D_MODEL]; for head h, copy into columns [h*D_HEAD, (h+1)*D_HEAD)
        // We do this with a simple kernel launch.
        // For simplicity, we launch one thread per element in the head.
        int numElems = SEQ_LEN * D_HEAD;
        addKernel<<<(numElems+255)/256, 256>>>(d_OutAttn + h*D_HEAD, d_headOut, numElems);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    CHECK_CUDA(cudaFree(d_scores));
    CHECK_CUDA(cudaFree(d_headOut));

    // ------------------------------
    // 3. Final linear projection: Out = d_OutAttn * Wo, shape: [SEQ_LEN x D_MODEL]
    gridDim = dim3((D_MODEL + blockDim.x - 1)/blockDim.x, (SEQ_LEN + blockDim.y - 1)/blockDim.y);
    float* d_AttnProj;
    CHECK_CUDA(cudaMalloc(&d_AttnProj, inputSize * sizeof(float)));
    matMulKernel<<<gridDim, blockDim>>>(d_OutAttn, d_Wo, d_AttnProj, SEQ_LEN, D_MODEL, D_MODEL);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 4. Residual connection + LayerNorm: Y1 = LN(X + AttnProj)
    // First, add d_X and d_AttnProj elementwise.
    int total = inputSize;
    addKernel<<<(total+255)/256, 256>>>(d_AttnProj, d_X, total);
    CHECK_CUDA(cudaDeviceSynchronize());
    // Then layer normalization on each token (each row of length D_MODEL)
    layerNormKernel<<<SEQ_LEN, 1>>>(d_AttnProj, d_Y1, D_MODEL);
    CHECK_CUDA(cudaDeviceSynchronize());

    // ------------------------------
    // 5. Feed–Forward Network (FFN):
    // FFN = ReLU(Y1 * W1) * W2. (Shapes: [SEQ_LEN x FFN_DIM] then [SEQ_LEN x D_MODEL])
    float* d_FFN1;
    CHECK_CUDA(cudaMalloc(&d_FFN1, SEQ_LEN * FFN_DIM * sizeof(float)));
    // Y1: [SEQ_LEN x D_MODEL] * W1: [D_MODEL x FFN_DIM] = [SEQ_LEN x FFN_DIM]
    gridDim = dim3((FFN_DIM + blockDim.x - 1)/blockDim.x, (SEQ_LEN + blockDim.y - 1)/blockDim.y);
    matMulKernel<<<gridDim, blockDim>>>(d_Y1, d_W1, d_FFN1, SEQ_LEN, FFN_DIM, D_MODEL);
    CHECK_CUDA(cudaDeviceSynchronize());
    // Apply ReLU elementwise
    total = SEQ_LEN * FFN_DIM;
    reluKernel<<<(total+255)/256, 256>>>(d_FFN1, total);
    CHECK_CUDA(cudaDeviceSynchronize());
    // Second linear: [SEQ_LEN x FFN_DIM] * W2: [FFN_DIM x D_MODEL] = [SEQ_LEN x D_MODEL]
    matMulKernel<<<gridDim, blockDim>>>(d_FFN1, d_W2, d_FFN, SEQ_LEN, D_MODEL, FFN_DIM);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(d_FFN1));

    // 6. Add residual and layer-norm: Output = LN(Y1 + FFN)
    addKernel<<<(inputSize+255)/256, 256>>>(d_FFN, d_Y1, inputSize);
    CHECK_CUDA(cudaDeviceSynchronize());
    layerNormKernel<<<SEQ_LEN, 1>>>(d_FFN, d_Y2, D_MODEL);
    CHECK_CUDA(cudaDeviceSynchronize());

    // At this point, d_Y2 contains the output of the transformer layer.
    // (For demonstration, we copy a few values back to host.)
    float *h_output = (float*) malloc(inputSize * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_output, d_Y2, inputSize * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Transformer layer output (first 10 values):\n");
    for (int i = 0; i < 10; i++){
        printf("%f ", h_output[i]);
    }
    printf("\n");

    // Free all memory.
    free(h_X); free(h_Wq); free(h_Wk); free(h_Wv); free(h_Wo);
    free(h_W1); free(h_W2); free(h_output);
    CHECK_CUDA(cudaFree(d_X));  CHECK_CUDA(cudaFree(d_Wq)); CHECK_CUDA(cudaFree(d_Wk));
    CHECK_CUDA(cudaFree(d_Wv)); CHECK_CUDA(cudaFree(d_Wo));
    CHECK_CUDA(cudaFree(d_Q));  CHECK_CUDA(cudaFree(d_K));  CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_OutAttn)); CHECK_CUDA(cudaFree(d_AttnProj));
    CHECK_CUDA(cudaFree(d_Y1)); CHECK_CUDA(cudaFree(d_FFN)); CHECK_CUDA(cudaFree(d_Y2));
    CHECK_CUDA(cudaFree(d_W1)); CHECK_CUDA(cudaFree(d_W2));

    return 0;
}

