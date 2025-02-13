#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                                                               \
    {                                                                                                                  \
        cudaError_t err = call;                                                                                        \
        if(err != cudaSuccess) {                                                                                       \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                         \
            exit(err);                                                                                                 \
        }                                                                                                              \
    }

#define BATCH      1
#define SEQ_LEN    128
#define D_MODEL    512
#define HEADS      8
#define D_HEAD     (D_MODEL / HEADS)
#define FFN_DIM    2048
#define TILE_WIDTH 32

// -----------------------------------------------------------------------------
// Fused Projection Kernel: Computes Q, K, and V from input X and weight matrices
// -----------------------------------------------------------------------------
__global__ void fusedProjectionKernel(const float* __restrict__ X, 
                                        const float* __restrict__ Wq,
                                        const float* __restrict__ Wk,
                                        const float* __restrict__ Wv,
                                        float* __restrict__ Q,
                                        float* __restrict__ K,
                                        float* __restrict__ V,
                                        int M, int N, int K_dim) {
    __shared__ float tileX[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileWq[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileWk[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileWv[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float sum_q = 0.0f, sum_k = 0.0f, sum_v = 0.0f;

    for (int t = 0; t < (K_dim + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        int tiledCol = t * TILE_WIDTH + threadIdx.x;
        // Load tile of X.
        if (row < M && tiledCol < K_dim)
            tileX[threadIdx.y][threadIdx.x] = X[row * K_dim + tiledCol];
        else
            tileX[threadIdx.y][threadIdx.x] = 0.0f;

        int tiledRow = t * TILE_WIDTH + threadIdx.y;
        // Load corresponding tiles from each weight matrix.
        if (tiledRow < K_dim && col < N) {
            tileWq[threadIdx.y][threadIdx.x] = Wq[tiledRow * N + col];
            tileWk[threadIdx.y][threadIdx.x] = Wk[tiledRow * N + col];
            tileWv[threadIdx.y][threadIdx.x] = Wv[tiledRow * N + col];
        } else {
            tileWq[threadIdx.y][threadIdx.x] = 0.0f;
            tileWk[threadIdx.y][threadIdx.x] = 0.0f;
            tileWv[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            float a = tileX[threadIdx.y][k];
            sum_q += a * tileWq[k][threadIdx.x];
            sum_k += a * tileWk[k][threadIdx.x];
            sum_v += a * tileWv[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        Q[row * N + col] = sum_q;
        K[row * N + col] = sum_k;
        V[row * N + col] = sum_v;
    }
}

// -----------------------------------------------------------------------------
// Optimized Tiled Matrix Multiplication Kernel (unchanged)
// -----------------------------------------------------------------------------
__global__ void optimizedTiledMatMulKernelV2(const float* __restrict__ A, const float* __restrict__ B,
                                             float* __restrict__ C, int M, int N, int K, int ldb) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (row < M && (t * TILE_WIDTH + threadIdx.x) < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_WIDTH + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (t * TILE_WIDTH + threadIdx.y) < K)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * ldb + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// -----------------------------------------------------------------------------
// MatMul with transposed second matrix (unchanged)
// -----------------------------------------------------------------------------
__global__ void matMulKernelTransposedV2(const float* __restrict__ A, const float* __restrict__ B,
                                         float* __restrict__ C, int M, int N, int K) {
    const int block_size = 32;
    __shared__ float tileA[block_size][block_size];
    __shared__ float tileB[block_size][block_size];

    int row = blockIdx.y * block_size + threadIdx.y;
    int col = blockIdx.x * block_size + threadIdx.x;
    float sum = 0.0f;

    for (int tile_k = 0; tile_k < (K + block_size - 1) / block_size; ++tile_k) {
        if (row < M && (tile_k * block_size + threadIdx.x) < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + tile_k * block_size + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (tile_k * block_size + threadIdx.y) < K)
            tileB[threadIdx.y][threadIdx.x] = B[col * K + tile_k * block_size + threadIdx.y];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < block_size; ++k)
            sum += tileA[threadIdx.y][k] * tileB[threadIdx.x][k];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// -----------------------------------------------------------------------------
// Optimized fused scale and softmax kernel (unchanged)
// -----------------------------------------------------------------------------
__global__ void optimizedFusedScaleSoftmaxKernelV2(float* data, int rows, int cols, float scale) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (tid >= cols) return;

    float x = data[row * cols + tid] * scale;
    unsigned mask = 0xffffffff;
    float max_val = x;
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        max_val = fmaxf(max_val, __shfl_down_sync(mask, max_val, offset));

    __shared__ float s_max[32];
    int lane = tid % warpSize;
    int warpId = tid / warpSize;
    if (lane == 0)
        s_max[warpId] = max_val;
    __syncthreads();
    if (tid < warpSize) {
        float tmp = s_max[tid];
        int numWarps = (cols + warpSize - 1) / warpSize;
        #pragma unroll
        for (int i = tid + warpSize; i < numWarps; i++)
            tmp = fmaxf(tmp, s_max[i]);
        s_max[tid] = tmp;
    }
    __syncthreads();
    max_val = s_max[0];

    float exp_val = expf(x - max_val);
    float sum_val = exp_val;
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum_val += __shfl_down_sync(mask, sum_val, offset);

    __shared__ float s_sum[32];
    if (lane == 0)
        s_sum[warpId] = sum_val;
    __syncthreads();
    if (tid < warpSize) {
        float tmp = s_sum[tid];
        int numWarps = (cols + warpSize - 1) / warpSize;
        #pragma unroll
        for (int i = tid + warpSize; i < numWarps; i++)
            tmp += s_sum[i];
        s_sum[tid] = tmp;
    }
    __syncthreads();
    float total_sum = s_sum[0];

    data[row * cols + tid] = exp_val / total_sum;
}

// -----------------------------------------------------------------------------
// ReLU Kernel (unchanged)
// -----------------------------------------------------------------------------
__global__ void reluKernelV2(float* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        A[idx] = fmaxf(0.0f, A[idx]);
}

// -----------------------------------------------------------------------------
// Fused Add & LayerNorm Kernel: Reads A and B, adds them elementwise, then normalizes.
// -----------------------------------------------------------------------------
__global__ void fusedAddLayerNormKernel(const float* __restrict__ A, 
                                        const float* __restrict__ B, 
                                        float* __restrict__ out, 
                                        int feature_size) {
    int row = blockIdx.x;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float val = (tid < feature_size) ? (A[row * feature_size + tid] + B[row * feature_size + tid]) : 0.0f;
    sdata[tid] = val;
    __syncthreads();

    // Compute mean.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < feature_size)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / feature_size;
    __syncthreads();

    float diff = val - mean;
    sdata[tid] = (tid < feature_size) ? diff * diff : 0.0f;
    __syncthreads();
    // Compute variance.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < feature_size)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float variance = sdata[0] / feature_size;
    float inv_std = rsqrtf(variance + 1e-5f);
    if (tid < feature_size)
        out[row * feature_size + tid] = (val - mean) * inv_std;
}

// -----------------------------------------------------------------------------
// (Optional) Standalone LayerNorm Kernel (unchanged)
// -----------------------------------------------------------------------------
__global__ void optimizedLayerNormKernelV2(const float* input, float* output, int size) {
    int row = blockIdx.x;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float val = (tid < size) ? input[row * size + tid] : 0.0f;
    sdata[tid] = (tid < size) ? val : 0.0f;
    __syncthreads();

    #pragma unroll
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < size)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / size;
    __syncthreads();

    float diff = (tid < size) ? (input[row * size + tid] - mean) : 0.0f;
    sdata[tid] = (tid < size) ? diff * diff : 0.0f;
    __syncthreads();

    #pragma unroll
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < size)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float variance = sdata[0] / size;
    float inv_std = rsqrtf(variance + 1e-5f);

    if (tid < size)
        output[row * size + tid] = (input[row * size + tid] - mean) * inv_std;
}

// -----------------------------------------------------------------------------
// Host helper to fill an array with random numbers
// -----------------------------------------------------------------------------
void fillRandomV2(float* data, int size) {
    for (int i = 0; i < size; i++){
        data[i] = ((float) rand() / (float) RAND_MAX) - 0.5f;
    }
}

// -----------------------------------------------------------------------------
// Main Function
// -----------------------------------------------------------------------------
int main() {
    srand(0);
    const int inputSize = SEQ_LEN * D_MODEL;
    const int projSize  = D_MODEL * D_MODEL;
    const int ffnSize1  = D_MODEL * FFN_DIM;
    const int ffnSize2  = FFN_DIM * D_MODEL;

    float *h_X  = (float*) malloc(inputSize * sizeof(float)); // input X
    float *h_Wq = (float*) malloc(projSize  * sizeof(float));
    float *h_Wk = (float*) malloc(projSize  * sizeof(float));
    float *h_Wv = (float*) malloc(projSize  * sizeof(float));
    float *h_Wo = (float*) malloc(projSize  * sizeof(float));
    float *h_W1 = (float*) malloc(ffnSize1  * sizeof(float));
    float *h_W2 = (float*) malloc(ffnSize2  * sizeof(float));

    fillRandomV2(h_X, inputSize);
    fillRandomV2(h_Wq, projSize);
    fillRandomV2(h_Wk, projSize);
    fillRandomV2(h_Wv, projSize);
    fillRandomV2(h_Wo, projSize);
    fillRandomV2(h_W1, ffnSize1);
    fillRandomV2(h_W2, ffnSize2);

    float *d_X, *d_Wq, *d_Wk, *d_Wv, *d_Wo;
    float *d_Q, *d_K, *d_V;
    float *d_OutAttn;
    float *d_Y1;
    float *d_FFN, *d_Y2;

    CHECK_CUDA(cudaMalloc(&d_X,      inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wq,     projSize  * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wk,     projSize  * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wv,     projSize  * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wo,     projSize  * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Q,      inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K,      inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V,      inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_OutAttn,inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Y1,     inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_FFN,    inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Y2,     inputSize * sizeof(float)));

    float *d_W1, *d_W2;
    CHECK_CUDA(cudaMalloc(&d_W1, ffnSize1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W2, ffnSize2 * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_X,  h_X,  inputSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wq, h_Wq, projSize  * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wk, h_Wk, projSize  * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv, h_Wv, projSize  * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo, h_Wo, projSize  * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W1, h_W1, ffnSize1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W2, h_W2, ffnSize2 * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    // -------------------------------------------------------------------------
    // 1. Fused Projections for Q, K, and V
    // -------------------------------------------------------------------------
    dim3 blockDim_proj(TILE_WIDTH, TILE_WIDTH);
    int M = SEQ_LEN, N = D_MODEL, K_dim = D_MODEL;
    dim3 gridDim_proj((N + blockDim_proj.x - 1) / blockDim_proj.x,
                      (M + blockDim_proj.y - 1) / blockDim_proj.y);
    fusedProjectionKernel<<<gridDim_proj, blockDim_proj>>>(d_X, d_Wq, d_Wk, d_Wv, d_Q, d_K, d_V, M, N, K_dim);

    // -------------------------------------------------------------------------
    // 2. Attention per Head
    // -------------------------------------------------------------------------
    float *d_scores, *d_headOut;
    CHECK_CUDA(cudaMalloc(&d_scores, SEQ_LEN * SEQ_LEN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_headOut, SEQ_LEN * D_HEAD * sizeof(float)));
    float scale = 1.0f / sqrtf((float)D_HEAD);

    for (int h = 0; h < HEADS; h++) {
        const float* Q_h = d_Q + h * D_HEAD;
        const float* K_h = d_K + h * D_HEAD;
        const float* V_h = d_V + h * D_HEAD;

        dim3 blockDim_transposed(32, 32);
        dim3 gridDim_transposed((D_HEAD + blockDim_transposed.x - 1) / blockDim_transposed.x,
                                (SEQ_LEN + blockDim_transposed.y - 1) / blockDim_transposed.y);
        // Compute attention scores (using transposed matmul for K)
        matMulKernelTransposedV2<<<gridDim_transposed, blockDim_transposed>>>(Q_h, K_h, d_scores, SEQ_LEN, SEQ_LEN, D_HEAD);

        // Softmax (scaling is fused in the kernel)
        optimizedFusedScaleSoftmaxKernelV2<<<SEQ_LEN, SEQ_LEN>>>(d_scores, SEQ_LEN, SEQ_LEN, scale);

        // Multiply scores with V
        dim3 blockDimTile(TILE_WIDTH, TILE_WIDTH);
        dim3 gridDimTile((D_HEAD + TILE_WIDTH - 1) / TILE_WIDTH,
                         (SEQ_LEN + TILE_WIDTH - 1) / TILE_WIDTH);
        optimizedTiledMatMulKernelV2<<<gridDimTile, blockDimTile>>>(d_scores, V_h, d_headOut, SEQ_LEN, D_HEAD, SEQ_LEN, D_MODEL);

        // Since each head writes into a separate slice of d_OutAttn, copy directly.
        int numElems = SEQ_LEN * D_HEAD;
        CHECK_CUDA(cudaMemcpy(d_OutAttn + h * D_HEAD, d_headOut, numElems * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    CHECK_CUDA(cudaFree(d_scores));
    CHECK_CUDA(cudaFree(d_headOut));

    // -------------------------------------------------------------------------
    // 3. Attention Projection
    // -------------------------------------------------------------------------
    float* d_AttnProj;
    CHECK_CUDA(cudaMalloc(&d_AttnProj, inputSize * sizeof(float)));
    dim3 gridDim_attn_proj((D_MODEL + blockDim_proj.x - 1) / blockDim_proj.x,
                           (SEQ_LEN + blockDim_proj.y - 1) / blockDim_proj.y);
    optimizedTiledMatMulKernelV2<<<gridDim_attn_proj, blockDim_proj>>>(d_OutAttn, d_Wo, d_AttnProj, SEQ_LEN, D_MODEL, D_MODEL, D_MODEL);

    // -------------------------------------------------------------------------
    // 4. Fused Add & LayerNorm after Attention (Residual Connection)
    // -------------------------------------------------------------------------
    fusedAddLayerNormKernel<<<SEQ_LEN, D_MODEL, D_MODEL * sizeof(float)>>>(d_AttnProj, d_X, d_Y1, D_MODEL);

    // -------------------------------------------------------------------------
    // 5. Feed-Forward Network (FFN)
    // -------------------------------------------------------------------------
    float* d_FFN1;
    CHECK_CUDA(cudaMalloc(&d_FFN1, SEQ_LEN * FFN_DIM * sizeof(float)));
    dim3 gridDim_ffn1((FFN_DIM + blockDim_proj.x - 1) / blockDim_proj.x,
                      (SEQ_LEN + blockDim_proj.y - 1) / blockDim_proj.y);
    optimizedTiledMatMulKernelV2<<<gridDim_ffn1, blockDim_proj>>>(d_Y1, d_W1, d_FFN1, SEQ_LEN, FFN_DIM, D_MODEL, FFN_DIM);

    int total = SEQ_LEN * FFN_DIM;
    reluKernelV2<<<(total + 255) / 256, 256>>>(d_FFN1, total);
    optimizedTiledMatMulKernelV2<<<gridDim_proj, blockDim_proj>>>(d_FFN1, d_W2, d_FFN, SEQ_LEN, D_MODEL, FFN_DIM, D_MODEL);
    CHECK_CUDA(cudaFree(d_FFN1));

    // -------------------------------------------------------------------------
    // 6. Fused Add & LayerNorm after FFN (Residual Connection)
    // -------------------------------------------------------------------------
    fusedAddLayerNormKernel<<<SEQ_LEN, D_MODEL, D_MODEL * sizeof(float)>>>(d_FFN, d_Y1, d_Y2, D_MODEL);

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsed_ms;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    printf("Optimized transformer layer kernels V2 execution time: %.3f ms\n", elapsed_ms);

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    free(h_X); free(h_Wq); free(h_Wk); free(h_Wv); free(h_Wo);
    free(h_W1); free(h_W2);
    CHECK_CUDA(cudaFree(d_X));  CHECK_CUDA(cudaFree(d_Wq));  CHECK_CUDA(cudaFree(d_Wk));
    CHECK_CUDA(cudaFree(d_Wv)); CHECK_CUDA(cudaFree(d_Wo));
    CHECK_CUDA(cudaFree(d_Q));  CHECK_CUDA(cudaFree(d_K));   CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_OutAttn));
    CHECK_CUDA(cudaFree(d_AttnProj));
    CHECK_CUDA(cudaFree(d_Y1)); CHECK_CUDA(cudaFree(d_FFN));  CHECK_CUDA(cudaFree(d_Y2));
    CHECK_CUDA(cudaFree(d_W1)); CHECK_CUDA(cudaFree(d_W2));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}

