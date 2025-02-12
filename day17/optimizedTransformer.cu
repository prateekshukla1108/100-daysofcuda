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

#define BATCH       1
#define SEQ_LEN     128
#define D_MODEL     512
#define HEADS       8
#define D_HEAD      (D_MODEL / HEADS)
#define FFN_DIM     2048
#define TILE_WIDTH 16

__global__ void tiledMatMulKernel(const float* A, const float* B, float* C,
                                  int M, int N, int K, int ldb) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        if (row < M && t * TILE_WIDTH + threadIdx.x < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_WIDTH + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_WIDTH + threadIdx.y < K)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * ldb + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++){
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = sum;
}

__global__ void matMulKernelTransposed(const float* A, const float* B, float* C,
                                       int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++){
            sum += A[row * K + i] * B[col * K + i];
        }
        C[row * N + col] = sum;
    }
}

__global__ void scaleKernel(float* data, int size, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        data[idx] *= scale;
}

__global__ void optimizedSoftmaxKernel(float* data, int rows, int cols) {
    int row = blockIdx.x;
    extern __shared__ float sdata[];  // dynamic shared memory
    int tid = threadIdx.x;

    float val = (tid < cols) ? data[row * cols + tid] : -1e20f;
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < cols)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    float exp_val = (tid < cols) ? expf(data[row * cols + tid] - max_val) : 0.0f;
    sdata[tid] = (tid < cols) ? exp_val : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum = sdata[0];

    if (tid < cols)
        data[row * cols + tid] = exp_val / sum;
}


__global__ void addKernel(float* A, const float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
        A[idx] += B[idx];
}

__global__ void reluKernel(float* A, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
        A[idx] = fmaxf(0.0f, A[idx]);
}

__global__ void optimizedLayerNormKernel(const float* input, float* output, int size) {
    int row = blockIdx.x;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float val = (tid < size) ? input[row * size + tid] : 0.0f;
    sdata[tid] = (tid < size) ? val : 0.0f;
    __syncthreads();

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

void fillRandom(float* data, int size)
{
    for (int i = 0; i < size; i++){
        data[i] = ((float) rand() / (float) RAND_MAX) - 0.5f;
    }
}

int main()
{
    srand(0);
    const int inputSize = SEQ_LEN * D_MODEL;
    const int projSize  = D_MODEL * D_MODEL;
    const int ffnSize1  = D_MODEL * FFN_DIM;
    const int ffnSize2  = FFN_DIM * D_MODEL;

    float *h_X     = (float*) malloc(inputSize * sizeof(float)); // input X
    float *h_Wq    = (float*) malloc(projSize * sizeof(float));
    float *h_Wk    = (float*) malloc(projSize * sizeof(float));
    float *h_Wv    = (float*) malloc(projSize * sizeof(float));
    float *h_Wo    = (float*) malloc(projSize * sizeof(float));
    float *h_W1    = (float*) malloc(ffnSize1 * sizeof(float));
    float *h_W2    = (float*) malloc(ffnSize2 * sizeof(float));

    fillRandom(h_X, inputSize);
    fillRandom(h_Wq, projSize);
    fillRandom(h_Wk, projSize);
    fillRandom(h_Wv, projSize);
    fillRandom(h_Wo, projSize);
    fillRandom(h_W1, ffnSize1);
    fillRandom(h_W2, ffnSize2);

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

    CHECK_CUDA(cudaMemcpy(d_X, h_X, inputSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wq, h_Wq, projSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wk, h_Wk, projSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv, h_Wv, projSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo, h_Wo, projSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W1, h_W1, ffnSize1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W2, h_W2, ffnSize2 * sizeof(float), cudaMemcpyHostToDevice));

    printf("Input X (first 10 values):\n");
    for (int i = 0; i < 10; i++){
        printf("%f ", h_X[i]);
    }
    printf("\n");

    dim3 blockDim(16, 16);
    int M = SEQ_LEN, N = D_MODEL, K_dim = D_MODEL;
    dim3 gridDim((N + blockDim.x - 1)/blockDim.x, (M + blockDim.y - 1)/blockDim.y);
    tiledMatMulKernel<<<gridDim, blockDim>>>(d_X, d_Wq, d_Q, M, N, K_dim, K_dim);
    tiledMatMulKernel<<<gridDim, blockDim>>>(d_X, d_Wk, d_K, M, N, K_dim, K_dim);
    tiledMatMulKernel<<<gridDim, blockDim>>>(d_X, d_Wv, d_V, M, N, K_dim, K_dim);
    CHECK_CUDA(cudaDeviceSynchronize());

    float *h_Q = (float*) malloc(inputSize * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_Q, d_Q, inputSize * sizeof(float), cudaMemcpyDeviceToHost));
    printf("Q Projection (first 10 values):\n");
    for (int i = 0; i < 10; i++){
        printf("%f ", h_Q[i]);
    }
    printf("\n");
    free(h_Q);

    float *h_K = (float*) malloc(inputSize * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_K, d_K, inputSize * sizeof(float), cudaMemcpyDeviceToHost));
    printf("K Projection (first 10 values):\n");
    for (int i = 0; i < 10; i++){
        printf("%f ", h_K[i]);
    }
    printf("\n");
    free(h_K);

    float *h_V = (float*) malloc(inputSize * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_V, d_V, inputSize * sizeof(float), cudaMemcpyDeviceToHost));
    printf("V Projection (first 10 values):\n");
    for (int i = 0; i < 10; i++){
        printf("%f ", h_V[i]);
    }
    printf("\n");
    free(h_V);


    float *d_scores, *d_headOut;
    CHECK_CUDA(cudaMalloc(&d_scores, SEQ_LEN * SEQ_LEN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_headOut, SEQ_LEN * D_HEAD * sizeof(float)));

    float scale = 1.0f / sqrtf((float)D_HEAD);

    for (int h = 0; h < HEADS; h++) {
        const float* Q_h = d_Q + h * D_HEAD; // every row: Q[i][h*D_HEAD ... h*D_HEAD+D_HEAD-1]
        const float* K_h = d_K + h * D_HEAD;
        const float* V_h = d_V + h * D_HEAD;

        gridDim = dim3((SEQ_LEN + blockDim.x - 1)/blockDim.x, (SEQ_LEN + blockDim.y - 1)/blockDim.y);
        matMulKernelTransposed<<<gridDim, blockDim>>>(Q_h, K_h, d_scores, SEQ_LEN, SEQ_LEN, D_HEAD);
        CHECK_CUDA(cudaDeviceSynchronize());

        float *h_scores_before_scale = (float*) malloc(SEQ_LEN * SEQ_LEN * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_scores_before_scale, d_scores, SEQ_LEN * SEQ_LEN * sizeof(float), cudaMemcpyDeviceToHost));
        printf("Attention Scores before Scale (Head %d, first 10 values):\n", h);
        for (int i = 0; i < 10; i++){
            printf("%f ", h_scores_before_scale[i]);
        }
        printf("\n");
        free(h_scores_before_scale);


        int totalScores = SEQ_LEN * SEQ_LEN;
        int threads = 256;
        int blocks = (totalScores + threads - 1) / threads;
        scaleKernel<<<blocks, threads>>>(d_scores, totalScores, scale);
        CHECK_CUDA(cudaDeviceSynchronize());

        float *h_scores_scaled = (float*) malloc(SEQ_LEN * SEQ_LEN * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_scores_scaled, d_scores, SEQ_LEN * SEQ_LEN * sizeof(float), cudaMemcpyDeviceToHost));
        printf("Attention Scores after Scale (Head %d, first 10 values):\n", h);
        for (int i = 0; i < 10; i++){
            printf("%f ", h_scores_scaled[i]);
        }
        printf("\n");
        free(h_scores_scaled);


        optimizedSoftmaxKernel<<<SEQ_LEN, 256, 256 * sizeof(float)>>>(d_scores, SEQ_LEN, SEQ_LEN);
        CHECK_CUDA(cudaDeviceSynchronize());

        float *h_scores_softmax = (float*) malloc(SEQ_LEN * SEQ_LEN * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_scores_softmax, d_scores, SEQ_LEN * SEQ_LEN * sizeof(float), cudaMemcpyDeviceToHost));
        printf("Attention Scores after Softmax (Head %d, first 10 values):\n", h);
        for (int i = 0; i < 10; i++){
            printf("%f ", h_scores_softmax[i]);
        }
        printf("\n");
        free(h_scores_softmax);


dim3 gridDim((D_HEAD + TILE_WIDTH - 1) / TILE_WIDTH,
             (SEQ_LEN + TILE_WIDTH - 1) / TILE_WIDTH);
dim3 blockDim(TILE_WIDTH, TILE_WIDTH);

tiledMatMulKernel<<<gridDim, blockDim>>>(d_scores, V_h, d_headOut,
                                          SEQ_LEN,   // M: number of rows in d_scores (and C)
                                          D_HEAD,    // N: number of columns in V_h (logical)
                                          SEQ_LEN,   // K: inner dimension of d_scores and V_h
                                          D_MODEL);  // ldb: actual row stride of V_h
CHECK_CUDA(cudaDeviceSynchronize());

        float *h_headOut = (float*) malloc(SEQ_LEN * D_HEAD * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_headOut, d_headOut, SEQ_LEN * D_HEAD * sizeof(float), cudaMemcpyDeviceToHost));
        printf("Head Output (Head %d, first 10 values):\n", h);
        for (int i = 0; i < 10; i++){
            printf("%f ", h_headOut[i]);
        }
        printf("\n");
        free(h_headOut);


        int numElems = SEQ_LEN * D_HEAD;
        addKernel<<<(numElems+255)/256, 256>>>(d_OutAttn + h*D_HEAD, d_headOut, numElems);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    CHECK_CUDA(cudaFree(d_scores));
    CHECK_CUDA(cudaFree(d_headOut));

    float *h_OutAttn = (float*) malloc(inputSize * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_OutAttn, d_OutAttn, inputSize * sizeof(float), cudaMemcpyDeviceToHost));
    printf("Concatenated Attention Output (first 10 values):\n");
    for (int i = 0; i < 10; i++){
        printf("%f ", h_OutAttn[i]);
    }
    printf("\n");
    free(h_OutAttn);


    gridDim = dim3((D_MODEL + blockDim.x - 1)/blockDim.x, (SEQ_LEN + blockDim.y - 1)/blockDim.y);
    float* d_AttnProj;
    CHECK_CUDA(cudaMalloc(&d_AttnProj, inputSize * sizeof(float)));
    tiledMatMulKernel<<<gridDim, blockDim>>>(d_OutAttn, d_Wo, d_AttnProj, 128, 512, 512, 512);
    CHECK_CUDA(cudaDeviceSynchronize());

    float *h_AttnProj = (float*) malloc(inputSize * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_AttnProj, d_AttnProj, inputSize * sizeof(float), cudaMemcpyDeviceToHost));
    printf("Attention Projection Output (first 10 values):\n");
    for (int i = 0; i < 10; i++){
        printf("%f ", h_AttnProj[i]);
    }
    printf("\n");
    free(h_AttnProj);


    int total = inputSize;
    addKernel<<<(total+255)/256, 256>>>(d_AttnProj, d_X, total);
    CHECK_CUDA(cudaDeviceSynchronize());
    optimizedLayerNormKernel<<<SEQ_LEN, D_MODEL, D_MODEL * sizeof(float)>>>(d_AttnProj, d_Y1, D_MODEL);
    CHECK_CUDA(cudaDeviceSynchronize());

    float *h_Y1 = (float*) malloc(inputSize * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_Y1, d_Y1, inputSize * sizeof(float), cudaMemcpyDeviceToHost));
    printf("LayerNorm Output after Attention (Y1, first 10 values):\n");
    for (int i = 0; i < 10; i++){
        printf("%f ", h_Y1[i]);
    }
    printf("\n");
    free(h_Y1);


    float* d_FFN1;
    CHECK_CUDA(cudaMalloc(&d_FFN1, SEQ_LEN * FFN_DIM * sizeof(float)));
    gridDim = dim3((FFN_DIM + blockDim.x - 1)/blockDim.x, (SEQ_LEN + blockDim.y - 1)/blockDim.y);
    // Corrected: ldb should be 2048 because d_W1 is 512 x 2048
    tiledMatMulKernel<<<gridDim, blockDim>>>(d_Y1, d_W1, d_FFN1, 128, 2048, 512, 2048);
    CHECK_CUDA(cudaDeviceSynchronize());

    float *h_FFN1 = (float*) malloc(SEQ_LEN * FFN_DIM * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_FFN1, d_FFN1, SEQ_LEN * FFN_DIM * sizeof(float), cudaMemcpyDeviceToHost));
    printf("FFN Layer 1 Output (before ReLU, first 10 values):\n");
    for (int i = 0; i < 10; i++){
        printf("%f ", h_FFN1[i]);
    }
    printf("\n");
    free(h_FFN1);


    total = SEQ_LEN * FFN_DIM;
    reluKernel<<<(total+255)/256, 256>>>(d_FFN1, total);
    CHECK_CUDA(cudaDeviceSynchronize());

    float *h_FFN1_relu = (float*) malloc(SEQ_LEN * FFN_DIM * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_FFN1_relu, d_FFN1, SEQ_LEN * FFN_DIM * sizeof(float), cudaMemcpyDeviceToHost));
    printf("FFN Layer 1 Output (after ReLU, first 10 values):\n");
    for (int i = 0; i < 10; i++){
        printf("%f ", h_FFN1_relu[i]);
    }
    printf("\n");
    free(h_FFN1_relu);


    // Corrected: ldb should be 512 because d_W2 is 2048 x 512
    tiledMatMulKernel<<<gridDim, blockDim>>>(d_FFN1, d_W2, d_FFN, 128, 512, 2048, 512);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(d_FFN1));

    float *h_FFN = (float*) malloc(inputSize * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_FFN, d_FFN, inputSize * sizeof(float), cudaMemcpyDeviceToHost));
    printf("FFN Layer 2 Output (first 10 values):\n");
    for (int i = 0; i < 10; i++){
        printf("%f ", h_FFN[i]);
    }
    printf("\n");
    free(h_FFN);


    addKernel<<<(inputSize+255)/256, 256>>>(d_FFN, d_Y1, inputSize);
    CHECK_CUDA(cudaDeviceSynchronize());
    optimizedLayerNormKernel<<<SEQ_LEN, D_MODEL, D_MODEL * sizeof(float)>>>(d_FFN, d_Y2, D_MODEL);

    CHECK_CUDA(cudaDeviceSynchronize());

    float *h_output = (float*) malloc(inputSize * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_output, d_Y2, inputSize * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Transformer layer output (Y2, first 10 values):\n");
    for (int i = 0; i < 10; i++){
        printf("%f ", h_output[i]);
    }
    printf("\n");

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
