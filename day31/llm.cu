#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdlib>
#include <random>
#include <cmath>
#include <algorithm>

#define CONTEXT_LENGTH 32
#define EMBED_DIM 128
#define HIDDEN_DIM1 256
#define HIDDEN_DIM2 192
#define HIDDEN_DIM3 224
#define HIDDEN_DIM4 160
#define HIDDEN_DIM5 256
#define MAX_VOCAB 1024
#define LEARNING_RATE 0.01f
#define BATCH_SIZE 16
#define EPOCHS 10

#define CHECK_CUDA(call) do { cudaError_t err = call; if(err != cudaSuccess){ std::cerr << "CUDA error " << cudaGetErrorString(err) << std::endl; exit(EXIT_FAILURE); } } while(0)

__global__ void embeddingLookupKernel(const float* d_embedding, const int* d_context, float* d_input) {
    int tid = threadIdx.x;
    if(tid < EMBED_DIM) {
        float sum = 0.0f;
        for (int i = 0; i < CONTEXT_LENGTH; i++) {
            int token = d_context[i];
            sum += d_embedding[token * EMBED_DIM + tid];
        }
        d_input[tid] = sum / (CONTEXT_LENGTH == 0 ? 1.0f : (float)CONTEXT_LENGTH);
    }
}

__global__ void fusedForwardLossKernel(const float* d_input,
                                         const float* d_W1, const float* d_b1,
                                         const float* d_W2, const float* d_b2,
                                         const float* d_W3, const float* d_b3,
                                         const float* d_W4, const float* d_b4,
                                         const float* d_W5, const float* d_b5,
                                         const float* d_Wout, const float* d_bout,
                                         float* d_hidden1, float* d_hidden2, float* d_hidden3, float* d_hidden4, float* d_hidden5,
                                         float* d_logits, float* d_probs, float* d_loss,
                                         int vocabSize, int targetIdx) {
    extern __shared__ float shared[];
    float* s_hidden1 = shared;
    float* s_hidden2 = s_hidden1 + HIDDEN_DIM1;
    float* s_hidden3 = s_hidden2 + HIDDEN_DIM2;
    float* s_hidden4 = s_hidden3 + HIDDEN_DIM3;
    float* s_hidden5 = s_hidden4 + HIDDEN_DIM4;
    float* sdata = s_hidden5 + HIDDEN_DIM5;
    int tid = threadIdx.x;
    for (int h = tid; h < HIDDEN_DIM1; h += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < EMBED_DIM; i++) {
            sum += d_input[i] * d_W1[i * HIDDEN_DIM1 + h];
        }
        sum += d_b1[h];
        s_hidden1[h] = (sum > 0.0f) ? sum : 0.0f;
        d_hidden1[h] = s_hidden1[h];
    }
    __syncthreads();
    for (int h = tid; h < HIDDEN_DIM2; h += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < HIDDEN_DIM1; i++) {
            sum += s_hidden1[i] * d_W2[i * HIDDEN_DIM2 + h];
        }
        sum += d_b2[h];
        s_hidden2[h] = (sum > 0.0f) ? sum : 0.0f;
        d_hidden2[h] = s_hidden2[h];
    }
    __syncthreads();
    for (int h = tid; h < HIDDEN_DIM3; h += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < HIDDEN_DIM2; i++) {
            sum += s_hidden2[i] * d_W3[i * HIDDEN_DIM3 + h];
        }
        sum += d_b3[h];
        s_hidden3[h] = (sum > 0.0f) ? sum : 0.0f;
        d_hidden3[h] = s_hidden3[h];
    }
    __syncthreads();
    for (int h = tid; h < HIDDEN_DIM4; h += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < HIDDEN_DIM3; i++) {
            sum += s_hidden3[i] * d_W4[i * HIDDEN_DIM4 + h];
        }
        sum += d_b4[h];
        s_hidden4[h] = (sum > 0.0f) ? sum : 0.0f;
        d_hidden4[h] = s_hidden4[h];
    }
    __syncthreads();
    for (int h = tid; h < HIDDEN_DIM5; h += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < HIDDEN_DIM4; i++) {
            sum += s_hidden4[i] * d_W5[i * HIDDEN_DIM5 + h];
        }
        sum += d_b5[h];
        s_hidden5[h] = (sum > 0.0f) ? sum : 0.0f;
        d_hidden5[h] = s_hidden5[h];
    }
    __syncthreads();
    for (int v = tid; v < vocabSize; v += blockDim.x) {
        float sum = 0.0f;
        for (int h = 0; h < HIDDEN_DIM5; h++) {
            sum += s_hidden5[h] * d_Wout[h * MAX_VOCAB + v];
        }
        sum += d_bout[v];
        d_logits[v] = sum;
    }
    __syncthreads();
    float local_max = -1e20f;
    for (int v = tid; v < vocabSize; v += blockDim.x) {
        local_max = fmaxf(local_max, d_logits[v]);
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();
    float local_sum = 0.0f;
    for (int v = tid; v < vocabSize; v += blockDim.x) {
        float exp_val = expf(d_logits[v] - max_val);
        d_probs[v] = exp_val;
        local_sum += exp_val;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float sum_val = sdata[0];
    __syncthreads();
    for (int v = tid; v < vocabSize; v += blockDim.x) {
        d_probs[v] /= sum_val;
    }
    __syncthreads();
    if (tid == 0) {
        float prob = d_probs[targetIdx];
        if (prob < 1e-15f) prob = 1e-15f;
        *d_loss = -logf(prob);
    }
}

__global__ void fusedBackwardKernel(const float* d_input,
                                      const float* d_hidden1, const float* d_hidden2,
                                      const float* d_hidden3, const float* d_hidden4, const float* d_hidden5,
                                      const float* d_probs, int targetIdx,
                                      const float* d_W1, const float* d_W2, const float* d_W3,
                                      const float* d_W4, const float* d_W5,
                                      const float* d_Wout,
                                      float* d_dW1, float* d_db1,
                                      float* d_dW2, float* d_db2,
                                      float* d_dW3, float* d_db3,
                                      float* d_dW4, float* d_db4,
                                      float* d_dW5, float* d_db5,
                                      float* d_dWout, float* d_dbout,
                                      float* d_dinput, int vocabSize) {
    extern __shared__ float s_data[];
    float* s_dlogits = s_data;
    float* s_dhidden5 = s_dlogits + MAX_VOCAB;
    float* s_dhidden4 = s_dhidden5 + HIDDEN_DIM5;
    float* s_dhidden3 = s_dhidden4 + HIDDEN_DIM4;
    float* s_dhidden2 = s_dhidden3 + HIDDEN_DIM3;
    float* s_dhidden1 = s_dhidden2 + HIDDEN_DIM2;
    int tid = threadIdx.x;
    for (int v = tid; v < vocabSize; v += blockDim.x) {
        s_dlogits[v] = d_probs[v] - ((v == targetIdx) ? 1.0f : 0.0f);
    }
    __syncthreads();
    for (int h = 0; h < HIDDEN_DIM5; h++) {
        for (int v = tid; v < vocabSize; v += blockDim.x) {
            d_dWout[h * MAX_VOCAB + v] += s_dlogits[v] * d_hidden5[h];
        }
    }
    for (int v = tid; v < vocabSize; v += blockDim.x) {
        d_dbout[v] += s_dlogits[v];
    }
    __syncthreads();
    for (int h = tid; h < HIDDEN_DIM5; h += blockDim.x) {
        float sum = 0.0f;
        for (int v = 0; v < vocabSize; v++) {
            sum += s_dlogits[v] * d_Wout[h * MAX_VOCAB + v];
        }
        s_dhidden5[h] = (d_hidden5[h] > 0.0f) ? sum : 0.0f;
    }
    __syncthreads();
    for (int h = tid; h < HIDDEN_DIM4; h += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < HIDDEN_DIM5; i++) {
            sum += s_dhidden5[i] * d_W5[h * HIDDEN_DIM5 + i];
        }
        s_dhidden4[h] = (d_hidden4[h] > 0.0f) ? sum : 0.0f;
    }
    __syncthreads();
    for (int h = tid; h < HIDDEN_DIM3; h += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < HIDDEN_DIM4; i++) {
            sum += s_dhidden4[i] * d_W4[h * HIDDEN_DIM4 + i];
        }
        s_dhidden3[h] = (d_hidden3[h] > 0.0f) ? sum : 0.0f;
    }
    __syncthreads();
    for (int h = tid; h < HIDDEN_DIM2; h += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < HIDDEN_DIM3; i++) {
            sum += s_dhidden3[i] * d_W3[h * HIDDEN_DIM3 + i];
        }
        s_dhidden2[h] = (d_hidden2[h] > 0.0f) ? sum : 0.0f;
    }
    __syncthreads();
    for (int h = tid; h < HIDDEN_DIM1; h += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < HIDDEN_DIM2; i++) {
            sum += s_dhidden2[i] * d_W2[h * HIDDEN_DIM2 + i];
        }
        s_dhidden1[h] = (d_hidden1[h] > 0.0f) ? sum : 0.0f;
    }
    __syncthreads();
    for (int i = 0; i < EMBED_DIM; i++) {
        for (int h = tid; h < HIDDEN_DIM1; h += blockDim.x) {
            d_dW1[i * HIDDEN_DIM1 + h] += s_dhidden1[h] * d_input[i];
        }
    }
    for (int h = tid; h < HIDDEN_DIM1; h += blockDim.x) {
        d_db1[h] += s_dhidden1[h];
    }
    for (int i = 0; i < HIDDEN_DIM1; i++) {
        for (int h = tid; h < HIDDEN_DIM2; h += blockDim.x) {
            d_dW2[i * HIDDEN_DIM2 + h] += s_dhidden2[h] * d_hidden1[i];
        }
    }
    for (int h = tid; h < HIDDEN_DIM2; h += blockDim.x) {
        d_db2[h] += s_dhidden2[h];
    }
    for (int i = 0; i < HIDDEN_DIM2; i++) {
        for (int h = tid; h < HIDDEN_DIM3; h += blockDim.x) {
            d_dW3[i * HIDDEN_DIM3 + h] += s_dhidden3[h] * d_hidden2[i];
        }
    }
    for (int h = tid; h < HIDDEN_DIM3; h += blockDim.x) {
        d_db3[h] += s_dhidden3[h];
    }
    for (int i = 0; i < HIDDEN_DIM3; i++) {
        for (int h = tid; h < HIDDEN_DIM4; h += blockDim.x) {
            d_dW4[i * HIDDEN_DIM4 + h] += s_dhidden4[h] * d_hidden3[i];
        }
    }
    for (int h = tid; h < HIDDEN_DIM4; h += blockDim.x) {
        d_db4[h] += s_dhidden4[h];
    }
    for (int i = 0; i < HIDDEN_DIM4; i++) {
        for (int h = tid; h < HIDDEN_DIM5; h += blockDim.x) {
            d_dW5[i * HIDDEN_DIM5 + h] += s_dhidden5[h] * d_hidden4[i];
        }
    }
    for (int h = tid; h < HIDDEN_DIM5; h += blockDim.x) {
        d_db5[h] += s_dhidden5[h];
    }
    __syncthreads();
    for (int i = tid; i < EMBED_DIM; i += blockDim.x) {
        float sum = 0.0f;
        for (int h = 0; h < HIDDEN_DIM1; h++) {
            sum += s_dhidden1[h] * d_W1[i * HIDDEN_DIM1 + h];
        }
        d_dinput[i] = sum;
    }
}

__global__ void fusedUpdateKernel(float* d_W1, float* d_b1,
                                    float* d_W2, float* d_b2,
                                    float* d_W3, float* d_b3,
                                    float* d_W4, float* d_b4,
                                    float* d_W5, float* d_b5,
                                    float* d_Wout, float* d_bout,
                                    const float* d_dW1, const float* d_db1,
                                    const float* d_dW2, const float* d_db2,
                                    const float* d_dW3, const float* d_db3,
                                    const float* d_dW4, const float* d_db4,
                                    const float* d_dW5, const float* d_db5,
                                    const float* d_dWout, const float* d_dbout,
                                    float learningRate, int vocabSize) {
    int tid = threadIdx.x;
    for (int i = 0; i < EMBED_DIM; i++) {
        for (int h = tid; h < HIDDEN_DIM1; h += blockDim.x) {
            d_W1[i * HIDDEN_DIM1 + h] -= learningRate * d_dW1[i * HIDDEN_DIM1 + h];
        }
    }
    for (int h = tid; h < HIDDEN_DIM1; h += blockDim.x) {
        d_b1[h] -= learningRate * d_db1[h];
    }
    for (int i = 0; i < HIDDEN_DIM1; i++) {
        for (int h = tid; h < HIDDEN_DIM2; h += blockDim.x) {
            d_W2[i * HIDDEN_DIM2 + h] -= learningRate * d_dW2[i * HIDDEN_DIM2 + h];
        }
    }
    for (int h = tid; h < HIDDEN_DIM2; h += blockDim.x) {
        d_b2[h] -= learningRate * d_db2[h];
    }
    for (int i = 0; i < HIDDEN_DIM2; i++) {
        for (int h = tid; h < HIDDEN_DIM3; h += blockDim.x) {
            d_W3[i * HIDDEN_DIM3 + h] -= learningRate * d_dW3[i * HIDDEN_DIM3 + h];
        }
    }
    for (int h = tid; h < HIDDEN_DIM3; h += blockDim.x) {
        d_b3[h] -= learningRate * d_db3[h];
    }
    for (int i = 0; i < HIDDEN_DIM3; i++) {
        for (int h = tid; h < HIDDEN_DIM4; h += blockDim.x) {
            d_W4[i * HIDDEN_DIM4 + h] -= learningRate * d_dW4[i * HIDDEN_DIM4 + h];
        }
    }
    for (int h = tid; h < HIDDEN_DIM4; h += blockDim.x) {
        d_b4[h] -= learningRate * d_db4[h];
    }
    for (int i = 0; i < HIDDEN_DIM4; i++) {
        for (int h = tid; h < HIDDEN_DIM5; h += blockDim.x) {
            d_W5[i * HIDDEN_DIM5 + h] -= learningRate * d_dW5[i * HIDDEN_DIM5 + h];
        }
    }
    for (int h = tid; h < HIDDEN_DIM5; h += blockDim.x) {
        d_b5[h] -= learningRate * d_db5[h];
    }
    for (int h = 0; h < HIDDEN_DIM5; h++) {
        for (int v = tid; v < vocabSize; v += blockDim.x) {
            d_Wout[h * MAX_VOCAB + v] -= learningRate * d_dWout[h * MAX_VOCAB + v];
        }
    }
    for (int v = tid; v < vocabSize; v += blockDim.x) {
        d_bout[v] -= learningRate * d_dbout[v];
    }
}

int main(int argc, char *argv[]) {
    std::string inputText;
    if (argc > 1) {
        inputText = std::string(argv[1]);
    } else {
        std::ifstream file("input.txt");
        std::stringstream ss;
        ss << file.rdbuf();
        inputText = ss.str();
    }
    std::istringstream iss(inputText);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    std::unordered_map<std::string, int> vocabMap;
    std::vector<std::string> vocab;
    std::vector<int> tokenIds;
    for (const auto &str : tokens) {
        if (vocabMap.find(str) == vocabMap.end()) {
            if (vocab.size() < MAX_VOCAB) {
                int id = vocab.size();
                vocabMap[str] = id;
                vocab.push_back(str);
            } else {
                vocabMap[str] = 0;
            }
        }
        tokenIds.push_back(vocabMap[str]);
    }
    int vocabSize = vocab.size();
    std::random_device rd;
    std::mt19937 gen(rd());
    float embedScale = sqrtf(6.0f / (MAX_VOCAB + EMBED_DIM));
    float w1Scale = sqrtf(6.0f / (EMBED_DIM + HIDDEN_DIM1));
    float w2Scale = sqrtf(6.0f / (HIDDEN_DIM1 + HIDDEN_DIM2));
    float w3Scale = sqrtf(6.0f / (HIDDEN_DIM2 + HIDDEN_DIM3));
    float w4Scale = sqrtf(6.0f / (HIDDEN_DIM3 + HIDDEN_DIM4));
    float w5Scale = sqrtf(6.0f / (HIDDEN_DIM4 + HIDDEN_DIM5));
    float outScale = sqrtf(6.0f / (HIDDEN_DIM5 + MAX_VOCAB));
    std::uniform_real_distribution<> embedDist(-embedScale, embedScale);
    std::uniform_real_distribution<> w1Dist(-w1Scale, w1Scale);
    std::uniform_real_distribution<> w2Dist(-w2Scale, w2Scale);
    std::uniform_real_distribution<> w3Dist(-w3Scale, w3Scale);
    std::uniform_real_distribution<> w4Dist(-w4Scale, w4Scale);
    std::uniform_real_distribution<> w5Dist(-w5Scale, w5Scale);
    std::uniform_real_distribution<> outDist(-outScale, outScale);
    std::uniform_real_distribution<> bDist(-0.1f, 0.1f);
    std::vector<float> h_embedding(MAX_VOCAB * EMBED_DIM);
    std::vector<float> h_W1(EMBED_DIM * HIDDEN_DIM1);
    std::vector<float> h_b1(HIDDEN_DIM1);
    std::vector<float> h_W2(HIDDEN_DIM1 * HIDDEN_DIM2);
    std::vector<float> h_b2(HIDDEN_DIM2);
    std::vector<float> h_W3(HIDDEN_DIM2 * HIDDEN_DIM3);
    std::vector<float> h_b3(HIDDEN_DIM3);
    std::vector<float> h_W4(HIDDEN_DIM3 * HIDDEN_DIM4);
    std::vector<float> h_b4(HIDDEN_DIM4);
    std::vector<float> h_W5(HIDDEN_DIM4 * HIDDEN_DIM5);
    std::vector<float> h_b5(HIDDEN_DIM5);
    std::vector<float> h_Wout(HIDDEN_DIM5 * MAX_VOCAB);
    std::vector<float> h_bout(MAX_VOCAB);
    for (int i = 0; i < MAX_VOCAB * EMBED_DIM; i++) {
        h_embedding[i] = embedDist(gen);
    }
    for (int i = 0; i < EMBED_DIM * HIDDEN_DIM1; i++) {
        h_W1[i] = w1Dist(gen);
    }
    for (int i = 0; i < HIDDEN_DIM1; i++) {
        h_b1[i] = bDist(gen);
    }
    for (int i = 0; i < HIDDEN_DIM1 * HIDDEN_DIM2; i++) {
        h_W2[i] = w2Dist(gen);
    }
    for (int i = 0; i < HIDDEN_DIM2; i++) {
        h_b2[i] = bDist(gen);
    }
    for (int i = 0; i < HIDDEN_DIM2 * HIDDEN_DIM3; i++) {
        h_W3[i] = w3Dist(gen);
    }
    for (int i = 0; i < HIDDEN_DIM3; i++) {
        h_b3[i] = bDist(gen);
    }
    for (int i = 0; i < HIDDEN_DIM3 * HIDDEN_DIM4; i++) {
        h_W4[i] = w4Dist(gen);
    }
    for (int i = 0; i < HIDDEN_DIM4; i++) {
        h_b4[i] = bDist(gen);
    }
    for (int i = 0; i < HIDDEN_DIM4 * HIDDEN_DIM5; i++) {
        h_W5[i] = w5Dist(gen);
    }
    for (int i = 0; i < HIDDEN_DIM5; i++) {
        h_b5[i] = bDist(gen);
    }
    for (int i = 0; i < HIDDEN_DIM5 * MAX_VOCAB; i++) {
        h_Wout[i] = outDist(gen);
    }
    for (int i = 0; i < MAX_VOCAB; i++) {
        h_bout[i] = bDist(gen);
    }
    float *d_embedding;
    float *d_W1, *d_b1, *d_W2, *d_b2, *d_W3, *d_b3, *d_W4, *d_b4, *d_W5, *d_b5;
    float *d_Wout, *d_bout;
    float *d_dW1, *d_db1, *d_dW2, *d_db2, *d_dW3, *d_db3, *d_dW4, *d_db4, *d_dW5, *d_db5;
    float *d_dWout, *d_dbout;
    float *d_hidden1, *d_hidden2, *d_hidden3, *d_hidden4, *d_hidden5;
    float *d_logits, *d_probs;
    float *d_loss;
    float *d_input, *d_dinput;
    int *d_context;
    CHECK_CUDA(cudaMalloc(&d_embedding, MAX_VOCAB * EMBED_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W1, EMBED_DIM * HIDDEN_DIM1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b1, HIDDEN_DIM1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W2, HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b2, HIDDEN_DIM2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W3, HIDDEN_DIM2 * HIDDEN_DIM3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b3, HIDDEN_DIM3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W4, HIDDEN_DIM3 * HIDDEN_DIM4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b4, HIDDEN_DIM4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W5, HIDDEN_DIM4 * HIDDEN_DIM5 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b5, HIDDEN_DIM5 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wout, HIDDEN_DIM5 * MAX_VOCAB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bout, MAX_VOCAB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dW1, EMBED_DIM * HIDDEN_DIM1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_db1, HIDDEN_DIM1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dW2, HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_db2, HIDDEN_DIM2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dW3, HIDDEN_DIM2 * HIDDEN_DIM3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_db3, HIDDEN_DIM3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dW4, HIDDEN_DIM3 * HIDDEN_DIM4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_db4, HIDDEN_DIM4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dW5, HIDDEN_DIM4 * HIDDEN_DIM5 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_db5, HIDDEN_DIM5 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dWout, HIDDEN_DIM5 * MAX_VOCAB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dbout, MAX_VOCAB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_hidden1, HIDDEN_DIM1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_hidden2, HIDDEN_DIM2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_hidden3, HIDDEN_DIM3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_hidden4, HIDDEN_DIM4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_hidden5, HIDDEN_DIM5 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_logits, MAX_VOCAB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_probs, MAX_VOCAB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_loss, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_input, EMBED_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dinput, EMBED_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_context, CONTEXT_LENGTH * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_embedding, h_embedding.data(), MAX_VOCAB * EMBED_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W1, h_W1.data(), EMBED_DIM * HIDDEN_DIM1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b1, h_b1.data(), HIDDEN_DIM1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W2, h_W2.data(), HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b2, h_b2.data(), HIDDEN_DIM2 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W3, h_W3.data(), HIDDEN_DIM2 * HIDDEN_DIM3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b3, h_b3.data(), HIDDEN_DIM3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W4, h_W4.data(), HIDDEN_DIM3 * HIDDEN_DIM4 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b4, h_b4.data(), HIDDEN_DIM4 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W5, h_W5.data(), HIDDEN_DIM4 * HIDDEN_DIM5 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b5, h_b5.data(), HIDDEN_DIM5 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wout, h_Wout.data(), HIDDEN_DIM5 * MAX_VOCAB * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bout, h_bout.data(), MAX_VOCAB * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_dW1, 0, EMBED_DIM * HIDDEN_DIM1 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_db1, 0, HIDDEN_DIM1 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_dW2, 0, HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_db2, 0, HIDDEN_DIM2 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_dW3, 0, HIDDEN_DIM2 * HIDDEN_DIM3 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_db3, 0, HIDDEN_DIM3 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_dW4, 0, HIDDEN_DIM3 * HIDDEN_DIM4 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_db4, 0, HIDDEN_DIM4 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_dW5, 0, HIDDEN_DIM4 * HIDDEN_DIM5 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_db5, 0, HIDDEN_DIM5 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_dWout, 0, HIDDEN_DIM5 * MAX_VOCAB * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_dbout, 0, MAX_VOCAB * sizeof(float)));
    int blockDimVal = 256;
    int sharedMemSize = (HIDDEN_DIM1 + HIDDEN_DIM2 + HIDDEN_DIM3 + HIDDEN_DIM4 + HIDDEN_DIM5 + MAX_VOCAB) * sizeof(float);
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float totalLoss = 0.0f;
        for (int i = 0; i < (int)tokenIds.size() - CONTEXT_LENGTH - 1; i += BATCH_SIZE) {
            CHECK_CUDA(cudaMemset(d_dW1, 0, EMBED_DIM * HIDDEN_DIM1 * sizeof(float)));
            CHECK_CUDA(cudaMemset(d_db1, 0, HIDDEN_DIM1 * sizeof(float)));
            CHECK_CUDA(cudaMemset(d_dW2, 0, HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(float)));
            CHECK_CUDA(cudaMemset(d_db2, 0, HIDDEN_DIM2 * sizeof(float)));
            CHECK_CUDA(cudaMemset(d_dW3, 0, HIDDEN_DIM2 * HIDDEN_DIM3 * sizeof(float)));
            CHECK_CUDA(cudaMemset(d_db3, 0, HIDDEN_DIM3 * sizeof(float)));
            CHECK_CUDA(cudaMemset(d_dW4, 0, HIDDEN_DIM3 * HIDDEN_DIM4 * sizeof(float)));
            CHECK_CUDA(cudaMemset(d_db4, 0, HIDDEN_DIM4 * sizeof(float)));
            CHECK_CUDA(cudaMemset(d_dW5, 0, HIDDEN_DIM4 * HIDDEN_DIM5 * sizeof(float)));
            CHECK_CUDA(cudaMemset(d_db5, 0, HIDDEN_DIM5 * sizeof(float)));
            CHECK_CUDA(cudaMemset(d_dWout, 0, HIDDEN_DIM5 * MAX_VOCAB * sizeof(float)));
            CHECK_CUDA(cudaMemset(d_dbout, 0, MAX_VOCAB * sizeof(float)));
            for (int batch_idx = 0; batch_idx < BATCH_SIZE && i + batch_idx < (int)tokenIds.size() - CONTEXT_LENGTH - 1; ++batch_idx) {
                int targetWordIndex = tokenIds[i + batch_idx + CONTEXT_LENGTH];
                int context_start_index = i + batch_idx;
                std::vector<int> h_context(CONTEXT_LENGTH);
                for (int context_offset = 0; context_offset < CONTEXT_LENGTH; ++context_offset) {
                    h_context[context_offset] = tokenIds[context_start_index + context_offset];
                }
                CHECK_CUDA(cudaMemcpy(d_context, h_context.data(), CONTEXT_LENGTH * sizeof(int), cudaMemcpyHostToDevice));
                embeddingLookupKernel<<<1, EMBED_DIM>>>(d_embedding, d_context, d_input);
                CHECK_CUDA(cudaDeviceSynchronize());
                fusedForwardLossKernel<<<1, blockDimVal, sharedMemSize>>>(d_input,
                    d_W1, d_b1, d_W2, d_b2, d_W3, d_b3, d_W4, d_b4, d_W5, d_b5,
                    d_Wout, d_bout,
                    d_hidden1, d_hidden2, d_hidden3, d_hidden4, d_hidden5,
                    d_logits, d_probs, d_loss,
                    vocabSize, targetWordIndex);
                CHECK_CUDA(cudaDeviceSynchronize());
                fusedBackwardKernel<<<1, blockDimVal, sharedMemSize>>>(d_input,
                    d_hidden1, d_hidden2, d_hidden3, d_hidden4, d_hidden5,
                    d_probs, targetWordIndex,
                    d_W1, d_W2, d_W3, d_W4, d_W5,
                    d_Wout,
                    d_dW1, d_db1, d_dW2, d_db2, d_dW3, d_db3, d_dW4, d_db4, d_dW5, d_db5,
                    d_dWout, d_dbout,
                    d_dinput, vocabSize);
                CHECK_CUDA(cudaDeviceSynchronize());
                float currentLoss;
                CHECK_CUDA(cudaMemcpy(&currentLoss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
                totalLoss += currentLoss;
            }
            fusedUpdateKernel<<<1, blockDimVal>>>(d_W1, d_b1, d_W2, d_b2, d_W3, d_b3, d_W4, d_b4, d_W5, d_b5,
                d_Wout, d_bout,
                d_dW1, d_db1, d_dW2, d_db2, d_dW3, d_db3, d_dW4, d_db4, d_dW5, d_db5,
                d_dWout, d_dbout,
                LEARNING_RATE, vocabSize);
            CHECK_CUDA(cudaDeviceSynchronize());
            if (i % (BATCH_SIZE * 10) == 0) {
                float avgLoss = totalLoss / (((i / BATCH_SIZE) + 1));
                std::cout << "Epoch: " << epoch << ", Batch: " << i / BATCH_SIZE << ", Avg Loss: " << avgLoss << std::endl;
            }
        }
    }
    CHECK_CUDA(cudaFree(d_embedding));
    CHECK_CUDA(cudaFree(d_W1)); CHECK_CUDA(cudaFree(d_b1)); CHECK_CUDA(cudaFree(d_W2)); CHECK_CUDA(cudaFree(d_b2)); CHECK_CUDA(cudaFree(d_W3)); CHECK_CUDA(cudaFree(d_b3)); CHECK_CUDA(cudaFree(d_W4)); CHECK_CUDA(cudaFree(d_b4)); CHECK_CUDA(cudaFree(d_W5)); CHECK_CUDA(cudaFree(d_b5));
    CHECK_CUDA(cudaFree(d_Wout)); CHECK_CUDA(cudaFree(d_bout));
    CHECK_CUDA(cudaFree(d_dW1)); CHECK_CUDA(cudaFree(d_db1)); CHECK_CUDA(cudaFree(d_dW2)); CHECK_CUDA(cudaFree(d_db2)); CHECK_CUDA(cudaFree(d_dW3)); CHECK_CUDA(cudaFree(d_db3)); CHECK_CUDA(cudaFree(d_dW4)); CHECK_CUDA(cudaFree(d_db4)); CHECK_CUDA(cudaFree(d_dW5)); CHECK_CUDA(cudaFree(d_db5));
    CHECK_CUDA(cudaFree(d_dWout)); CHECK_CUDA(cudaFree(d_dbout));
    CHECK_CUDA(cudaFree(d_hidden1)); CHECK_CUDA(cudaFree(d_hidden2)); CHECK_CUDA(cudaFree(d_hidden3)); CHECK_CUDA(cudaFree(d_hidden4)); CHECK_CUDA(cudaFree(d_hidden5));
    CHECK_CUDA(cudaFree(d_logits)); CHECK_CUDA(cudaFree(d_probs)); CHECK_CUDA(cudaFree(d_loss)); CHECK_CUDA(cudaFree(d_input)); CHECK_CUDA(cudaFree(d_dinput));
    CHECK_CUDA(cudaFree(d_context));
    return 0;
}

