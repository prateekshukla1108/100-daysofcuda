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
#define HIDDEN_DIM 256
#define MAX_VOCAB 1024
#define LEARNING_RATE 0.01f
#define BATCH_SIZE 16
#define EPOCHS 10

__global__ void fusedForwardLossKernel(const float* d_input, const float* d_W1, const float* d_b1, const float* d_W2, const float* d_b2, float* d_hidden, float* d_logits, float* d_probs, float* d_loss, int vocabSize, int targetIdx) {
  extern __shared__ float shared[];
  float* s_hidden = shared;
  float* sdata = shared + HIDDEN_DIM;
  int tid = threadIdx.x;
  for (int h = tid; h < HIDDEN_DIM; h += blockDim.x) {
    float sum = 0.0f;
    for (int i = 0; i < EMBED_DIM; i++) {
      sum += d_input[i] * d_W1[i * HIDDEN_DIM + h];
    }
    sum += d_b1[h];
    s_hidden[h] = (sum > 0.0f) ? sum : 0.0f;
    d_hidden[h] = s_hidden[h];
  }
  __syncthreads();
  for (int v = tid; v < vocabSize; v += blockDim.x) {
    float sum = 0.0f;
    for (int h = 0; h < HIDDEN_DIM; h++) {
      sum += s_hidden[h] * d_W2[h * MAX_VOCAB + v];
    }
    sum += d_b2[v];
    d_logits[v] = sum;
  }
  __syncthreads();
  float local_max = -1e20f;
  for (int v = tid; v < vocabSize; v += blockDim.x) {
    local_max = fmaxf(local_max, d_logits[v]);
  }
  sdata[tid] = local_max;
  __syncthreads();
  for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
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
  for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
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
    if(prob < 1e-15f) prob = 1e-15f;
    *d_loss = -logf(prob);
  }
}

__global__ void fusedBackwardKernel(const float* d_input, const float* d_hidden, const float* d_probs, int targetIdx, const float* d_W1, const float* d_W2, float* d_dW1, float* d_db1, float* d_dW2, float* d_db2, float* d_dinput, int vocabSize) {
  extern __shared__ float s_data[];
  float* s_dlogits = s_data;
  float* s_dhidden = s_data + vocabSize;
  int tid = threadIdx.x;
  for (int v = tid; v < vocabSize; v += blockDim.x) {
    s_dlogits[v] = d_probs[v] - (v == targetIdx ? 1.0f : 0.0f);
  }
  __syncthreads();
  for (int h = 0; h < HIDDEN_DIM; h++) {
    for (int v = tid; v < vocabSize; v += blockDim.x) {
      atomicAdd(&d_dW2[h * MAX_VOCAB + v], s_dlogits[v] * d_hidden[h]);
    }
  }
  for (int v = tid; v < vocabSize; v += blockDim.x) {
    atomicAdd(&d_db2[v], s_dlogits[v]);
  }
  __syncthreads();
  for (int h = tid; h < HIDDEN_DIM; h += blockDim.x) {
    float sum = 0.0f;
    for (int v = 0; v < vocabSize; v++) {
      sum += s_dlogits[v] * d_W2[h * MAX_VOCAB + v];
    }
    s_dhidden[h] = (d_hidden[h] > 0.0f) ? sum : 0.0f;
  }
  __syncthreads();
  for (int i = 0; i < EMBED_DIM; i++) {
    for (int h = tid; h < HIDDEN_DIM; h += blockDim.x) {
      atomicAdd(&d_dW1[i * HIDDEN_DIM + h], s_dhidden[h] * d_input[i]);
    }
  }
  for (int h = tid; h < HIDDEN_DIM; h += blockDim.x) {
    atomicAdd(&d_db1[h], s_dhidden[h]);
  }
  __syncthreads();
  for (int i = tid; i < EMBED_DIM; i += blockDim.x) {
    float sum = 0.0f;
    for (int h = 0; h < HIDDEN_DIM; h++) {
      sum += s_dhidden[h] * d_W1[i * HIDDEN_DIM + h];
    }
    d_dinput[i] = sum;
  }
}

__global__ void fusedUpdateKernel(float* d_W1, float* d_b1, float* d_W2, float* d_b2, const float* d_dW1, const float* d_db1, const float* d_dW2, const float* d_db2, float learningRate, int vocabSize) {
  int tid = threadIdx.x;
  for (int i = 0; i < EMBED_DIM; i++) {
    for (int h = tid; h < HIDDEN_DIM; h += blockDim.x) {
      d_W1[i * HIDDEN_DIM + h] -= learningRate * d_dW1[i * HIDDEN_DIM + h];
    }
  }
  for (int h = tid; h < HIDDEN_DIM; h += blockDim.x) {
    d_b1[h] -= learningRate * d_db1[h];
  }
  for (int h = 0; h < HIDDEN_DIM; h++) {
    for (int v = tid; v < vocabSize; v += blockDim.x) {
      d_W2[h * MAX_VOCAB + v] -= learningRate * d_dW2[h * MAX_VOCAB + v];
    }
  }
  for (int v = tid; v < vocabSize; v += blockDim.x) {
    d_b2[v] -= learningRate * d_db2[v];
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
  float w1Scale = sqrtf(6.0f / (EMBED_DIM + HIDDEN_DIM));
  float w2Scale = sqrtf(6.0f / (HIDDEN_DIM + MAX_VOCAB));
  std::uniform_real_distribution<> embedDist(-embedScale, embedScale);
  std::uniform_real_distribution<> w1Dist(-w1Scale, w1Scale);
  std::uniform_real_distribution<> w2Dist(-w2Scale, w2Scale);
  std::uniform_real_distribution<> bDist(-0.1f, 0.1f);
  std::vector<float> h_embedding(MAX_VOCAB * EMBED_DIM);
  for (int i = 0; i < MAX_VOCAB * EMBED_DIM; i++) {
    h_embedding[i] = embedDist(gen);
  }
  std::vector<float> h_W1(EMBED_DIM * HIDDEN_DIM);
  std::vector<float> h_b1(HIDDEN_DIM);
  std::vector<float> h_W2(HIDDEN_DIM * MAX_VOCAB);
  std::vector<float> h_b2(MAX_VOCAB);
  for (int i = 0; i < EMBED_DIM * HIDDEN_DIM; i++) {
    h_W1[i] = w1Dist(gen);
  }
  for (int i = 0; i < HIDDEN_DIM; i++) {
    h_b1[i] = bDist(gen);
  }
  for (int i = 0; i < HIDDEN_DIM * MAX_VOCAB; i++) {
    h_W2[i] = w2Dist(gen);
  }
  for (int i = 0; i < MAX_VOCAB; i++) {
    h_b2[i] = bDist(gen);
  }
  float *d_embedding, *d_W1, *d_b1, *d_W2, *d_b2;
  float *d_dW1, *d_db1, *d_dW2, *d_db2;
  float *d_loss;
  cudaMalloc(&d_embedding, MAX_VOCAB * EMBED_DIM * sizeof(float));
  cudaMalloc(&d_W1, EMBED_DIM * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_b1, HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_W2, HIDDEN_DIM * MAX_VOCAB * sizeof(float));
  cudaMalloc(&d_b2, MAX_VOCAB * sizeof(float));
  cudaMalloc(&d_dW1, EMBED_DIM * HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_db1, HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_dW2, HIDDEN_DIM * MAX_VOCAB * sizeof(float));
  cudaMalloc(&d_db2, MAX_VOCAB * sizeof(float));
  cudaMalloc(&d_loss, sizeof(float));
  cudaMemcpy(d_embedding, h_embedding.data(), MAX_VOCAB * EMBED_DIM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W1, h_W1.data(), EMBED_DIM * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b1, h_b1.data(), HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W2, h_W2.data(), HIDDEN_DIM * MAX_VOCAB * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b2, h_b2.data(), MAX_VOCAB * sizeof(float), cudaMemcpyHostToDevice);
  float *d_input, *d_hidden, *d_logits, *d_probs, *d_dinput;
  cudaMalloc(&d_input, EMBED_DIM * sizeof(float));
  cudaMalloc(&d_hidden, HIDDEN_DIM * sizeof(float));
  cudaMalloc(&d_logits, vocabSize * sizeof(float));
  cudaMalloc(&d_probs, vocabSize * sizeof(float));
  cudaMalloc(&d_dinput, EMBED_DIM * sizeof(float));
  dim3 block(256);
  std::cout << "Starting training for " << EPOCHS << " epochs..." << std::endl;
  float totalLoss = 0.0f;
  int totalSamples = 0;
  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    totalLoss = 0.0f;
    totalSamples = 0;
    for (int i = 0; i < (int)tokenIds.size() - 1; i++) {
      int inputId = tokenIds[i];
      int targetId = tokenIds[i + 1];
      std::vector<float> h_input(EMBED_DIM, 0.0f);
      for (int j = 0; j < EMBED_DIM; j++) {
        h_input[j] = h_embedding[inputId * EMBED_DIM + j];
      }
      cudaMemcpy(d_input, h_input.data(), EMBED_DIM * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemset(d_dW1, 0, EMBED_DIM * HIDDEN_DIM * sizeof(float));
      cudaMemset(d_db1, 0, HIDDEN_DIM * sizeof(float));
      cudaMemset(d_dW2, 0, HIDDEN_DIM * MAX_VOCAB * sizeof(float));
      cudaMemset(d_db2, 0, MAX_VOCAB * sizeof(float));
      size_t forwardSharedMem = (HIDDEN_DIM + 256) * sizeof(float);
      fusedForwardLossKernel<<<1, block, forwardSharedMem>>>(d_input, d_W1, d_b1, d_W2, d_b2, d_hidden, d_logits, d_probs, d_loss, vocabSize, targetId);
      size_t backwardSharedMem = (vocabSize + HIDDEN_DIM) * sizeof(float);
      fusedBackwardKernel<<<1, block, backwardSharedMem>>>(d_input, d_hidden, d_probs, targetId, d_W1, d_W2, d_dW1, d_db1, d_dW2, d_db2, d_dinput, vocabSize);
      fusedUpdateKernel<<<1, block>>>(d_W1, d_b1, d_W2, d_b2, d_dW1, d_db1, d_dW2, d_db2, LEARNING_RATE, vocabSize);
      std::vector<float> h_dinput(EMBED_DIM);
      cudaMemcpy(h_dinput.data(), d_dinput, EMBED_DIM * sizeof(float), cudaMemcpyDeviceToHost);
      for (int j = 0; j < EMBED_DIM; j++) {
        h_embedding[inputId * EMBED_DIM + j] -= LEARNING_RATE * h_dinput[j];
      }
      cudaMemcpy(d_embedding, h_embedding.data(), MAX_VOCAB * EMBED_DIM * sizeof(float), cudaMemcpyHostToDevice);
      float h_loss;
      cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
      totalLoss += h_loss;
      totalSamples++;
      if (i % 100 == 0) {
        std::cout << "Epoch " << epoch+1 << ", Sample " << i << ", Loss: " << h_loss << ", Avg Loss: " << totalLoss / totalSamples << std::endl;
      }
    }
    std::cout << "Epoch " << epoch+1 << " completed. Average loss: " << totalLoss / totalSamples << std::endl;
  }
  std::cout << "\nTraining complete! Demonstrating inference:" << std::endl;
  int start = tokenIds.size() >= CONTEXT_LENGTH ? tokenIds.size() - CONTEXT_LENGTH : 0;
  std::vector<std::string> contextTokens;
  for (int i = start; i < tokenIds.size(); i++) {
    contextTokens.push_back(tokens[i]);
  }
  std::vector<float> h_input_infer(EMBED_DIM, 0.0f);
  for (int i = start; i < tokenIds.size(); i++) {
    int id = tokenIds[i];
    for (int j = 0; j < EMBED_DIM; j++) {
      h_input_infer[j] += h_embedding[id * EMBED_DIM + j];
    }
  }
  for (int j = 0; j < EMBED_DIM; j++) {
    h_input_infer[j] /= (tokenIds.size() - start);
  }
  cudaMemcpy(d_input, h_input_infer.data(), EMBED_DIM * sizeof(float), cudaMemcpyHostToDevice);
  size_t forwardSharedMem = (HIDDEN_DIM + 256) * sizeof(float);
  fusedForwardLossKernel<<<1, block, forwardSharedMem>>>(d_input, d_W1, d_b1, d_W2, d_b2, d_hidden, d_logits, d_probs, d_loss, vocabSize, 0);
  std::vector<float> h_probs(vocabSize);
  cudaMemcpy(h_probs.data(), d_probs, vocabSize * sizeof(float), cudaMemcpyDeviceToHost);
  int maxIdx = 0;
  float maxProb = h_probs[0];
  for (int i = 1; i < vocabSize; i++) {
    if (h_probs[i] > maxProb) {
      maxProb = h_probs[i];
      maxIdx = i;
    }
  }
  std::cout << "Input Context: ";
  for (const auto &s : contextTokens) {
    std::cout << s << " ";
  }
  std::cout << "\nPredicted token: " << vocab[maxIdx] << " with probability " << maxProb << std::endl;
  cudaFree(d_embedding);
  cudaFree(d_W1);
  cudaFree(d_b1);
  cudaFree(d_W2);
  cudaFree(d_b2);
  cudaFree(d_dW1);
  cudaFree(d_db1);
  cudaFree(d_dW2);
  cudaFree(d_db2);
  cudaFree(d_loss);
  cudaFree(d_input);
  cudaFree(d_hidden);
  cudaFree(d_logits);
  cudaFree(d_probs);
  cudaFree(d_dinput);
  return 0;
}

