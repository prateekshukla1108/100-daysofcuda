#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
using namespace std;

#define TRAIN_SIZE 60000
#define TEST_SIZE 10000
#define INPUT_SIZE 784
#define INITIAL_NEURONS 64
#define NUM_CLASSES 10
#define SPARSITY 0.1f
#define LEARNING_RATE 0.001f
#define EPOCHS 5
#define MARGIN 1.0f
#define NUM_LAYERS 3
#define MAX_NEURONS 1024
#define TOPK_FRAC 0.10f

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
  if(code != cudaSuccess){
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    if(abort) exit(code);
  }
}

struct CompareAbs {
  __host__ __device__
  bool operator()(const float a, const float b) const {
    return fabsf(a) > fabsf(b);
  }
};

__device__ float relu(float x){ return x > 0.0f ? x : 0.0f; }

__device__ float atomicMaxFloat(float* address, float val){
  int* address_as_i = (int*)address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    if(__int_as_float(assumed) >= val)
      break;
    old = atomicCAS(address_as_i, assumed, __float_as_int(val));
  } while(assumed != old);
  return __int_as_float(old);
}

__global__ void computeDeltaKernel(const float* input_pos, const float* input_neg, 
                                     const float* act_pos, const float* act_neg, 
                                     const float* theta, int n_in, int n_out, int label, float* d_delta) {
  int j = blockIdx.x;
  int i = threadIdx.x;
  if(j < n_out && i < n_in){
    float cp = (act_pos[j] > 0.0f) ? input_pos[i] : 0.0f;
    float cn = (act_neg[j] > 0.0f) ? input_neg[i] : 0.0f;
    float delta = LEARNING_RATE * theta[label * n_out + j] * (cp - cn);
    d_delta[i * n_out + j] = delta;
  }
}

__global__ void topKWeightUpdateKernel(const int* topKIndices, const float* d_delta, float* W, int topK) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < topK){
    int flatIdx = topKIndices[idx];
    W[flatIdx] += d_delta[flatIdx];
  }
}

__global__ void fusedForwardKernel(const float* input_pos, const float* input_neg, 
                                     float* act_pos, float* act_neg, 
                                     const float* W, const float* mask, 
                                     int n_in, int n_out) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(j < n_out){
    float sum_pos = 0.0f, sum_neg = 0.0f;
    for (int i = 0; i < n_in; i++){
      int idx = i * n_out + j;
      float w = W[idx], m = mask[idx];
      sum_pos += m * w * input_pos[i];
      sum_neg += m * w * input_neg[i];
    }
    act_pos[j] = relu(sum_pos);
    act_neg[j] = relu(sum_neg);
  }
}

__global__ void fusedGoodnessKernel(const float* act_pos, const float* act_neg, 
                                      const float* theta, 
                                      float* goodness_pos, float* goodness_neg, 
                                      int n_out, int label) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float s_pos[256];
  __shared__ float s_neg[256];
  int tid = threadIdx.x;
  int valid = min(blockDim.x, n_out - blockIdx.x * blockDim.x);
  float local_pos = 0.0f, local_neg = 0.0f;
  if(tid < valid && idx < n_out){
    float tval = theta[label * n_out + idx];
    local_pos = act_pos[idx] * tval;
    local_neg = act_neg[idx] * tval;
  }
  s_pos[tid] = local_pos; s_neg[tid] = local_neg;
  __syncthreads();
  for(unsigned int stride = valid/2; stride > 0; stride >>= 1){
    if(tid < stride && tid + stride < valid){
      s_pos[tid] += s_pos[tid+stride];
      s_neg[tid] += s_neg[tid+stride];
    }
    __syncthreads();
  }
  if(tid == 0){
    atomicAdd(goodness_pos, s_pos[0]);
    atomicAdd(goodness_neg, s_neg[0]);
  }
}

__global__ void updateThetaKernel(float* theta, const float* act_pos, const float* act_neg, float loss, int label, int n_out) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(loss > 0.0f && j < n_out)
    theta[label * n_out + j] += LEARNING_RATE * (act_pos[j] - act_neg[j]);
}

struct Layer {
  int n_in, n_out;
  float* h_W;
  float* h_mask;
  float* h_theta;
  float* d_W;
  float* d_mask;
  float* d_theta;
  float* d_act_pos;
  float* d_act_neg;
};

void allocateLayer(Layer &layer, int n_in, int n_out){
  if(n_in > MAX_NEURONS){ cerr << "Layer width exceeds CUDA limits" << endl; exit(1); }
  layer.n_in = n_in; layer.n_out = n_out;
  int weightSize = n_in * n_out, thetaSize = n_out * NUM_CLASSES;
  layer.h_W = new float[weightSize];
  layer.h_mask = new float[weightSize];
  layer.h_theta = new float[thetaSize];
  for(int i = 0; i < n_in; i++){
    for(int j = 0; j < n_out; j++){
      int idx = i * n_out + j;
      layer.h_W[idx] = (((float)rand()/RAND_MAX)-0.5f)/5.0f;
      float r = (float)rand()/RAND_MAX;
      layer.h_mask[idx] = (r < SPARSITY) ? 1.0f : 0.0f;
    }
  }
  for(int i = 0; i < NUM_CLASSES; i++){
    for(int j = 0; j < n_out; j++){
      int idx = i * n_out + j;
      layer.h_theta[idx] = (((float)rand()/RAND_MAX)-0.5f)/5.0f;
    }
  }
  cudaCheckError(cudaMalloc(&layer.d_W, weightSize * sizeof(float)));
  cudaCheckError(cudaMalloc(&layer.d_mask, weightSize * sizeof(float)));
  cudaCheckError(cudaMalloc(&layer.d_theta, thetaSize * sizeof(float)));
  cudaCheckError(cudaMemcpy(layer.d_W, layer.h_W, weightSize * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(layer.d_mask, layer.h_mask, weightSize * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(layer.d_theta, layer.h_theta, thetaSize * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheckError(cudaMalloc(&layer.d_act_pos, n_out * sizeof(float)));
  cudaCheckError(cudaMalloc(&layer.d_act_neg, n_out * sizeof(float)));
}

void freeLayer(Layer &layer){
  cudaFree(layer.d_W); cudaFree(layer.d_mask); cudaFree(layer.d_theta);
  cudaFree(layer.d_act_pos); cudaFree(layer.d_act_neg);
  delete[] layer.h_W; delete[] layer.h_mask; delete[] layer.h_theta;
}

void dynamicCapacityAllocation(Layer &layer, Layer *next, const vector<float>& neuronGoodnessAvg) {
  int n = layer.n_out;
  float maxGood = *max_element(neuronGoodnessAvg.begin(), neuronGoodnessAvg.end());
  float pruneThreshold = 0.2f * maxGood;
  int keepCount = 0;
  for(int j = 0; j < n; j++){
    if(neuronGoodnessAvg[j] >= pruneThreshold) keepCount++;
  }
  if(keepCount < n){
    int new_n_out = keepCount;
    int newWeightSize = layer.n_in * new_n_out, newThetaSize = new_n_out * NUM_CLASSES;
    float* new_h_W = new float[newWeightSize];
    float* new_h_mask = new float[newWeightSize];
    float* new_h_theta = new float[newThetaSize];
    int newj = 0;
    for(int j = 0; j < n; j++){
      if(neuronGoodnessAvg[j] >= pruneThreshold){
        for(int i = 0; i < layer.n_in; i++){
          new_h_W[i * new_n_out + newj] = layer.h_W[i * n + j];
          new_h_mask[i * new_n_out + newj] = layer.h_mask[i * n + j];
        }
        for(int i = 0; i < NUM_CLASSES; i++){
          new_h_theta[i * new_n_out + newj] = layer.h_theta[i * n + j];
        }
        newj++;
      }
    }
    cudaCheckError(cudaFree(layer.d_W)); cudaCheckError(cudaFree(layer.d_mask));
    cudaCheckError(cudaFree(layer.d_theta)); cudaCheckError(cudaFree(layer.d_act_pos));
    cudaCheckError(cudaFree(layer.d_act_neg));
    delete[] layer.h_W; delete[] layer.h_mask; delete[] layer.h_theta;
    layer.h_W = new_h_W; layer.h_mask = new_h_mask; layer.h_theta = new_h_theta;
    layer.n_out = new_n_out;
    int weightSize = layer.n_in * layer.n_out, thetaSize = layer.n_out * NUM_CLASSES;
    cudaCheckError(cudaMalloc(&layer.d_W, weightSize * sizeof(float)));
    cudaCheckError(cudaMalloc(&layer.d_mask, weightSize * sizeof(float)));
    cudaCheckError(cudaMalloc(&layer.d_theta, thetaSize * sizeof(float)));
    cudaCheckError(cudaMemcpy(layer.d_W, layer.h_W, weightSize * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(layer.d_mask, layer.h_mask, weightSize * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(layer.d_theta, layer.h_theta, thetaSize * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMalloc(&layer.d_act_pos, layer.n_out * sizeof(float)));
    cudaCheckError(cudaMalloc(&layer.d_act_neg, layer.n_out * sizeof(float)));
  }
}

struct Prediction {
  vector<float> prob;
  int pred;
};

void softmax(const vector<float>& in, vector<float>& out) {
  float maxVal = *max_element(in.begin(), in.end());
  float sum = 0;
  for(size_t i = 0; i < in.size(); i++){
    out[i] = exp(in[i] - maxVal);
    sum += out[i];
  }
  for(size_t i = 0; i < in.size(); i++) out[i] /= sum;
}

int argmax(const vector<float>& v) {
  return distance(v.begin(), max_element(v.begin(), v.end()));
}

Prediction predictSample(const vector<Layer>& layers, float* d_input, int blockSize) {
  int numLayers = layers.size();
  float* d_act = d_input;
  for(int l = 0; l < numLayers; l++){
    int n_in = (l == 0) ? INPUT_SIZE : layers[l-1].n_out;
    int n_out = layers[l].n_out;
    int grid = (n_out + blockSize - 1) / blockSize;
    fusedForwardKernel<<<grid, blockSize>>>(d_act, d_act, layers[l].d_act_pos, layers[l].d_act_neg,
                                              layers[l].d_W, layers[l].d_mask, n_in, n_out);
    cudaDeviceSynchronize();
    d_act = layers[l].d_act_pos;
  }
  int final_n = layers[numLayers-1].n_out;
  float goodness[NUM_CLASSES] = {0};
  int grid = (final_n + blockSize - 1) / blockSize;
  float* d_zero; cudaCheckError(cudaMalloc(&d_zero, final_n * sizeof(float)));
  cudaCheckError(cudaMemset(d_zero, 0, final_n * sizeof(float)));
  float* d_goodness; cudaCheckError(cudaMalloc(&d_goodness, sizeof(float)));
  for(int c = 0; c < NUM_CLASSES; c++){
    cudaCheckError(cudaMemset(d_goodness, 0, sizeof(float)));
    fusedGoodnessKernel<<<grid, blockSize>>>(d_act, d_zero, layers[numLayers-1].d_theta, d_goodness, d_goodness, final_n, c);
    cudaDeviceSynchronize();
    float val; cudaCheckError(cudaMemcpy(&val, d_goodness, sizeof(float), cudaMemcpyDeviceToHost));
    goodness[c] = val;
  }
  cudaCheckError(cudaFree(d_zero)); cudaCheckError(cudaFree(d_goodness));
  vector<float> agg(goodness, goodness + NUM_CLASSES);
  vector<float> prob(NUM_CLASSES, 0);
  softmax(agg, prob);
  int pred = argmax(prob);
  Prediction p = {prob, pred};
  return p;
}

int main(){
  srand(time(NULL));
  ifstream trainImages("mnist_train_images.bin", ios::binary);
  ifstream trainLabels("mnist_train_labels.bin", ios::binary);
  ifstream testImages("mnist_test_images.bin", ios::binary);
  ifstream testLabels("mnist_test_labels.bin", ios::binary);
  if(!trainImages.is_open() || !trainLabels.is_open() || !testImages.is_open() || !testLabels.is_open()){
    cout << "Error opening MNIST binary files" << endl; return -1;
  }
  trainImages.seekg(16, ios::beg); trainLabels.seekg(8, ios::beg);
  testImages.seekg(16, ios::beg); testLabels.seekg(8, ios::beg);
  unsigned char* trainImagesUC = new unsigned char[TRAIN_SIZE * INPUT_SIZE];
  unsigned char* trainLabelsUC = new unsigned char[TRAIN_SIZE];
  unsigned char* testImagesUC  = new unsigned char[TEST_SIZE * INPUT_SIZE];
  unsigned char* testLabelsUC  = new unsigned char[TEST_SIZE];
  trainImages.read((char*)trainImagesUC, TRAIN_SIZE * INPUT_SIZE);
  trainLabels.read((char*)trainLabelsUC, TRAIN_SIZE);
  testImages.read((char*)testImagesUC, TEST_SIZE * INPUT_SIZE);
  testLabels.read((char*)testLabelsUC, TEST_SIZE);
  float* trainImagesF = new float[TRAIN_SIZE * INPUT_SIZE];
  float* testImagesF  = new float[TEST_SIZE * INPUT_SIZE];
  for(int i = 0; i < TRAIN_SIZE * INPUT_SIZE; i++) trainImagesF[i] = ((float)trainImagesUC[i]) / 255.0f;
  for(int i = 0; i < TEST_SIZE * INPUT_SIZE; i++) testImagesF[i] = ((float)testImagesUC[i]) / 255.0f;
  int* trainLabelsI = new int[TRAIN_SIZE];
  int* testLabelsI  = new int[TEST_SIZE];
  for(int i = 0; i < TRAIN_SIZE; i++) trainLabelsI[i] = (int)trainLabelsUC[i];
  for(int i = 0; i < TEST_SIZE; i++) testLabelsI[i] = (int)testLabelsUC[i];
  delete[] trainImagesUC; delete[] trainLabelsUC; delete[] testImagesUC; delete[] testLabelsUC;
  vector<Layer> layers(NUM_LAYERS);
  allocateLayer(layers[0], INPUT_SIZE, INITIAL_NEURONS);
  for(int l = 1; l < NUM_LAYERS; l++) allocateLayer(layers[l], layers[l-1].n_out, INITIAL_NEURONS);
  int blockSize = 256;
  float* d_input; cudaCheckError(cudaMalloc(&d_input, INPUT_SIZE * sizeof(float)));
  float* d_input_corrupt; cudaCheckError(cudaMalloc(&d_input_corrupt, INPUT_SIZE * sizeof(float)));
  float* d_goodness; cudaCheckError(cudaMalloc(&d_goodness, sizeof(float)));
  float* d_maxDelta; cudaCheckError(cudaMalloc(&d_maxDelta, sizeof(float)));
  vector<float*> d_delta_buffers(NUM_LAYERS, nullptr);
  for(int l = 0; l < NUM_LAYERS; l++){
    int totalWeights = layers[l].n_in * layers[l].n_out;
    cudaCheckError(cudaMalloc(&d_delta_buffers[l], totalWeights * sizeof(float)));
  }
  
  // Training loop
  for(int epoch = 0; epoch < EPOCHS; epoch++){
    float totalLoss = 0.0f;
    vector<float> goodnessSum(layers[NUM_LAYERS-1].n_out, 0.0f);
    vector<int> sampleCount(layers[NUM_LAYERS-1].n_out, 0);
    for(int sample = 0; sample < TRAIN_SIZE; sample++){
      cudaCheckError(cudaMemcpy(d_input, &trainImagesF[sample * INPUT_SIZE], INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
      float* corruptImage = new float[INPUT_SIZE];
      for(int i = 0; i < INPUT_SIZE; i++)
        corruptImage[i] = trainImagesF[sample * INPUT_SIZE + i] + ((((float)rand()/RAND_MAX) - 0.5f) * 0.2f);
      cudaCheckError(cudaMemcpy(d_input_corrupt, corruptImage, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
      delete[] corruptImage;
      float* d_act_pos_prev = d_input;
      float* d_act_neg_prev = d_input_corrupt;
      for(int l = 0; l < NUM_LAYERS; l++){
        int n_in = layers[l].n_in, n_out = layers[l].n_out;
        int grid = (n_out + blockSize - 1) / blockSize;
        fusedForwardKernel<<<grid, blockSize>>>(d_act_pos_prev, d_act_neg_prev, layers[l].d_act_pos, layers[l].d_act_neg,
                                                  layers[l].d_W, layers[l].d_mask, n_in, n_out);
        cudaDeviceSynchronize();
        d_act_pos_prev = layers[l].d_act_pos;
        d_act_neg_prev = layers[l].d_act_neg;
      }
      int final_n = layers[NUM_LAYERS-1].n_out;
      int gridFinal = (final_n + blockSize - 1) / blockSize;
      cudaCheckError(cudaMemset(d_goodness, 0, sizeof(float)));
      fusedGoodnessKernel<<<gridFinal, blockSize>>>(layers[NUM_LAYERS-1].d_act_pos, layers[NUM_LAYERS-1].d_act_neg,
                                                    layers[NUM_LAYERS-1].d_theta, d_goodness, d_goodness, final_n, trainLabelsI[sample]);
      cudaDeviceSynchronize();
      float h_goodness = 0.0f;
      cudaCheckError(cudaMemcpy(&h_goodness, d_goodness, sizeof(float), cudaMemcpyDeviceToHost));
      totalLoss += (MARGIN - h_goodness) > 0 ? (MARGIN - h_goodness) : 0.0f;
      
      d_act_pos_prev = d_input; d_act_neg_prev = d_input_corrupt;
      for(int l = 0; l < NUM_LAYERS; l++){
        int n_in = layers[l].n_in, n_out = layers[l].n_out;
        int grid = n_out;
        computeDeltaKernel<<<grid, n_in>>>(d_act_pos_prev, d_act_neg_prev, layers[l].d_act_pos, layers[l].d_act_neg,
                                            layers[l].d_theta, n_in, n_out, trainLabelsI[sample], d_delta_buffers[l]);
        cudaDeviceSynchronize();
        int totalWeights = n_in * n_out;
        thrust::device_vector<float> deltaVec(totalWeights);
        thrust::copy(thrust::device, d_delta_buffers[l], d_delta_buffers[l] + totalWeights, deltaVec.begin());
        thrust::device_vector<int> indices(totalWeights);
        for(int i = 0; i < totalWeights; i++)
          indices[i] = i;
        thrust::sort_by_key(deltaVec.begin(), deltaVec.end(), indices.begin(), CompareAbs());
        int topK = (int)(TOPK_FRAC * totalWeights);
        thrust::device_vector<int> d_topK(indices.begin(), indices.begin() + topK);
        int threads = 256;
        int blocks = (topK + threads - 1) / threads;
        topKWeightUpdateKernel<<<blocks, threads>>>(thrust::raw_pointer_cast(d_topK.data()),
                                                     thrust::raw_pointer_cast(deltaVec.data()),
                                                     layers[l].d_W, topK);
        cudaDeviceSynchronize();
        int gridTheta = (n_out + blockSize - 1) / blockSize;
        float loss = (MARGIN - h_goodness) > 0 ? (MARGIN - h_goodness) : 0.0f;
        updateThetaKernel<<<gridTheta, blockSize>>>(layers[l].d_theta, layers[l].d_act_pos, layers[l].d_act_neg, loss, trainLabelsI[sample], n_out);
        cudaDeviceSynchronize();
        d_act_pos_prev = layers[l].d_act_pos;
        d_act_neg_prev = layers[l].d_act_neg;
      }
      int finalLayer = NUM_LAYERS - 1;
      int n_final = layers[finalLayer].n_out;
      float* h_final = new float[n_final];
      cudaCheckError(cudaMemcpy(h_final, layers[finalLayer].d_act_pos, n_final * sizeof(float), cudaMemcpyDeviceToHost));
      for(int j = 0; j < n_final; j++){
        goodnessSum[j] += h_final[j];
        sampleCount[j]++;
      }
      delete[] h_final;
    }
    cout << "Epoch " << epoch+1 << ": Avg Loss = " << totalLoss / ((float)TRAIN_SIZE) << endl;
    vector<float> neuronGoodnessAvg(layers[NUM_LAYERS-1].n_out, 0.0f);
    for(int j = 0; j < layers[NUM_LAYERS-1].n_out; j++){
      neuronGoodnessAvg[j] = (sampleCount[j] > 0 ? goodnessSum[j] / sampleCount[j] : 0);
    }
    dynamicCapacityAllocation(layers[NUM_LAYERS-1], nullptr, neuronGoodnessAvg);
  }
  int correct = 0;
  for(int sample = 0; sample < TEST_SIZE; sample++){
    Prediction p = predictSample(layers, d_input, blockSize);
    if(p.pred == testLabelsI[sample]) correct++;
  }
  cout << "Test Accuracy: " << ((float)correct) / TEST_SIZE * 100 << "%" << endl;
  for(int l = 0; l < NUM_LAYERS; l++) freeLayer(layers[l]);
  cudaCheckError(cudaFree(d_input)); cudaCheckError(cudaFree(d_input_corrupt));
  cudaCheckError(cudaFree(d_goodness)); cudaCheckError(cudaFree(d_maxDelta));
  for(int l = 0; l < NUM_LAYERS; l++) cudaCheckError(cudaFree(d_delta_buffers[l]));
  cudaDeviceReset();
  delete[] trainImagesF; delete[] testImagesF; delete[] trainLabelsI; delete[] testLabelsI;
  return 0;
}

