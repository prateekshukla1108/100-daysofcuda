#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <stdio.h>
#include <time.h>

using namespace std;

// Utility macro for error checking.
#define CUDA_CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Global constants
const int INPUT_SIZE = 784;      // 28x28 MNIST images
const int NUM_CLASSES = 10;      // Number of classes
const float SPARSITY = 0.1f;       // 10% connectivity
const float TOPK_THRESHOLD = 0.001f;
const float LEARNING_RATE = 0.001f;
const int BATCH_SIZE = 32;
const int NUM_EPOCHS = 10;

// ----------------------------------
// CUDA Kernels
// ----------------------------------

// Forward pass kernel: each thread computes one neuron's activation.
// activation = ReLU( sum_i mask * W * input )
__global__ void forwardKernel(const float* input, float* activations, 
                              const float* W, const float* mask,
                              int n_in, int n_out)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j < n_out) {
        float sum = 0.0f;
        for (int i = 0; i < n_in; i++) {
            float m = mask[j * n_in + i];
            sum += m * W[j * n_in + i] * input[i];
        }
        activations[j] = (sum > 0.0f ? sum : 0.0f);  // ReLU activation
    }
}

// Kernel to compute “goodness” for a layer for a given class label.
// Each thread computes activation * theta for one neuron; the block sums via atomicAdd.
__global__ void computeGoodnessKernel(const float* activations, const float* theta,
                                      float* goodness, int n_out, int label)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j < n_out) {
        float val = activations[j] * theta[label * n_out + j];
        atomicAdd(goodness, val);
    }
}

// Weight update kernel: updates each weight based on the difference between
// positive and negative passes. Only weights with delta magnitude above TOPK_THRESHOLD are updated.
__global__ void updateWeightsKernel(const float* input_pos, const float* input_neg,
                                    const float* act_pos, const float* act_neg,
                                    float* W, const float* mask,
                                    const float* theta, int n_in, int n_out, int label)
{
    int j = blockIdx.x;       // neuron index (output)
    int i = threadIdx.x;      // input index
    if(j < n_out && i < n_in) {
        float contrib_pos = (act_pos[j] > 0.0f) ? input_pos[i] : 0.0f;
        float contrib_neg = (act_neg[j] > 0.0f) ? input_neg[i] : 0.0f;
        float delta = LEARNING_RATE * theta[label * n_out + j] * (contrib_pos - contrib_neg);
        if(fabsf(delta) > TOPK_THRESHOLD) {
            if(mask[j * n_in + i] > 0.5f) {
                W[j * n_in + i] += delta;
            }
        }
    }
}

// ----------------------------------
// Host-side Utility Functions
// ----------------------------------

// Initialize an array with random floats in approximately [-0.1, 0.1].
void initializeRandom(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = ((float)rand()/RAND_MAX - 0.5f) / 5.0f;
    }
}

// Initialize a binary mask with a given sparsity.
void initializeMask(float* mask, int size, float sparsity) {
    for (int i = 0; i < size; i++) {
        float r = (float)rand()/RAND_MAX;
        mask[i] = (r < sparsity) ? 1.0f : 0.0f;
    }
}

// ----------------------------------
// DFFLayer Class
// ----------------------------------

class DFFLayer {
public:
    int n_in, n_out;
    float *d_W, *d_mask, *d_theta;
    float *d_activations; // Temporary buffer for activations

    DFFLayer(int n_in_, int n_out_) : n_in(n_in_), n_out(n_out_) {
        int weightSize = n_out * n_in;
        int thetaSize = n_out * NUM_CLASSES;
        CUDA_CHECK_ERROR(cudaMalloc(&d_W, weightSize * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMalloc(&d_mask, weightSize * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMalloc(&d_theta, thetaSize * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMalloc(&d_activations, n_out * sizeof(float)));

        float* h_W = new float[weightSize];
        float* h_mask = new float[weightSize];
        float* h_theta = new float[thetaSize];

        initializeRandom(h_W, weightSize);
        initializeMask(h_mask, weightSize, SPARSITY);
        initializeRandom(h_theta, thetaSize);

        CUDA_CHECK_ERROR(cudaMemcpy(d_W, h_W, weightSize*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(d_mask, h_mask, weightSize*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(d_theta, h_theta, thetaSize*sizeof(float), cudaMemcpyHostToDevice));

        delete[] h_W;
        delete[] h_mask;
        delete[] h_theta;
    }

    ~DFFLayer() {
        cudaFree(d_W);
        cudaFree(d_mask);
        cudaFree(d_theta);
        cudaFree(d_activations);
    }

    // Forward pass for one sample (input assumed to be on device).
    void forward(const float* d_input) {
        int blockSize = 256;
        int gridSize = (n_out + blockSize - 1) / blockSize;
        forwardKernel<<<gridSize, blockSize>>>(d_input, d_activations, d_W, d_mask, n_in, n_out);
        CUDA_CHECK_ERROR(cudaPeekAtLastError());
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }

    // Compute layer goodness for a given label.
    float computeGoodness(const int label) {
        float h_goodness = 0.0f;
        float* d_goodness;
        CUDA_CHECK_ERROR(cudaMalloc(&d_goodness, sizeof(float)));
        CUDA_CHECK_ERROR(cudaMemcpy(d_goodness, &h_goodness, sizeof(float), cudaMemcpyHostToDevice));
        int blockSize = 256;
        int gridSize = (n_out + blockSize - 1) / blockSize;
        computeGoodnessKernel<<<gridSize, blockSize>>>(d_activations, d_theta, d_goodness, n_out, label);
        CUDA_CHECK_ERROR(cudaPeekAtLastError());
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        CUDA_CHECK_ERROR(cudaMemcpy(&h_goodness, d_goodness, sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d_goodness);
        return h_goodness;
    }

    // Update weights based on a positive and negative pass for one sample.
    void updateWeights(const float* d_input_pos, const float* d_input_neg,
                       const float* d_act_pos, const float* d_act_neg, int label) {
        dim3 grid(n_out, 1, 1);
        dim3 block(n_in, 1, 1);
        updateWeightsKernel<<<grid, block>>>(d_input_pos, d_input_neg, d_act_pos, d_act_neg,
                                               d_W, d_mask, d_theta, n_in, n_out, label);
        CUDA_CHECK_ERROR(cudaPeekAtLastError());
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }

    // Return pointer to current activations.
    float* getActivations() {
        return d_activations;
    }

    // Basic dynamic capacity adjustment: reinitialize neurons with low "goodness".
    // (In a real system, one might add new neurons and adjust memory dynamically.)
    void adjustCapacity(const int label, float threshold) {
        float* h_activations = new float[n_out];
        CUDA_CHECK_ERROR(cudaMemcpy(h_activations, d_activations, n_out*sizeof(float), cudaMemcpyDeviceToHost));
        
        float* h_theta = new float[n_out];
        CUDA_CHECK_ERROR(cudaMemcpy(h_theta, d_theta + label * n_out, n_out*sizeof(float), cudaMemcpyDeviceToHost));

        int weightRowSize = n_in;
        float* h_W = new float[n_out * n_in];
        CUDA_CHECK_ERROR(cudaMemcpy(h_W, d_W, n_out * n_in * sizeof(float), cudaMemcpyDeviceToHost));
        
        bool updated = false;
        for (int j = 0; j < n_out; j++) {
            float goodness = h_activations[j] * h_theta[j];
            if (goodness < threshold) {
                // Reinitialize weights for this neuron.
                for (int i = 0; i < n_in; i++) {
                    h_W[j * n_in + i] = ((float)rand()/RAND_MAX - 0.5f) / 5.0f;
                }
                updated = true;
            }
        }
        if(updated) {
            CUDA_CHECK_ERROR(cudaMemcpy(d_W, h_W, n_out * n_in * sizeof(float), cudaMemcpyHostToDevice));
        }
        delete[] h_activations;
        delete[] h_theta;
        delete[] h_W;
    }
};

// ----------------------------------
// DFFNetwork Class
// ----------------------------------

class DFFNetwork {
public:
    vector<DFFLayer*> layers;
    int numLayers;

    // layerSizes: vector of neuron counts (including input size as first element)
    DFFNetwork(const vector<int>& layerSizes) {
        numLayers = layerSizes.size() - 1;
        for (int i = 0; i < numLayers; i++) {
            layers.push_back(new DFFLayer(layerSizes[i], layerSizes[i+1]));
        }
    }

    ~DFFNetwork() {
        for (auto layer : layers)
            delete layer;
    }

    // Forward pass through all layers for one sample.
    void forward(const float* d_input) {
        const float* current_input = d_input;
        for (int i = 0; i < numLayers; i++) {
            layers[i]->forward(current_input);
            current_input = layers[i]->getActivations();
        }
    }

    // Update weights layer-wise for one sample using positive and negative inputs.
    void updateWeights(const float* d_input_pos, const float* d_input_neg, int label) {
        float* d_in_pos = const_cast<float*>(d_input_pos);
        float* d_in_neg = const_cast<float*>(d_input_neg);
        for (int i = 0; i < numLayers; i++) {
            layers[i]->forward(d_in_pos);
            float* d_act_pos = layers[i]->getActivations();
            layers[i]->forward(d_in_neg);
            float* d_act_neg = layers[i]->getActivations();

            layers[i]->updateWeights(d_in_pos, d_in_neg, d_act_pos, d_act_neg, label);
            d_in_pos = d_act_pos;
            d_in_neg = d_act_neg;
        }
    }

    // Aggregate goodness from all layers for prediction.
    float predictGoodness(const int label) {
        float totalGoodness = 0.0f;
        for (int i = 0; i < numLayers; i++) {
            totalGoodness += layers[i]->computeGoodness(label);
        }
        return totalGoodness;
    }

    // Adjust capacity (dynamically reinitialize weak neurons) for all layers.
    void adjustCapacityAll(const int label, float threshold) {
        for (int i = 0; i < numLayers; i++) {
            layers[i]->adjustCapacity(label, threshold);
        }
    }
};

// ----------------------------------
// Data Simulation Utilities
// ----------------------------------

// Simulate a batch of MNIST-like data (here, random data for demonstration).
void simulateMNISTBatch(vector<vector<float>>& inputs, vector<int>& labels, int batchSize) {
    inputs.resize(batchSize, vector<float>(INPUT_SIZE));
    labels.resize(batchSize);
    for (int b = 0; b < batchSize; b++) {
        for (int i = 0; i < INPUT_SIZE; i++) {
            inputs[b][i] = (float)rand()/RAND_MAX;
        }
        labels[b] = rand() % NUM_CLASSES;
    }
}

// Create a corrupted version of an input (for the negative pass).
void corruptInput(const vector<float>& input, vector<float>& output) {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = input[i] + (((float)rand()/RAND_MAX - 0.5f) * 0.2f);
    }
}

// ----------------------------------
// Main Training Loop
// ----------------------------------

int main() {
    srand(time(NULL));

    // Define network architecture: Input -> Hidden1 -> Hidden2
    vector<int> layerSizes = {INPUT_SIZE, 128, 64};
    DFFNetwork network(layerSizes);

    // Allocate device memory for a single sample (inputs are copied per sample).
    float *d_input, *d_input_corrupt;
    CUDA_CHECK_ERROR(cudaMalloc(&d_input, INPUT_SIZE * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_input_corrupt, INPUT_SIZE * sizeof(float)));

    // Training loop over epochs and batches.
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        cout << "Epoch " << epoch+1 << "/" << NUM_EPOCHS << endl;
        vector<vector<float>> batchInputs;
        vector<int> batchLabels;
        simulateMNISTBatch(batchInputs, batchLabels, BATCH_SIZE);

        for (int b = 0; b < BATCH_SIZE; b++) {
            // Copy current sample to device.
            CUDA_CHECK_ERROR(cudaMemcpy(d_input, batchInputs[b].data(), INPUT_SIZE*sizeof(float), cudaMemcpyHostToDevice));

            // Generate corrupted input for negative pass.
            vector<float> corrupted;
            corruptInput(batchInputs[b], corrupted);
            CUDA_CHECK_ERROR(cudaMemcpy(d_input_corrupt, corrupted.data(), INPUT_SIZE*sizeof(float), cudaMemcpyHostToDevice));

            int label = batchLabels[b];

            // Forward pass on positive sample (for logging/prediction purposes).
            network.forward(d_input);
            // Update weights using both positive and negative passes.
            network.updateWeights(d_input, d_input_corrupt, label);

            // For demonstration, compute and log the aggregated goodness.
            float goodness = network.predictGoodness(label);
            if (b % 10 == 0) {
                cout << "  Sample " << b << " (Label: " << label << ") Goodness: " << goodness << endl;
            }
        }

        // After each epoch, adjust capacity across layers.
        // (Here we simply reinitialize neurons with low goodness using a fixed threshold.
        //  In a production system, this routine would be more sophisticated.)
        network.adjustCapacityAll(/*label*/0, 0.1f);

        // In practice, you would also evaluate validation performance here.
    }

    cudaFree(d_input);
    cudaFree(d_input_corrupt);
    cout << "Training completed." << endl;
    return 0;
}

