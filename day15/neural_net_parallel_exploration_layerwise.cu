#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h> // For memcpy

#define INPUT_SIZE    3
#define LAYER1_SIZE   5
#define LAYER2_SIZE   6
#define LAYER3_SIZE   7
#define OUTPUT_SIZE   7   // softmax output (7 classes)

// Total number of parameters per candidate:
// W1: 3x5, b1: 5, W2: 5x6, b2: 6, W3: 6x7, b3: 7.
#define PARAMS_PER_SET (INPUT_SIZE*LAYER1_SIZE + LAYER1_SIZE + \
                        LAYER1_SIZE*LAYER2_SIZE + LAYER2_SIZE + \
                        LAYER2_SIZE*LAYER3_SIZE + LAYER3_SIZE)

#define NUM_SAMPLES   10000   // number of synthetic samples
#define NUM_CANDIDATES 200     // INCREASED: Number of candidate parameter sets
#define ITERATIONS  100        // INCREASED: Number of training iterations

// Device activation functions
__device__ float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

__device__ void softmax(const float *input, float *output, int length) {
    float max_val = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}

// Forward pass: computes the network output given an input vector and a candidate parameter set.
// Layer-wise version for targeted optimization
__device__ void forward_pass_layerwise(const float *input, const float *params, float *output, int layer_id) {
    const float *W1 = params;
    const float *b1 = W1 + INPUT_SIZE * LAYER1_SIZE;
    const float *W2 = b1 + LAYER1_SIZE;
    const float *b2 = W2 + LAYER1_SIZE * LAYER2_SIZE;
    const float *W3 = b2 + LAYER2_SIZE;
    const float *b3 = W3 + LAYER2_SIZE * LAYER3_SIZE;

    float layer1[LAYER1_SIZE];
    for (int j = 0; j < LAYER1_SIZE; j++) {
        float sum = b1[j];
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += input[i] * W1[i * LAYER1_SIZE + j];
        }
        layer1[j] = relu(sum);
    }

    float layer2[LAYER2_SIZE];
    for (int j = 0; j < LAYER2_SIZE; j++) {
        float sum = b2[j];
        for (int i = 0; i < LAYER1_SIZE; i++) {
            sum += layer1[i] * W2[i * LAYER2_SIZE + j];
        }
        layer2[j] = relu(sum);
    }

    float layer3[LAYER3_SIZE];
    for (int j = 0; j < LAYER3_SIZE; j++) {
        float sum = b3[j];
        for (int i = 0; i < LAYER2_SIZE; i++) {
            sum += layer2[i] * W3[i * LAYER3_SIZE + j];
        }
        layer3[j] = sum; // no activation; softmax follows
    }
    softmax(layer3, output, LAYER3_SIZE);
}


// Original forward pass (for final prediction - uses all parameters at once)
__device__ void forward_pass(const float *input, const float *params, float *output) {
     forward_pass_layerwise(input, params, output, -1); // Just call layerwise with dummy layer_id
}


// Kernel to evaluate one candidate parameter set over a batch of samples.
// Modified to take layer_id for layer-wise optimization.
__global__ void evaluate_parameter_set(const float *input_data,
                                       const float *label_data,
                                       int num_samples,
                                       const float *param_sets, // All candidate sets are passed now
                                       int layer_id,           // ID of the layer being optimized (0, 1, or 2)
                                       float *losses) {
    int candidate_idx = blockIdx.x;
    const float *params = param_sets + candidate_idx * PARAMS_PER_SET;

    extern __shared__ float s_data[];
    float *loss_shared = s_data;

    float local_loss = 0.0f;
    for (int n = threadIdx.x; n < num_samples; n += blockDim.x) {
        const float *input = input_data + n * INPUT_SIZE;
        const float *label = label_data + n * OUTPUT_SIZE;
        float output[OUTPUT_SIZE];
        forward_pass_layerwise(input, params, output, layer_id); // Use layer-wise forward pass
        float sample_loss = 0.0f;
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            float pred = fmaxf(output[k], 1e-8f);
            sample_loss -= label[k] * logf(pred);
        }
        local_loss += sample_loss;
    }

    // Store each thread's local loss into shared memory
    loss_shared[threadIdx.x] = local_loss;
    __syncthreads();

    // Parallel reduction (tree-based)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            loss_shared[threadIdx.x] += loss_shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Only thread 0 writes the result for the block
    if (threadIdx.x == 0) {
        losses[candidate_idx] = loss_shared[0] / num_samples;
    }
}

// Kernel to compute predictions using a candidate parameter set.
__global__ void predict_kernel(const float *input_data,
                               const float *params,
                               int num_samples,
                               int *predictions) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < num_samples) {
        const float *input = input_data + n * INPUT_SIZE;
        float output[OUTPUT_SIZE];
        forward_pass(input, params, output); // Use original forward pass for prediction
        int best = 0;
        float best_val = output[0];
        for (int i = 1; i < OUTPUT_SIZE; i++) {
            if (output[i] > best_val) {
                best_val = output[i];
                best = i;
            }
        }
        predictions[n] = best;
    }
}

// Host helper: computes the index of the maximum element (used to decode one-hot labels).
int argmax(const float *vec, int len) {
    int best = 0;
    float best_val = vec[0];
    for (int i = 1; i < len; i++) {
        if (vec[i] > best_val) {
            best_val = vec[i];
            best = i;
        }
    }
    return best;
}

// Generates a synthetic dataset.
// Each sample’s features are drawn uniformly from [0,1],
// and the label is determined by averaging the features and mapping to one of OUTPUT_SIZE classes.
void generate_dataset(float *h_input, float *h_labels) {
    for (int n = 0; n < NUM_SAMPLES; n++) {
        float sum = 0.0f;
        for (int i = 0; i < INPUT_SIZE; i++) {
            float val = (float)rand() / (float)RAND_MAX;
            h_input[n * INPUT_SIZE + i] = val;
            sum += val;
        }
        int label_idx = (int)((sum / INPUT_SIZE) * OUTPUT_SIZE);
        if (label_idx >= OUTPUT_SIZE) label_idx = OUTPUT_SIZE - 1;
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            h_labels[n * OUTPUT_SIZE + k] = (k == label_idx) ? 1.0f : 0.0f;
        }
    }
}

// Host helper: Xavier initialization for a given layer size.
void xavier_init(float *weights, int rows, int cols) {
    float range = sqrtf(6.0f / (rows + cols)); // Xavier range
    for (int i = 0; i < rows * cols; i++) {
        weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * range;
    }
}


// Refined candidate update function for LAYER-WISE optimization.
void update_candidates_layerwise(float *h_candidates, int best_idx, int layer_id, float decay_factor) {
    float *best = h_candidates + best_idx * PARAMS_PER_SET;
    for (int cand = 0; cand < NUM_CANDIDATES; cand++) {
        if (cand == best_idx) continue;
        float *current = h_candidates + cand * PARAMS_PER_SET;

        float noise_range = 0.002f * decay_factor;
        float noise;

        if (layer_id == 0) { // Layer 1
            float *W1_best = best;
            float *W1_current = current;
            for (int i = 0; i < INPUT_SIZE * LAYER1_SIZE; ++i) {
                noise = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * noise_range;
                W1_current[i] = W1_best[i] + noise;
            }
            float *b1_best = W1_best + INPUT_SIZE * LAYER1_SIZE;
            float *b1_current = W1_current + INPUT_SIZE * LAYER1_SIZE;
            for (int i = 0; i < LAYER1_SIZE; ++i) {
                noise = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * noise_range;
                b1_current[i] = b1_best[i] + noise;
            }
            // Keep Layer 2 and Layer 3 parameters the same as the best candidate.
            memcpy(current + INPUT_SIZE * LAYER1_SIZE + LAYER1_SIZE, best + INPUT_SIZE * LAYER1_SIZE + LAYER1_SIZE, PARAMS_PER_SET - (INPUT_SIZE * LAYER1_SIZE + LAYER1_SIZE) * sizeof(float));

        } else if (layer_id == 1) { // Layer 2
            // Keep Layer 1 parameters the same as best
            memcpy(current, best, (INPUT_SIZE * LAYER1_SIZE + LAYER1_SIZE) * sizeof(float));
            float *W2_best = best + INPUT_SIZE * LAYER1_SIZE + LAYER1_SIZE;
            float *W2_current = current + INPUT_SIZE * LAYER1_SIZE + LAYER1_SIZE;
            for (int i = 0; i < LAYER1_SIZE * LAYER2_SIZE; ++i) {
                noise = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * noise_range;
                W2_current[i] = W2_best[i] + noise;
            }
            float *b2_best = W2_best + LAYER1_SIZE * LAYER2_SIZE;
            float *b2_current = current + INPUT_SIZE * LAYER1_SIZE + LAYER1_SIZE + LAYER1_SIZE * LAYER2_SIZE;
            for (int i = 0; i < LAYER2_SIZE; ++i) {
                noise = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * noise_range;
                b2_current[i] = b2_best[i] + noise;
            }
            // Keep Layer 3 params same as best
             memcpy(current + INPUT_SIZE * LAYER1_SIZE + LAYER1_SIZE + LAYER1_SIZE * LAYER2_SIZE + LAYER2_SIZE, best + INPUT_SIZE * LAYER1_SIZE + LAYER1_SIZE + LAYER1_SIZE * LAYER2_SIZE + LAYER2_SIZE, PARAMS_PER_SET - (INPUT_SIZE * LAYER1_SIZE + LAYER1_SIZE + LAYER1_SIZE * LAYER2_SIZE + LAYER2_SIZE) * sizeof(float));

        } else if (layer_id == 2) { // Layer 3
             // Keep Layer 1 and Layer 2 parameters same as best
            memcpy(current, best, (INPUT_SIZE * LAYER1_SIZE + LAYER1_SIZE + LAYER1_SIZE * LAYER2_SIZE + LAYER2_SIZE) * sizeof(float));
            float *W3_best = best + INPUT_SIZE * LAYER1_SIZE + LAYER1_SIZE + LAYER1_SIZE * LAYER2_SIZE + LAYER2_SIZE;
            float *W3_current = current + INPUT_SIZE * LAYER1_SIZE + LAYER1_SIZE + LAYER1_SIZE * LAYER2_SIZE + LAYER2_SIZE;
            for (int i = 0; i < LAYER2_SIZE * LAYER3_SIZE; ++i) {
                noise = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * noise_range;
                W3_current[i] = W3_best[i] + noise;
            }
            float *b3_best = W3_best + LAYER2_SIZE * LAYER3_SIZE;
            float *b3_current = current + INPUT_SIZE * LAYER1_SIZE + LAYER1_SIZE + LAYER1_SIZE * LAYER2_SIZE + LAYER2_SIZE + LAYER2_SIZE * LAYER3_SIZE;
            for (int i = 0; i < LAYER3_SIZE; ++i) {
                noise = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * noise_range;
                b3_current[i] = b3_best[i] + noise;
            }
        }
    }
}

int main() {
    srand(time(NULL));

    // Allocate and generate host dataset.
    float *h_input = (float*)malloc(NUM_SAMPLES * INPUT_SIZE * sizeof(float));
    float *h_labels = (float*)malloc(NUM_SAMPLES * OUTPUT_SIZE * sizeof(float));
    generate_dataset(h_input, h_labels);

    // Allocate host candidate parameter sets and Xavier initialize them.
    float *h_candidates = (float*)malloc(NUM_CANDIDATES * PARAMS_PER_SET * sizeof(float));
    for (int cand = 0; cand < NUM_CANDIDATES; cand++) {
        float *params = h_candidates + cand * PARAMS_PER_SET;
        float *W1 = params;
        float *b1 = W1 + INPUT_SIZE * LAYER1_SIZE;
        float *W2 = b1 + LAYER1_SIZE;
        float *b2 = W2 + LAYER1_SIZE * LAYER2_SIZE;
        float *W3 = b2 + LAYER2_SIZE;
        float *b3 = W3 + LAYER2_SIZE * LAYER3_SIZE;

        xavier_init(W1, INPUT_SIZE, LAYER1_SIZE);
        for(int i=0; i<LAYER1_SIZE; ++i) b1[i] = 0.0f; // Initialize biases to 0

        xavier_init(W2, LAYER1_SIZE, LAYER2_SIZE);
        for(int i=0; i<LAYER2_SIZE; ++i) b2[i] = 0.0f;

        xavier_init(W3, LAYER2_SIZE, LAYER3_SIZE);
        for(int i=0; i<LAYER3_SIZE; ++i) b3[i] = 0.0f;
    }

    // Allocate device memory.
    float *d_input, *d_labels, *d_candidates, *d_losses;
    cudaMalloc(&d_input, NUM_SAMPLES * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_labels, NUM_SAMPLES * OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_candidates, NUM_CANDIDATES * PARAMS_PER_SET * sizeof(float));
    cudaMalloc(&d_losses, NUM_CANDIDATES * sizeof(float));

    // Copy dataset and initial candidates to device.
    cudaMemcpy(d_input, h_input, NUM_SAMPLES * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, NUM_SAMPLES * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_candidates, h_candidates, NUM_CANDIDATES * PARAMS_PER_SET * sizeof(float), cudaMemcpyHostToDevice);

    // Set up CUDA events for timing.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalKernelTime = 0.0f;
    float decay_factor = 0.8f;

    // Layer-wise iterative exploration loop
    for (int layer_iter = 0; layer_iter < 3; ++layer_iter) { // Iterate through layers (0, 1, 2)
        printf("--- Optimizing Layer %d ---\n", layer_iter + 1);
        for (int iter = 0; iter < ITERATIONS; iter++) { // Iterations per layer
            // Each block evaluates one candidate, optimizing parameters of 'layer_iter' layer.
            int threadsPerBlock = 256;
            cudaEventRecord(start, 0);
            size_t shared_mem_size = threadsPerBlock * sizeof(float);
            evaluate_parameter_set<<<NUM_CANDIDATES, threadsPerBlock, shared_mem_size>>>(d_input, d_labels, NUM_SAMPLES, d_candidates, layer_iter, d_losses); // Pass layer_iter
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float elapsed;
            cudaEventElapsedTime(&elapsed, start, stop);
            totalKernelTime += elapsed;

            // Copy candidate losses back to host.
            float h_losses[NUM_CANDIDATES];
            cudaMemcpy(h_losses, d_losses, NUM_CANDIDATES * sizeof(float), cudaMemcpyDeviceToHost);

            // Select candidate with the lowest loss.
            int best_idx = 0;
            float best_loss = h_losses[0];
            for (int cand = 1; cand < NUM_CANDIDATES; cand++) {
                if (h_losses[cand] < best_loss) {
                    best_loss = h_losses[cand];
                    best_idx = cand;
                }
            }
            printf("  Layer %d, Iteration %d: Best Loss = %f (Candidate %d)\n", layer_iter + 1, iter, best_loss, best_idx);

            // Refine candidates for the current layer.
            update_candidates_layerwise(h_candidates, best_idx, layer_iter, decay_factor); // Pass layer_iter
            decay_factor *= 0.9f;

            // Copy updated candidate parameters back to device.
            cudaMemcpy(d_candidates, h_candidates, NUM_CANDIDATES * PARAMS_PER_SET * sizeof(float), cudaMemcpyHostToDevice);
        }
        decay_factor = 1.0f; // Reset decay for next layer optimization (optional, can experiment with keeping decay)
    }


    printf("Average evaluation kernel time: %f ms\n", totalKernelTime / (ITERATIONS * 3)); // Adjusted average time

    // Select final best candidate.
    float final_losses[NUM_CANDIDATES];
    cudaMemcpy(final_losses, d_losses, NUM_CANDIDATES * sizeof(float), cudaMemcpyDeviceToHost);
    int best_candidate_idx = 0;
    double best_loss = final_losses[0];
    for (int cand = 1; cand < NUM_CANDIDATES; cand++) {
        if (final_losses[cand] < best_loss) {
            best_loss = final_losses[cand];
            best_candidate_idx = cand;
        }
    }
    printf("Final selected candidate: %d with loss %f\n", best_candidate_idx, best_loss);

    // Compute predictions using the best candidate.
    int *d_predictions;
    cudaMalloc(&d_predictions, NUM_SAMPLES * sizeof(int));
    int threads = 256;
    int blocks = (NUM_SAMPLES + threads - 1) / threads;
    float *d_best_params = d_candidates + best_candidate_idx * PARAMS_PER_SET;
    predict_kernel<<<blocks, threads>>>(d_input, d_best_params, NUM_SAMPLES, d_predictions);

    // Copy predictions back to host and calculate accuracy.
    int *h_predictions = (int*)malloc(NUM_SAMPLES * sizeof(int));
    cudaMemcpy(h_predictions, d_predictions, NUM_SAMPLES * sizeof(int), cudaMemcpyDeviceToHost);

    int correct = 0;
    for (int n = 0; n < NUM_SAMPLES; n++) {
        int true_label = argmax(h_labels + n * OUTPUT_SIZE, OUTPUT_SIZE);
        if (h_predictions[n] == true_label)
            correct++;
    }
    float accuracy = (float)correct / NUM_SAMPLES * 100.0f;
    printf("Final accuracy on training data: %f%%\n", accuracy);

    // Cleanup.
    cudaFree(d_input);
    cudaFree(d_labels);
    cudaFree(d_candidates);
    cudaFree(d_losses);
    cudaFree(d_predictions);
    free(h_input);
    free(h_labels);
    free(h_candidates);
    free(h_predictions);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

