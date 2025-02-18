#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define IMG_WIDTH 28
#define IMG_HEIGHT 28
#define IMG_SIZE (IMG_WIDTH * IMG_HEIGHT)
#define NUM_CLASSES 10
#define BATCH_SIZE 64
#define EPOCHS 10
#define LR 0.01f
#define CONV1_OUT 6
#define CONV1_K 5
#define CONV1_OUT_SIZE (IMG_WIDTH - CONV1_K + 1)
#define POOL1_S 2
#define POOL1_OUT_SIZE (CONV1_OUT_SIZE / POOL1_S)
#define CONV2_OUT 16
#define CONV2_K 5
#define CONV2_OUT_SIZE (POOL1_OUT_SIZE - CONV2_K + 1)
#define POOL2_S 2
#define POOL2_OUT_SIZE (CONV2_OUT_SIZE / POOL2_S)
#define FC1_IN (CONV2_OUT * POOL2_OUT_SIZE * POOL2_OUT_SIZE)
#define FC1_OUT 120
#define FC2_OUT 84

// Utility functions for loading data remain unchanged.
float* loadImages(const char *fname, int count, int imSize) {
    std::ifstream file(fname, std::ios::binary);
    if(!file){ std::cerr << "Error opening " << fname << std::endl; exit(1); }
    unsigned char *byte_data = (unsigned char*)malloc(count * imSize * sizeof(unsigned char));
    if (!byte_data) { std::cerr << "Memory allocation error in loadImages" << std::endl; exit(1); }
    file.read(reinterpret_cast<char*>(byte_data), count * imSize * sizeof(unsigned char));
    float *data = (float*)malloc(count * imSize * sizeof(float));
    if (!data) { std::cerr << "Memory allocation error in loadImages" << std::endl; exit(1); }
    for (int i = 0; i < count * imSize; i++) data[i] = byte_data[i] / 255.0f;
    free(byte_data);
    file.close();
    return data;
}

int* loadLabels(const char *fname, int count) {
    std::ifstream file(fname, std::ios::binary);
    if(!file){ std::cerr << "Error opening " << fname << std::endl; exit(1); }
    unsigned char *byte_data = (unsigned char*)malloc(count * sizeof(unsigned char));
    if (!byte_data) { std::cerr << "Memory allocation error in loadLabels" << std::endl; exit(1); }
    file.read(reinterpret_cast<char*>(byte_data), count * sizeof(unsigned char));
    int *data = (int*)malloc(count * sizeof(int));
    if (!data) { std::cerr << "Memory allocation error in loadLabels" << std::endl; exit(1); }
    for (int i = 0; i < count; i++) data[i] = byte_data[i];
    free(byte_data);
    file.close();
    return data;
}

/*
 * conv1_fused_kernel:
 *  - Performs convolution with CONV1_K x CONV1_K kernels over the input image.
 *  - Uses valid convolution with no explicit padding (bounds are checked against IMG_WIDTH/IMG_HEIGHT).
 *  - Applies ReLU (implemented via fmaxf(0.0f, sum)) and then performs 2x2 max pooling.
 *  - Saves the argmax index for pooling and also stores a ReLU activation mask (1 if the convolution result was > 0, else 0)
 */
__global__ void conv1_fused_kernel(const float *in, const float *w, const float *b, 
                                     float *out, int *argmax, int *relu_mask, int batch) {
    int bidx = blockIdx.z;
    if(bidx >= batch) return;
    int chan = blockIdx.x;
    int ox = blockIdx.y * blockDim.x + threadIdx.x;
    int oy = threadIdx.y;
    if (ox < POOL1_OUT_SIZE && oy < POOL1_OUT_SIZE) {
        float mval = -1e20f;
        int max_idx = 0; // Index within the pooling window
        int active_flag = 0;
        // Iterate over the pooling window
        for (int i = 0; i < POOL1_S; i++) {
            for (int j = 0; j < POOL1_S; j++) {
                int cx = ox * POOL1_S + i;
                int cy = oy * POOL1_S + j;
                float sum = b[chan];
                // Convolution over a CONV1_K x CONV1_K kernel
                for (int m = 0; m < CONV1_K; m++) {
                    for (int n = 0; n < CONV1_K; n++) {
                        int ix = cx + m - CONV1_K / 2;
                        int iy = cy + n - CONV1_K / 2;
                        // Check against full image dimensions
                        if (ix >= 0 && ix < IMG_WIDTH && iy >= 0 && iy < IMG_HEIGHT) {
                            sum += in[bidx * IMG_SIZE + iy * IMG_WIDTH + ix] *
                                   w[chan * CONV1_K * CONV1_K + m * CONV1_K + n];
                        }
                    }
                }
                // Apply ReLU activation
                float act = fmaxf(0.0f, sum);
                int is_active = (sum > 0.0f) ? 1 : 0;
                // Max pooling: select the maximum activated response in the pooling window.
                if (act > mval) {
                    mval = act;
                    max_idx = i * POOL1_S + j;  // pooling window index
                    active_flag = is_active;    // store the ReLU derivative flag for the chosen element
                }
            }
        }
        int out_index = bidx * CONV1_OUT * POOL1_OUT_SIZE * POOL1_OUT_SIZE +
                        chan * POOL1_OUT_SIZE * POOL1_OUT_SIZE +
                        oy * POOL1_OUT_SIZE + ox;
        out[out_index] = mval;
        argmax[out_index] = max_idx;
        relu_mask[out_index] = active_flag;
    }
}

/*
 * conv2_fused_kernel:
 *  - Performs convolution over the output of conv1 (which has dimensions: CONV1_OUT x POOL1_OUT_SIZE x POOL1_OUT_SIZE).
 *  - Uses CONV2_K x CONV2_K kernels over each input channel.
 *  - Uses valid convolution (bounds checked against POOL1_OUT_SIZE).
 *  - Applies ReLU and then performs 2x2 max pooling.
 *  - Saves the pooling argmax and a ReLU activation mask.
 */
__global__ void conv2_fused_kernel(const float *in, const float *w, const float *b, 
                                     float *out, int *argmax, int *relu_mask, int batch) {
    int bidx = blockIdx.z;
    if (bidx >= batch) return;
    int outc = blockIdx.x;
    int ox = blockIdx.y * blockDim.x + threadIdx.x;
    int oy = threadIdx.y;
    if (ox < POOL2_OUT_SIZE && oy < POOL2_OUT_SIZE) {
        float mval = -1e20f;
        int max_idx = 0; // Index within the pooling window
        int active_flag = 0;
        // Iterate over the pooling window
        for (int i = 0; i < POOL2_S; i++) {
            for (int j = 0; j < POOL2_S; j++) {
                int cx = ox * POOL2_S + i;
                int cy = oy * POOL2_S + j;
                float sum = b[outc];
                // Sum over all input channels from conv1
                for (int inc = 0; inc < CONV1_OUT; inc++) {
                    for (int m = 0; m < CONV2_K; m++) {
                        for (int n = 0; n < CONV2_K; n++) {
                            int ix = cx + m - CONV2_K / 2;
                            int iy = cy + n - CONV2_K / 2;
                            // Input here is the output of conv1, whose spatial dimensions are POOL1_OUT_SIZE.
                            if (ix >= 0 && ix < POOL1_OUT_SIZE && iy >= 0 && iy < POOL1_OUT_SIZE) {
                                sum += in[bidx * CONV1_OUT * POOL1_OUT_SIZE * POOL1_OUT_SIZE +
                                          inc * POOL1_OUT_SIZE * POOL1_OUT_SIZE +
                                          iy * POOL1_OUT_SIZE + ix] *
                                       w[outc * CONV1_OUT * CONV2_K * CONV2_K +
                                         inc * CONV2_K * CONV2_K +
                                         m * CONV2_K + n];
                            }
                        }
                    }
                }
                float act = fmaxf(0.0f, sum);
                int is_active = (sum > 0.0f) ? 1 : 0;
                if (act > mval) {
                    mval = act;
                    max_idx = i * POOL2_S + j;
                    active_flag = is_active;
                }
            }
        }
        int out_index = bidx * CONV2_OUT * POOL2_OUT_SIZE * POOL2_OUT_SIZE +
                        outc * POOL2_OUT_SIZE * POOL2_OUT_SIZE +
                        oy * POOL2_OUT_SIZE + ox;
        out[out_index] = mval;
        argmax[out_index] = max_idx;
        relu_mask[out_index] = active_flag;
    }
}

/*
 * fc_forward_kernel:
 *  - Performs a fully connected (dense) layer forward pass.
 *  - If 'relu' is true, applies ReLU activation (note: no separate mask is stored here).
 */
__global__ void fc_forward_kernel(const float *in, const float *w, const float *b, 
                                    float *out, int in_dim, int out_dim, bool relu) {
    int bidx = blockIdx.x;
    int o = threadIdx.x;
    if(o < out_dim){
        float sum = b[o];
        for (int i = 0; i < in_dim; i++){
            sum += in[bidx * in_dim + i] * w[o * in_dim + i];
        }
        out[bidx * out_dim + o] = relu ? fmaxf(0.0f, sum) : sum;
    }
}

/*
 * softmax_loss_kernel:
 *  - Computes the softmax loss for each sample and the gradient of the logits.
 */
__global__ void softmax_loss_kernel(const float *logits, const int *labels, 
                                      float *loss, float *dlogits, int batch, int num_classes) {
    int b = blockIdx.x;
    if (b < batch) {
        float mval = -1e20f;
        for (int i = 0; i < num_classes; i++) {
            float v = logits[b * num_classes + i];
            if (v > mval) mval = v;
        }
        float sum = 0;
        for (int i = 0; i < num_classes; i++) {
            float expv = expf(logits[b * num_classes + i] - mval);
            dlogits[b * num_classes + i] = expv;
            sum += expv;
        }
        float ls = -((logits[b * num_classes + labels[b]] - mval) - logf(sum));
        loss[b] = ls;
        for (int i = 0; i < num_classes; i++) {
            dlogits[b * num_classes + i] = dlogits[b * num_classes + i] / sum;
        }
        dlogits[b * num_classes + labels[b]] -= 1.0f;
    }
}

/*
 * update_kernel:
 *  - A simple SGD update kernel.
 */
__global__ void update_kernel(float *param, const float *grad, float lr, int batch_size, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        param[idx] -= lr * (grad[idx] / batch_size);  // Now uses passed batch_size
}

/*
 * fc_backward_kernel:
 *  - Performs the backward pass for a fully connected layer.
 *  - (Note: This kernel does not incorporate a ReLU derivative; if your FC layer uses ReLU, you
 *    might need to modify it similarly to the convolution kernels.)
 */
__global__ void fc_backward_kernel(const float *dout, const float *in, const float *w, 
                                     float *dW, float *db, float *din, int in_dim, int out_dim, int batch) {
    int o = blockIdx.x;
    int i = threadIdx.x;
    float grad = 0;
    for (int b = 0; b < batch; b++){
        grad += dout[b * out_dim + o] * in[b * in_dim + i];
    }
    atomicAdd(&dW[o * in_dim + i], grad);
    if (i == 0) {
        float gb = 0;
        for (int b = 0; b < batch; b++){
            gb += dout[b * out_dim + o];
        }
        atomicAdd(&db[o], gb);
    }
    for (int b = 0; b < batch; b++){
        atomicAdd(&din[b * in_dim + i], dout[b * out_dim + o] * w[o * in_dim + i]);
    }
}

/*
 * conv2_backward_kernel:
 *  - Backpropagates gradients through the second convolution layer (and its pooling).
 *  - Multiplies the incoming gradient by the saved ReLU mask.
 */
__global__ void conv2_backward_kernel(const float *dout, const float *in, const float *w, 
                                        float *dW, float *db, float *d_in_grad, 
                                        const int *argmax, const int *relu_mask, int batch) {
    int b = blockIdx.z;
    if (b >= batch) return;
    int oc = blockIdx.x;
    int ox = blockIdx.y * blockDim.x + threadIdx.x;
    int oy = threadIdx.y;
    if (ox < POOL2_OUT_SIZE && oy < POOL2_OUT_SIZE) {
        int out_index = b * CONV2_OUT * POOL2_OUT_SIZE * POOL2_OUT_SIZE +
                        oc * POOL2_OUT_SIZE * POOL2_OUT_SIZE +
                        oy * POOL2_OUT_SIZE + ox;
        // Multiply gradient by the ReLU mask (0 if inactive, 1 if active)
        float grad = dout[out_index] * relu_mask[out_index];
        atomicAdd(&db[oc], grad);
        int max_idx_in_pool = argmax[out_index];
        int start_x = ox * POOL2_S;
        int start_y = oy * POOL2_S;
        int in_x = start_x + (max_idx_in_pool % POOL2_S);
        int in_y = start_y + (max_idx_in_pool / POOL2_S);
        int in_index = b * CONV1_OUT * POOL1_OUT_SIZE * POOL1_OUT_SIZE +
                       in_y * POOL1_OUT_SIZE + in_x; // index into conv1 output
        atomicAdd(&d_in_grad[in_index], grad);
    }
}

/*
 * conv1_backward_kernel:
 *  - Backpropagates gradients through the first convolution layer (and its pooling).
 *  - Uses a dedicated input-gradient buffer (d_in_grad) rather than modifying the input.
 *  - Multiplies the gradient by the saved ReLU mask.
 */
__global__ void conv1_backward_kernel(const float *dout, const float *in, const float *w, 
                                        float *dW, float *db, float *d_in_grad, 
                                        const int *argmax, const int *relu_mask, int batch) {
    int b = blockIdx.z;
    if (b >= batch) return;
    int c = blockIdx.x;
    int ox = blockIdx.y * blockDim.x + threadIdx.x;
    int oy = threadIdx.y;
    if (ox < POOL1_OUT_SIZE && oy < POOL1_OUT_SIZE) {
        int out_index = b * CONV1_OUT * POOL1_OUT_SIZE * POOL1_OUT_SIZE +
                        c * POOL1_OUT_SIZE * POOL1_OUT_SIZE +
                        oy * POOL1_OUT_SIZE + ox;
        // Multiply the gradient by the ReLU derivative stored in relu_mask
        float grad = dout[out_index] * relu_mask[out_index];
        atomicAdd(&db[c], grad);
        int max_idx_in_pool = argmax[out_index];
        int start_x = ox * POOL1_S;
        int start_y = oy * POOL1_S;
        int in_x = start_x + (max_idx_in_pool % POOL1_S);
        int in_y = start_y + (max_idx_in_pool / POOL1_S);
        int in_index = b * IMG_SIZE + in_y * IMG_WIDTH + in_x; // index into the original input image
        atomicAdd(&d_in_grad[in_index], grad);
    }
}


// (Assume the kernels are defined in an included file or above in this file.)

int main(){
    int train_size = 60000, test_size = 10000;

    // Load training and testing data from files.
    float *h_train = loadImages("mnist_train_images.bin", train_size, IMG_SIZE);
    int *h_train_labels = loadLabels("mnist_train_labels.bin", train_size);
    float *h_test = loadImages("mnist_test_images.bin", test_size, IMG_SIZE);
    int *h_test_labels = loadLabels("mnist_test_labels.bin", test_size);

    // Allocate device memory for training and testing data.
    float *d_train, *d_test;
    int *d_train_labels, *d_test_labels;
    cudaMalloc(&d_train, train_size * IMG_SIZE * sizeof(float));
    cudaMalloc(&d_train_labels, train_size * sizeof(int));
    cudaMalloc(&d_test, test_size * IMG_SIZE * sizeof(float));
    cudaMalloc(&d_test_labels, test_size * sizeof(int));
    cudaMemcpy(d_train, h_train, train_size * IMG_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_labels, h_train_labels, train_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test, h_test, test_size * IMG_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_labels, h_test_labels, test_size * sizeof(int), cudaMemcpyHostToDevice);

    // --- Initialize network parameters ---
    // Conv1 Layer
    size_t sz_conv1_out_batch = BATCH_SIZE * CONV1_OUT * POOL1_OUT_SIZE * POOL1_OUT_SIZE * sizeof(float);
    size_t sz_conv1_w = CONV1_OUT * CONV1_K * CONV1_K * sizeof(float);  // Must exist
    float *d_conv1_w, *d_conv1_b;
    cudaMalloc(&d_conv1_w, sz_conv1_w);
    cudaMalloc(&d_conv1_b, CONV1_OUT * sizeof(float));
    float *h_conv1_w = (float*)malloc(sz_conv1_w);
    float *h_conv1_b = (float*)malloc(CONV1_OUT * sizeof(float));
    for (int i = 0; i < CONV1_OUT * CONV1_K * CONV1_K; i++) 
        h_conv1_w[i] = (((float)rand() / RAND_MAX) - 0.5f) / 5;
    for (int i = 0; i < CONV1_OUT; i++) 
        h_conv1_b[i] = 0;
    cudaMemcpy(d_conv1_w, h_conv1_w, sz_conv1_w, cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1_b, h_conv1_b, CONV1_OUT * sizeof(float), cudaMemcpyHostToDevice);

    // Conv2 Layer
    size_t sz_conv2_w = CONV2_OUT * CONV1_OUT * CONV2_K * CONV2_K * sizeof(float);
    float *d_conv2_w, *d_conv2_b;
    cudaMalloc(&d_conv2_w, sz_conv2_w);
    cudaMalloc(&d_conv2_b, CONV2_OUT * sizeof(float));
    float *h_conv2_w = (float*)malloc(sz_conv2_w);
    float *h_conv2_b = (float*)malloc(CONV2_OUT * sizeof(float));
    for (int i = 0; i < CONV2_OUT * CONV1_OUT * CONV2_K * CONV2_K; i++) 
        h_conv2_w[i] = (((float)rand() / RAND_MAX) - 0.5f) / 5;
    for (int i = 0; i < CONV2_OUT; i++) 
        h_conv2_b[i] = 0;
    cudaMemcpy(d_conv2_w, h_conv2_w, sz_conv2_w, cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2_b, h_conv2_b, CONV2_OUT * sizeof(float), cudaMemcpyHostToDevice);

    // Fully Connected Layer 1
    size_t sz_fc1_w = FC1_IN * FC1_OUT * sizeof(float);
    float *d_fc1_w, *d_fc1_b;
    cudaMalloc(&d_fc1_w, sz_fc1_w);
    cudaMalloc(&d_fc1_b, FC1_OUT * sizeof(float));
    float *h_fc1_w = (float*)malloc(sz_fc1_w);
    float *h_fc1_b = (float*)malloc(FC1_OUT * sizeof(float));
    for (int i = 0; i < FC1_IN * FC1_OUT; i++) 
        h_fc1_w[i] = (((float)rand() / RAND_MAX) - 0.5f) / 5;
    for (int i = 0; i < FC1_OUT; i++) 
        h_fc1_b[i] = 0;
    cudaMemcpy(d_fc1_w, h_fc1_w, sz_fc1_w, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_b, h_fc1_b, FC1_OUT * sizeof(float), cudaMemcpyHostToDevice);

    // Fully Connected Layer 2
    size_t sz_fc2_w = FC1_OUT * FC2_OUT * sizeof(float);
    float *d_fc2_w, *d_fc2_b;
    cudaMalloc(&d_fc2_w, sz_fc2_w);
    cudaMalloc(&d_fc2_b, FC2_OUT * sizeof(float));
    float *h_fc2_w = (float*)malloc(sz_fc2_w);
    float *h_fc2_b = (float*)malloc(FC2_OUT * sizeof(float));
    for (int i = 0; i < FC1_OUT * FC2_OUT; i++) 
        h_fc2_w[i] = (((float)rand() / RAND_MAX) - 0.5f) / 5;
    for (int i = 0; i < FC2_OUT; i++) 
        h_fc2_b[i] = 0;
    cudaMemcpy(d_fc2_w, h_fc2_w, sz_fc2_w, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_b, h_fc2_b, FC2_OUT * sizeof(float), cudaMemcpyHostToDevice);

    // Fully Connected Layer 3 (Output Layer)
    size_t sz_fc3_w = FC2_OUT * NUM_CLASSES * sizeof(float);
    float *d_fc3_w, *d_fc3_b;
    cudaMalloc(&d_fc3_w, sz_fc3_w);
    cudaMalloc(&d_fc3_b, NUM_CLASSES * sizeof(float));
    float *h_fc3_w = (float*)malloc(sz_fc3_w);
    float *h_fc3_b = (float*)malloc(NUM_CLASSES * sizeof(float));
    for (int i = 0; i < FC2_OUT * NUM_CLASSES; i++) 
        h_fc3_w[i] = (((float)rand() / RAND_MAX) - 0.5f) / 5;
    for (int i = 0; i < NUM_CLASSES; i++) 
        h_fc3_b[i] = 0;
    cudaMemcpy(d_fc3_w, h_fc3_w, sz_fc3_w, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc3_b, h_fc3_b, NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice);

    // --- Allocate memory for layer outputs, argmax indices, and ReLU masks for training data ---
    size_t sz_conv1_out = train_size * CONV1_OUT * POOL1_OUT_SIZE * POOL1_OUT_SIZE * sizeof(float);
    float *d_conv1_out; cudaMalloc(&d_conv1_out, sz_conv1_out_batch);
    size_t sz_conv2_out = train_size * CONV2_OUT * POOL2_OUT_SIZE * POOL2_OUT_SIZE * sizeof(float);
    float *d_conv2_out; cudaMalloc(&d_conv2_out, sz_conv2_out);
    size_t sz_fc1_out = train_size * FC1_OUT * sizeof(float);
    float *d_fc1_out; cudaMalloc(&d_fc1_out, sz_fc1_out);
    size_t sz_fc2_out = train_size * FC2_OUT * sizeof(float);
    float *d_fc2_out; cudaMalloc(&d_fc2_out, sz_fc2_out);
    size_t sz_fc3_out = train_size * NUM_CLASSES * sizeof(float);
    float *d_fc3_out; cudaMalloc(&d_fc3_out, sz_fc3_out);

    // Argmax buffers for pooling layers.
    size_t sz_conv1_argmax = train_size * CONV1_OUT * POOL1_OUT_SIZE * POOL1_OUT_SIZE * sizeof(int);
    int *d_conv1_argmax; cudaMalloc(&d_conv1_argmax, sz_conv1_argmax);
    size_t sz_conv2_argmax = train_size * CONV2_OUT * POOL2_OUT_SIZE * POOL2_OUT_SIZE * sizeof(int);
    int *d_conv2_argmax; cudaMalloc(&d_conv2_argmax, sz_conv2_argmax);

    // ReLU mask buffers for conv layers.
    size_t sz_conv1_relu = train_size * CONV1_OUT * POOL1_OUT_SIZE * POOL1_OUT_SIZE * sizeof(int);
    int *d_conv1_relu_mask; cudaMalloc(&d_conv1_relu_mask, sz_conv1_relu);
    size_t sz_conv2_relu = train_size * CONV2_OUT * POOL2_OUT_SIZE * POOL2_OUT_SIZE * sizeof(int);
    int *d_conv2_relu_mask; cudaMalloc(&d_conv2_relu_mask, sz_conv2_relu);

    // --- Allocate memory for loss and gradients on GPU ---
    size_t sz_loss = BATCH_SIZE * sizeof(float);
    float *d_loss; cudaMalloc(&d_loss, sz_loss);
    size_t sz_dlogits = BATCH_SIZE * NUM_CLASSES * sizeof(float);
    float *d_dlogits; cudaMalloc(&d_dlogits, sz_dlogits);

    // Gradient buffers for FC layers.
    float *d_fc3_in_grad; cudaMalloc(&d_fc3_in_grad, BATCH_SIZE * FC2_OUT * sizeof(float));
    float *d_fc2_in_grad; cudaMalloc(&d_fc2_in_grad, BATCH_SIZE * FC1_OUT * sizeof(float));
    float *d_fc1_in_grad; cudaMalloc(&d_fc1_in_grad, BATCH_SIZE * FC1_IN * sizeof(float));

    // Buffer for gradient of conv2's output (which is the input to FC layer 1).
    float *d_conv2_out_grad; cudaMalloc(&d_conv2_out_grad, BATCH_SIZE * FC1_IN * sizeof(float));

    // --- Allocate dedicated gradient buffers for convolution layers --- 
    // For conv2: gradient input from conv1's output.
    float *d_conv2_in_grad; 
    cudaMalloc(&d_conv2_in_grad, BATCH_SIZE * CONV1_OUT * POOL1_OUT_SIZE * POOL1_OUT_SIZE * sizeof(float));
    // For conv1: gradient input (must be separate from d_train).
    float *d_conv1_in_grad;
    cudaMalloc(&d_conv1_in_grad, BATCH_SIZE * IMG_SIZE * sizeof(float));

    // Allocate memory for weight and bias gradients for each layer.
    float *d_conv1_w_grad, *d_conv1_b_grad;
    cudaMalloc(&d_conv1_w_grad, sz_conv1_w);
    cudaMalloc(&d_conv1_b_grad, CONV1_OUT * sizeof(float));
    float *d_conv2_w_grad, *d_conv2_b_grad;
    cudaMalloc(&d_conv2_w_grad, sz_conv2_w);
    cudaMalloc(&d_conv2_b_grad, CONV2_OUT * sizeof(float));
    float *d_fc1_w_grad, *d_fc1_b_grad;
    cudaMalloc(&d_fc1_w_grad, sz_fc1_w);
    cudaMalloc(&d_fc1_b_grad, FC1_OUT * sizeof(float));
    float *d_fc2_w_grad, *d_fc2_b_grad;
    cudaMalloc(&d_fc2_w_grad, sz_fc2_w);
    cudaMalloc(&d_fc2_b_grad, FC2_OUT * sizeof(float));
    float *d_fc3_w_grad, *d_fc3_b_grad;
    cudaMalloc(&d_fc3_w_grad, sz_fc3_w);
    cudaMalloc(&d_fc3_b_grad, NUM_CLASSES * sizeof(float));

    // Define grid and block dimensions for the convolution kernels.
    dim3 grid1(CONV1_OUT, (POOL1_OUT_SIZE + 15) / 16, BATCH_SIZE), block1(16, 16);
    dim3 grid2(CONV2_OUT, (POOL2_OUT_SIZE + 15) / 16, BATCH_SIZE), block2(16, 16);

    // --- Training Loop ---
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < train_size; i += BATCH_SIZE) {
            int cur = (i + BATCH_SIZE > train_size) ? (train_size - i) : BATCH_SIZE;

            // Initialize gradients and temporary buffers to zero.
            cudaMemset(d_conv1_w_grad, 0, sz_conv1_w);
            cudaMemset(d_conv1_b_grad, 0, CONV1_OUT * sizeof(float));
            cudaMemset(d_conv2_w_grad, 0, sz_conv2_w);
            cudaMemset(d_conv2_b_grad, 0, CONV2_OUT * sizeof(float));
            cudaMemset(d_fc1_w_grad, 0, sz_fc1_w);
            cudaMemset(d_fc1_b_grad, 0, FC1_OUT * sizeof(float));
            cudaMemset(d_fc2_w_grad, 0, sz_fc2_w);
            cudaMemset(d_fc2_b_grad, 0, FC2_OUT * sizeof(float));
            cudaMemset(d_fc3_w_grad, 0, sz_fc3_w);
            cudaMemset(d_fc3_b_grad, 0, NUM_CLASSES * sizeof(float));
            cudaMemset(d_loss, 0, cur * sizeof(float));
            cudaMemset(d_dlogits, 0, cur * NUM_CLASSES * sizeof(float));
            cudaMemset(d_fc3_in_grad, 0, cur * FC2_OUT * sizeof(float));
            cudaMemset(d_fc2_in_grad, 0, cur * FC1_OUT * sizeof(float));
            cudaMemset(d_fc1_in_grad, 0, cur * FC1_IN * sizeof(float));
            cudaMemset(d_conv2_out_grad, 0, cur * FC1_IN * sizeof(float));
            cudaMemset(d_conv2_in_grad, 0, cur * CONV1_OUT * POOL1_OUT_SIZE * POOL1_OUT_SIZE * sizeof(float));
            cudaMemset(d_conv1_in_grad, 0, cur * IMG_SIZE * sizeof(float));

            // Forward Pass for current batch:
            dim3 grid1_batch(CONV1_OUT, (POOL1_OUT_SIZE + 15) / 16, cur);
            conv1_fused_kernel<<<grid1_batch, block1>>>(
                d_train + i * IMG_SIZE,
                d_conv1_w, d_conv1_b,
                d_conv1_out + i * CONV1_OUT * POOL1_OUT_SIZE * POOL1_OUT_SIZE,
                d_conv1_argmax + i * CONV1_OUT * POOL1_OUT_SIZE * POOL1_OUT_SIZE,
                d_conv1_relu_mask + i * CONV1_OUT * POOL1_OUT_SIZE * POOL1_OUT_SIZE,
                cur
            );

            dim3 grid2_batch(CONV2_OUT, (POOL2_OUT_SIZE + 15) / 16, cur);
            conv2_fused_kernel<<<grid2_batch, block2>>>(
                d_conv1_out + i * CONV1_OUT * POOL1_OUT_SIZE * POOL1_OUT_SIZE,
                d_conv2_w, d_conv2_b,
                d_conv2_out + i * CONV2_OUT * POOL2_OUT_SIZE * POOL2_OUT_SIZE,
                d_conv2_argmax + i * CONV2_OUT * POOL2_OUT_SIZE * POOL2_OUT_SIZE,
                d_conv2_relu_mask + i * CONV2_OUT * POOL2_OUT_SIZE * POOL2_OUT_SIZE,
                cur
            );

            // Allocate temporary buffer for FC layer input.
            float *d_fc1_in; 
            cudaMalloc(&d_fc1_in, cur * FC1_IN * sizeof(float));
            cudaMemcpy(d_fc1_in, d_conv2_out + i * CONV2_OUT * POOL2_OUT_SIZE * POOL2_OUT_SIZE,
                       cur * FC1_IN * sizeof(float), cudaMemcpyDeviceToDevice);

            // Forward pass through fully connected layers.
            fc_forward_kernel<<<cur, FC1_OUT>>>(d_fc1_in, d_fc1_w, d_fc1_b, 
                                                d_fc1_out + i * FC1_OUT, FC1_IN, FC1_OUT, true);
            fc_forward_kernel<<<cur, FC2_OUT>>>(d_fc1_out + i * FC1_OUT, d_fc2_w, d_fc2_b, 
                                                d_fc2_out + i * FC2_OUT, FC1_OUT, FC2_OUT, true);
            fc_forward_kernel<<<cur, NUM_CLASSES>>>(d_fc2_out + i * FC2_OUT, d_fc3_w, d_fc3_b, 
                                                    d_fc3_out + i * NUM_CLASSES, FC2_OUT, NUM_CLASSES, false);

            // Compute softmax loss and gradients.
            softmax_loss_kernel<<<cur, 1>>>(d_fc3_out + i * NUM_CLASSES,
                                            d_train_labels + i,
                                            d_loss, d_dlogits, cur, NUM_CLASSES);

            // Backward Pass:
            fc_backward_kernel<<<NUM_CLASSES, FC2_OUT>>>(
                d_dlogits,
                d_fc2_out + i * FC2_OUT,
                d_fc3_w,
                d_fc3_w_grad,
                d_fc3_b_grad,
                d_fc3_in_grad,
                FC2_OUT, NUM_CLASSES, cur
            );
            fc_backward_kernel<<<FC2_OUT, FC1_OUT>>>(
                d_fc3_in_grad,
                d_fc1_out + i * FC1_OUT,
                d_fc2_w,
                d_fc2_w_grad,
                d_fc2_b_grad,
                d_fc2_in_grad,
                FC1_OUT, FC2_OUT, cur
            );
            fc_backward_kernel<<<FC1_OUT, 256>>>(
                d_fc2_in_grad,
                d_fc1_in,
                d_fc1_w,
                d_fc1_w_grad,
                d_fc1_b_grad,
                d_fc1_in_grad,
                FC1_IN, FC1_OUT, cur
            );

            // Propagate gradients back from the FC input to conv2's output.
            cudaMemcpy(d_conv2_out_grad, d_fc1_in_grad,
                       cur * FC1_IN * sizeof(float), cudaMemcpyDeviceToDevice);

            // Backward pass through convolution layers.
            conv2_backward_kernel<<<grid2_batch, block2>>>(
                d_conv2_out_grad,
                d_conv1_out + i * CONV1_OUT * POOL1_OUT_SIZE * POOL1_OUT_SIZE,
                d_conv2_w,
                d_conv2_w_grad,
                d_conv2_b_grad,
                d_conv2_in_grad + i * CONV1_OUT * POOL1_OUT_SIZE * POOL1_OUT_SIZE,
                d_conv2_argmax + i * CONV2_OUT * POOL2_OUT_SIZE * POOL2_OUT_SIZE,
                d_conv2_relu_mask + i * CONV2_OUT * POOL2_OUT_SIZE * POOL2_OUT_SIZE,
                cur
            );

            conv1_backward_kernel<<<grid1_batch, block1>>>(
                d_conv1_in_grad + i * IMG_SIZE,
                d_train + i * IMG_SIZE,
                d_conv1_w,
                d_conv1_w_grad,
                d_conv1_b_grad,
                d_conv1_in_grad + i * IMG_SIZE,
                d_conv1_argmax + i * CONV1_OUT * POOL1_OUT_SIZE * POOL1_OUT_SIZE,
                d_conv1_relu_mask + i * CONV1_OUT * POOL1_OUT_SIZE * POOL1_OUT_SIZE,
                cur
            );

            // Update network parameters using SGD.
// Update network parameters using SGD
int threads = 256, num, blocks;

// Conv1 weights
num = CONV1_OUT * CONV1_K * CONV1_K;
blocks = (num + threads - 1) / threads;
update_kernel<<<blocks, threads>>>(d_conv1_w, d_conv1_w_grad, LR, cur, num);

// Conv1 biases
num = CONV1_OUT;
blocks = (num + threads - 1) / threads;
update_kernel<<<blocks, threads>>>(d_conv1_b, d_conv1_b_grad, LR, cur, num);

// Conv2 weights
num = CONV2_OUT * CONV1_OUT * CONV2_K * CONV2_K;
blocks = (num + threads - 1) / threads;
update_kernel<<<blocks, threads>>>(d_conv2_w, d_conv2_w_grad, LR, cur, num);

// Conv2 biases
num = CONV2_OUT;
blocks = (num + threads - 1) / threads;
update_kernel<<<blocks, threads>>>(d_conv2_b, d_conv2_b_grad, LR, cur, num);

// FC1 weights
num = FC1_IN * FC1_OUT;
blocks = (num + threads - 1) / threads;
update_kernel<<<blocks, threads>>>(d_fc1_w, d_fc1_w_grad, LR, cur, num);

// FC1 biases
num = FC1_OUT;
blocks = (num + threads - 1) / threads;
update_kernel<<<blocks, threads>>>(d_fc1_b, d_fc1_b_grad, LR, cur, num);

// FC2 weights
num = FC1_OUT * FC2_OUT;
blocks = (num + threads - 1) / threads;
update_kernel<<<blocks, threads>>>(d_fc2_w, d_fc2_w_grad, LR, cur, num);

// FC2 biases
num = FC2_OUT;
blocks = (num + threads - 1) / threads;
update_kernel<<<blocks, threads>>>(d_fc2_b, d_fc2_b_grad, LR, cur, num);

// FC3 weights
num = FC2_OUT * NUM_CLASSES;
blocks = (num + threads - 1) / threads;
update_kernel<<<blocks, threads>>>(d_fc3_w, d_fc3_w_grad, LR, cur, num);

// FC3 biases
num = NUM_CLASSES;
blocks = (num + threads - 1) / threads;
update_kernel<<<blocks, threads>>>(d_fc3_b, d_fc3_b_grad, LR, cur, num);
            // Free temporary buffer for FC input.
            cudaFree(d_fc1_in);
        }
    }

    // --- Evaluation on Training Data ---
    int correct_train = 0;
    float *h_fc3_out = (float*)malloc(train_size * NUM_CLASSES * sizeof(float));
    cudaMemcpy(h_fc3_out, d_fc3_out, train_size * NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < train_size; i++){
        int pred = 0;
        float mval = -1e20f;
        for (int j = 0; j < NUM_CLASSES; j++){
            float v = h_fc3_out[i * NUM_CLASSES + j];
            if (v > mval) { mval = v; pred = j; }
        }
        if (pred == h_train_labels[i]) correct_train++;
    }

    // --- Evaluation on Test Data ---
    int correct_test = 0;
    // Allocate buffers for test forward pass.
    size_t sz_conv1_out_test = test_size * CONV1_OUT * POOL1_OUT_SIZE * POOL1_OUT_SIZE * sizeof(float);
    float *d_conv1_out_test; cudaMalloc(&d_conv1_out_test, sz_conv1_out_test);
    size_t sz_conv2_out_test = test_size * CONV2_OUT * POOL2_OUT_SIZE * POOL2_OUT_SIZE * sizeof(float);
    float *d_conv2_out_test; cudaMalloc(&d_conv2_out_test, sz_conv2_out_test);
    size_t sz_fc1_out_test = test_size * FC1_OUT * sizeof(float);
    float *d_fc1_out_test; cudaMalloc(&d_fc1_out_test, sz_fc1_out_test);
    size_t sz_fc2_out_test = test_size * FC2_OUT * sizeof(float);
    float *d_fc2_out_test; cudaMalloc(&d_fc2_out_test, sz_fc2_out_test);
    size_t sz_fc3_out_test = test_size * NUM_CLASSES * sizeof(float);
    float *d_fc3_out_test; cudaMalloc(&d_fc3_out_test, sz_fc3_out_test);
    size_t sz_conv1_argmax_test = test_size * CONV1_OUT * POOL1_OUT_SIZE * POOL1_OUT_SIZE * sizeof(int);
    int *d_conv1_argmax_test; cudaMalloc(&d_conv1_argmax_test, sz_conv1_argmax_test);
    size_t sz_conv2_argmax_test = test_size * CONV2_OUT * POOL2_OUT_SIZE * POOL2_OUT_SIZE * sizeof(int);
    int *d_conv2_argmax_test; cudaMalloc(&d_conv2_argmax_test, sz_conv2_argmax_test);
    // Allocate ReLU mask buffers for test data.
    size_t sz_conv1_relu_test = test_size * CONV1_OUT * POOL1_OUT_SIZE * POOL1_OUT_SIZE * sizeof(int);
    int *d_conv1_relu_mask_test; cudaMalloc(&d_conv1_relu_mask_test, sz_conv1_relu_test);
    size_t sz_conv2_relu_test = test_size * CONV2_OUT * POOL2_OUT_SIZE * POOL2_OUT_SIZE * sizeof(int);
    int *d_conv2_relu_mask_test; cudaMalloc(&d_conv2_relu_mask_test, sz_conv2_relu_test);

    // Forward pass on test data.
    dim3 grid1_test(CONV1_OUT, (POOL1_OUT_SIZE + 15) / 16, test_size);
    conv1_fused_kernel<<<grid1_test, block1>>>(
        d_test, d_conv1_w, d_conv1_b,
        d_conv1_out_test, d_conv1_argmax_test, d_conv1_relu_mask_test,
        test_size
    );
    dim3 grid2_test(CONV2_OUT, (POOL2_OUT_SIZE + 15) / 16, test_size);
    conv2_fused_kernel<<<grid2_test, block2>>>(
        d_conv1_out_test, d_conv2_w, d_conv2_b,
        d_conv2_out_test, d_conv2_argmax_test, d_conv2_relu_mask_test,
        test_size
    );
    float *d_fc1_in_test; 
    cudaMalloc(&d_fc1_in_test, test_size * FC1_IN * sizeof(float));
    cudaMemcpy(d_fc1_in_test, d_conv2_out_test, test_size * FC1_IN * sizeof(float), cudaMemcpyDeviceToDevice);
    fc_forward_kernel<<<test_size, FC1_OUT>>>(d_fc1_in_test, d_fc1_w, d_fc1_b, d_fc1_out_test, FC1_IN, FC1_OUT, true);
    fc_forward_kernel<<<test_size, FC2_OUT>>>(d_fc1_out_test, d_fc2_w, d_fc2_b, d_fc2_out_test, FC1_OUT, FC2_OUT, true);
    fc_forward_kernel<<<test_size, NUM_CLASSES>>>(d_fc2_out_test, d_fc3_w, d_fc3_b, d_fc3_out_test, FC2_OUT, NUM_CLASSES, false);

    float *h_fc3_out_test = (float*)malloc(test_size * NUM_CLASSES * sizeof(float));
    cudaMemcpy(h_fc3_out_test, d_fc3_out_test, test_size * NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < test_size; i++){
        int pred = 0;
        float mval = -1e20f;
        for (int j = 0; j < NUM_CLASSES; j++){
            float v = h_fc3_out_test[i * NUM_CLASSES + j];
            if (v > mval) { mval = v; pred = j; }
        }
        if (pred == h_test_labels[i]) correct_test++;
    }

    // --- Print Accuracy ---
    printf("Train Accuracy: %.2f%%\n", 100.0f * (float)correct_train / train_size);
    printf("Test Accuracy: %.2f%%\n", 100.0f * (float)correct_test / test_size);

    // --- Free Device and Host Memory ---
    cudaFree(d_train);           cudaFree(d_train_labels);
    cudaFree(d_test);            cudaFree(d_test_labels);
    cudaFree(d_conv1_w);         cudaFree(d_conv1_b);
    cudaFree(d_conv2_w);         cudaFree(d_conv2_b);
    cudaFree(d_fc1_w);           cudaFree(d_fc1_b);
    cudaFree(d_fc2_w);           cudaFree(d_fc2_b);
    cudaFree(d_fc3_w);           cudaFree(d_fc3_b);
    cudaFree(d_conv1_out);       cudaFree(d_conv2_out);
    cudaFree(d_fc1_out);         cudaFree(d_fc2_out);         cudaFree(d_fc3_out);
    cudaFree(d_conv1_argmax);    cudaFree(d_conv2_argmax);
    cudaFree(d_conv1_relu_mask); cudaFree(d_conv2_relu_mask);
    cudaFree(d_loss);            cudaFree(d_dlogits);
    cudaFree(d_fc3_in_grad);     cudaFree(d_fc2_in_grad);     cudaFree(d_fc1_in_grad);
    cudaFree(d_conv2_out_grad);
    cudaFree(d_conv2_in_grad);   cudaFree(d_conv1_in_grad);
    cudaFree(d_conv1_w_grad);    cudaFree(d_conv1_b_grad);
    cudaFree(d_conv2_w_grad);    cudaFree(d_conv2_b_grad);
    cudaFree(d_fc1_w_grad);      cudaFree(d_fc1_b_grad);
    cudaFree(d_fc2_w_grad);      cudaFree(d_fc2_b_grad);
    cudaFree(d_fc3_w_grad);      cudaFree(d_fc3_b_grad);
    cudaFree(d_conv1_out_test);  cudaFree(d_conv2_out_test);
    cudaFree(d_fc1_out_test);    cudaFree(d_fc2_out_test);    cudaFree(d_fc3_out_test);
    cudaFree(d_conv1_argmax_test); cudaFree(d_conv2_argmax_test);
    cudaFree(d_conv1_relu_mask_test); cudaFree(d_conv2_relu_mask_test);
    cudaFree(d_fc1_in_test);
    free(h_train); free(h_train_labels);
    free(h_test);  free(h_test_labels);
    free(h_conv1_w); free(h_conv1_b);
    free(h_conv2_w); free(h_conv2_b);
    free(h_fc1_w); free(h_fc1_b);
    free(h_fc2_w); free(h_fc2_b);
    free(h_fc3_w); free(h_fc3_b);
    free(h_fc3_out); free(h_fc3_out_test);

    return 0;
}


