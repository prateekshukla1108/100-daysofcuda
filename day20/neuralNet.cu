#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <float.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call;   \
    if(err != cudaSuccess){   \
        fprintf(stderr, "CUDA error in %s at %s:%d: %s\n", __FUNCTION__, __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);   \
    }                         \
} while(0)

__global__ void compute_gradients(
    const float* __restrict__ input,
    const float* __restrict__ target,
    const float* __restrict__ w1, const float* __restrict__ b1,
    const float* __restrict__ w2, const float* __restrict__ b2,
    const float* __restrict__ w3, const float* __restrict__ b3,
    float* d_grad_w1, float* d_grad_b1,
    float* d_grad_w2, float* d_grad_b2,
    float* d_grad_w3, float* d_grad_b3,
    const int batch_size,
    const int input_dim,
    const int hidden1_dim,
    const int hidden2_dim,
    const int output_dim,
    float* tmp_buffer, int tmp_stride)
{
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(sample_idx < batch_size) {
        const float* sample = input + sample_idx * input_dim;
        const float* sample_target = target + sample_idx * output_dim;
        float* base = tmp_buffer + sample_idx * tmp_stride;
        float* z1 = base;                              
        float* a1 = z1 + hidden1_dim;                  
        float* d1 = a1 + hidden1_dim;                  
        float* z2 = d1 + hidden1_dim;                  
        float* a2 = z2 + hidden2_dim;                  
        float* d2 = a2 + hidden2_dim;                  
        float* z3 = d2 + hidden2_dim;                  
        float* out = z3 + output_dim;                  
        float* d3 = out + output_dim;                  

        for (int i = 0; i < hidden1_dim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < input_dim; j++) {
                sum += sample[j] * w1[j * hidden1_dim + i];
            }
            sum += b1[i];
            z1[i] = sum;
            a1[i] = (sum > 0.0f) ? sum : 0.0f;
        }
        for (int i = 0; i < hidden2_dim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < hidden1_dim; j++) {
                sum += a1[j] * w2[j * hidden2_dim + i];
            }
            sum += b2[i];
            z2[i] = sum;
            a2[i] = (sum > 0.0f) ? sum : 0.0f;
        }
        float max_val = -FLT_MAX;
        for (int i = 0; i < output_dim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < hidden2_dim; j++) {
                sum += a2[j] * w3[j * output_dim + i];
            }
            sum += b3[i];
            z3[i] = sum;
            if (sum > max_val) max_val = sum;
        }
        float exp_sum = 0.0f;
        for (int i = 0; i < output_dim; i++) {
            float e = expf(z3[i] - max_val);
            out[i] = e;
            exp_sum += e;
        }
        for (int i = 0; i < output_dim; i++) {
            out[i] /= exp_sum;
        }

        for (int i = 0; i < output_dim; i++) {
            d3[i] = out[i] - sample_target[i];
        }
        for (int i = 0; i < output_dim; i++) {
            atomicAdd(&d_grad_b3[i], d3[i]);
            for (int j = 0; j < hidden2_dim; j++) {
                float grad = a2[j] * d3[i];
                atomicAdd(&d_grad_w3[j * output_dim + i], grad);
            }
        }
        for (int i = 0; i < hidden2_dim; i++) {
            float error_sum = 0.0f;
            for (int j = 0; j < output_dim; j++) {
                error_sum += w3[i * output_dim + j] * d3[j];
            }
            float relu_deriv = (z2[i] > 0.0f) ? 1.0f : 0.0f;
            d2[i] = error_sum * relu_deriv;
        }
        for (int i = 0; i < hidden2_dim; i++) {
            atomicAdd(&d_grad_b2[i], d2[i]);
            for (int j = 0; j < hidden1_dim; j++) {
                float grad = a1[j] * d2[i];
                atomicAdd(&d_grad_w2[j * hidden2_dim + i], grad);
            }
        }
        for (int i = 0; i < hidden1_dim; i++) {
            float error_sum = 0.0f;
            for (int j = 0; j < hidden2_dim; j++) {
                error_sum += w2[i * hidden2_dim + j] * d2[j];
            }
            float relu_deriv = (z1[i] > 0.0f) ? 1.0f : 0.0f;
            d1[i] = error_sum * relu_deriv;
        }
        for (int i = 0; i < hidden1_dim; i++) {
            atomicAdd(&d_grad_b1[i], d1[i]);
            for (int j = 0; j < input_dim; j++) {
                float grad = sample[j] * d1[i];
                atomicAdd(&d_grad_w1[j * hidden1_dim + i], grad);
            }
        }
    }
}

__global__ void update_weights(float* w, const float* d_grad, int count, float learning_rate, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < count){
        w[idx] -= learning_rate * (d_grad[idx] / batch_size);
    }
}

void forward_cpu(const float* input,
    const float* w1, const float* b1,
    const float* w2, const float* b2,
    const float* w3, const float* b3,
    int input_dim,
    int hidden1_dim,
    int hidden2_dim,
    int output_dim,
    float* output)
{
    float* z1 = (float*)malloc(hidden1_dim * sizeof(float));
    float* a1 = (float*)malloc(hidden1_dim * sizeof(float));
    float* z2 = (float*)malloc(hidden2_dim * sizeof(float));
    float* a2 = (float*)malloc(hidden2_dim * sizeof(float));
    float* z3 = (float*)malloc(output_dim * sizeof(float));
    float max_val = -FLT_MAX;
    for(int i = 0; i < hidden1_dim; i++){
        float sum = 0.0f;
        for(int j = 0; j < input_dim; j++){
            sum += input[j] * w1[j * hidden1_dim + i];
        }
        sum += b1[i];
        z1[i] = sum;
        a1[i] = (sum > 0.0f) ? sum : 0.0f;
    }
    for(int i = 0; i < hidden2_dim; i++){
        float sum = 0.0f;
        for(int j = 0; j < hidden1_dim; j++){
            sum += a1[j] * w2[j * hidden2_dim + i];
        }
        sum += b2[i];
        z2[i] = sum;
        a2[i] = (sum > 0.0f) ? sum : 0.0f;
    }
    for(int i = 0; i < output_dim; i++){
        float sum = 0.0f;
        for(int j = 0; j < hidden2_dim; j++){
            sum += a2[j] * w3[j * output_dim + i];
        }
        sum += b3[i];
        z3[i] = sum;
        if(sum > max_val) max_val = sum;
    }
    float exp_sum = 0.0f;
    for(int i = 0; i < output_dim; i++){
        float e = expf(z3[i] - max_val);
        output[i] = e;
        exp_sum += e;
    }
    for(int i = 0; i < output_dim; i++){
        output[i] /= exp_sum;
    }
    free(z1); free(a1); free(z2); free(a2); free(z3);
}

int main(){
    const int batch_size    = 128;
    const int input_dim     = 64;
    const int hidden1_dim   = 128;
    const int hidden2_dim   = 64;
    const int output_dim    = 10;
    const float learning_rate = 0.01f;
    const int num_iterations  = 10;

    size_t input_size  = batch_size * input_dim * sizeof(float);
    size_t target_size = batch_size * output_dim * sizeof(float);
    size_t w1_size = input_dim * hidden1_dim * sizeof(float);
    size_t b1_size = hidden1_dim * sizeof(float);
    size_t w2_size = hidden1_dim * hidden2_dim * sizeof(float);
    size_t b2_size = hidden2_dim * sizeof(float);
    size_t w3_size = hidden2_dim * output_dim * sizeof(float);
    size_t b3_size = output_dim * sizeof(float);

    float *h_input = (float*)malloc(input_size);
    float *h_target = (float*)malloc(target_size);
    float *h_w1 = (float*)malloc(w1_size);
    float *h_b1 = (float*)malloc(b1_size);
    float *h_w2 = (float*)malloc(w2_size);
    float *h_b2 = (float*)malloc(b2_size);
    float *h_w3 = (float*)malloc(w3_size);
    float *h_b3 = (float*)malloc(b3_size);

    for(int i = 0; i < batch_size * input_dim; i++){
        h_input[i] = (float)rand()/RAND_MAX;
    }
    for(int i = 0; i < batch_size * output_dim; i++){
        h_target[i] = 0.0f;
    }
    for(int i = 0; i < batch_size; i++){
        int label = rand() % output_dim;
        h_target[i * output_dim + label] = 1.0f;
    }
    for(int i = 0; i < input_dim * hidden1_dim; i++){
        h_w1[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
    }
    for(int i = 0; i < hidden1_dim; i++){
        h_b1[i] = 0.0f;
    }
    for(int i = 0; i < hidden1_dim * hidden2_dim; i++){
        h_w2[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
    }
    for(int i = 0; i < hidden2_dim; i++){
        h_b2[i] = 0.0f;
    }
    for(int i = 0; i < hidden2_dim * output_dim; i++){
        h_w3[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
    }
    for(int i = 0; i < output_dim; i++){
        h_b3[i] = 0.0f;
    }

    float *d_input, *d_target, *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3;
    float *d_grad_w1, *d_grad_b1, *d_grad_w2, *d_grad_b2, *d_grad_w3, *d_grad_b3;
    CUDA_CHECK(cudaMalloc((void**)&d_input, input_size));
    CUDA_CHECK(cudaMalloc((void**)&d_target, target_size));
    CUDA_CHECK(cudaMalloc((void**)&d_w1, w1_size));
    CUDA_CHECK(cudaMalloc((void**)&d_b1, b1_size));
    CUDA_CHECK(cudaMalloc((void**)&d_w2, w2_size));
    CUDA_CHECK(cudaMalloc((void**)&d_b2, b2_size));
    CUDA_CHECK(cudaMalloc((void**)&d_w3, w3_size));
    CUDA_CHECK(cudaMalloc((void**)&d_b3, b3_size));
    CUDA_CHECK(cudaMalloc((void**)&d_grad_w1, w1_size));
    CUDA_CHECK(cudaMalloc((void**)&d_grad_b1, b1_size));
    CUDA_CHECK(cudaMalloc((void**)&d_grad_w2, w2_size));
    CUDA_CHECK(cudaMalloc((void**)&d_grad_b2, b2_size));
    CUDA_CHECK(cudaMalloc((void**)&d_grad_w3, w3_size));
    CUDA_CHECK(cudaMalloc((void**)&d_grad_b3, b3_size));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, h_target, target_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w1, h_w1, w1_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, h_b1, b1_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, h_w2, w2_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, h_b2, b2_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w3, h_w3, w3_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3, h_b3, b3_size, cudaMemcpyHostToDevice));

    int tmp_stride = 3 * hidden1_dim + 3 * hidden2_dim + 3 * output_dim;
    size_t tmp_buffer_size = batch_size * tmp_stride * sizeof(float);
    float* d_tmp_buffer;
    CUDA_CHECK(cudaMalloc((void**)&d_tmp_buffer, tmp_buffer_size));

    int blockSize = 128;
    int gridSize = (batch_size + blockSize - 1) / blockSize;

    for(int iter = 0; iter < num_iterations; iter++){
        CUDA_CHECK(cudaMemset(d_grad_w1, 0, w1_size));
        CUDA_CHECK(cudaMemset(d_grad_b1, 0, b1_size));
        CUDA_CHECK(cudaMemset(d_grad_w2, 0, w2_size));
        CUDA_CHECK(cudaMemset(d_grad_b2, 0, b2_size));
        CUDA_CHECK(cudaMemset(d_grad_w3, 0, w3_size));
        CUDA_CHECK(cudaMemset(d_grad_b3, 0, b3_size));

        compute_gradients<<<gridSize, blockSize>>>(d_input, d_target,
            d_w1, d_b1, d_w2, d_b2, d_w3, d_b3,
            d_grad_w1, d_grad_b1, d_grad_w2, d_grad_b2, d_grad_w3, d_grad_b3,
            batch_size, input_dim, hidden1_dim, hidden2_dim, output_dim,
            d_tmp_buffer, tmp_stride);
        CUDA_CHECK(cudaDeviceSynchronize());

        int numThreads = 128, numBlocks;
        numBlocks = (input_dim * hidden1_dim + numThreads - 1) / numThreads;
        update_weights<<<numBlocks, numThreads>>>(d_w1, d_grad_w1, input_dim * hidden1_dim, learning_rate, batch_size);
        numBlocks = (hidden1_dim + numThreads - 1) / numThreads;
        update_weights<<<numBlocks, numThreads>>>(d_b1, d_grad_b1, hidden1_dim, learning_rate, batch_size);
        numBlocks = (hidden1_dim * hidden2_dim + numThreads - 1) / numThreads;
        update_weights<<<numBlocks, numThreads>>>(d_w2, d_grad_w2, hidden1_dim * hidden2_dim, learning_rate, batch_size);
        numBlocks = (hidden2_dim + numThreads - 1) / numThreads;
        update_weights<<<numBlocks, numThreads>>>(d_b2, d_grad_b2, hidden2_dim, learning_rate, batch_size);
        numBlocks = (hidden2_dim * output_dim + numThreads - 1) / numThreads;
        update_weights<<<numBlocks, numThreads>>>(d_w3, d_grad_w3, hidden2_dim * output_dim, learning_rate, batch_size);
        numBlocks = (output_dim + numThreads - 1) / numThreads;
        update_weights<<<numBlocks, numThreads>>>(d_b3, d_grad_b3, output_dim, learning_rate, batch_size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(h_w1, d_w1, w1_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b1, d_b1, b1_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w2, d_w2, w2_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b2, d_b2, b2_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w3, d_w3, w3_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b3, d_b3, b3_size, cudaMemcpyDeviceToHost));

    // Visualize several samples: show input, target, forward-pass prediction, and predicted class.
    printf("Visualizations:\n");
    for(int sample = 0; sample < 5; sample++){
        printf("Sample %d:\n", sample);
        printf("Input: ");
        for(int i = 0; i < input_dim; i++){
            printf("%.3f ", h_input[sample * input_dim + i]);
        }
        printf("\nTarget: ");
        for(int i = 0; i < output_dim; i++){
            printf("%.1f ", h_target[sample * output_dim + i]);
        }
        float* prediction = (float*)malloc(output_dim * sizeof(float));
        forward_cpu(h_input + sample * input_dim,
                    h_w1, h_b1, h_w2, h_b2, h_w3, h_b3,
                    input_dim, hidden1_dim, hidden2_dim, output_dim,
                    prediction);
        printf("\nPrediction: ");
        for(int i = 0; i < output_dim; i++){
            printf("%.3f ", prediction[i]);
        }
        float max_val = -1.0f;
        int pred_class = -1;
        for(int i = 0; i < output_dim; i++){
            if(prediction[i] > max_val){
                max_val = prediction[i];
                pred_class = i;
            }
        }
        printf("\nPredicted class: %d\n\n", pred_class);
        free(prediction);
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_target));
    CUDA_CHECK(cudaFree(d_w1));
    CUDA_CHECK(cudaFree(d_b1));
    CUDA_CHECK(cudaFree(d_w2));
    CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_w3));
    CUDA_CHECK(cudaFree(d_b3));
    CUDA_CHECK(cudaFree(d_grad_w1));
    CUDA_CHECK(cudaFree(d_grad_b1));
    CUDA_CHECK(cudaFree(d_grad_w2));
    CUDA_CHECK(cudaFree(d_grad_b2));
    CUDA_CHECK(cudaFree(d_grad_w3));
    CUDA_CHECK(cudaFree(d_grad_b3));
    CUDA_CHECK(cudaFree(d_tmp_buffer));

    free(h_input);
    free(h_target);
    free(h_w1);
    free(h_b1);
    free(h_w2);
    free(h_b2);
    free(h_w3);
    free(h_b3);

    return 0;
}
