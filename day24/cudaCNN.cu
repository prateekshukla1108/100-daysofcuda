#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
  if(code != cudaSuccess){
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if(abort) exit(code);
  }
}

#define TRAIN_SAMPLES 60000
#define BATCH_SIZE 64
#define IN_CHANNELS 1
#define IN_HEIGHT 28
#define IN_WIDTH 28
#define CONV_KERNEL_SIZE 5
#define CONV_OUT_CHANNELS 8
#define CONV_OUT_HEIGHT (IN_HEIGHT - CONV_KERNEL_SIZE + 1)
#define CONV_OUT_WIDTH  (IN_WIDTH - CONV_KERNEL_SIZE + 1)
#define POOL_SIZE 2
#define POOL_OUT_HEIGHT (CONV_OUT_HEIGHT/POOL_SIZE)
#define POOL_OUT_WIDTH  (CONV_OUT_WIDTH/POOL_SIZE)
#define FC_INPUT_SIZE (CONV_OUT_CHANNELS*POOL_OUT_HEIGHT*POOL_OUT_WIDTH)
#define FC_OUTPUT_SIZE 10
#define OPTIMIZER_SGD 0
#define OPTIMIZER_ADAM 1

__global__ void fused_train_kernel(
    const float* d_input,
    const int*   d_labels,
    float* d_conv_w,
    float* d_conv_b,
    float* d_fc_w,
    float* d_fc_b,
    float* d_conv_w_grad,
    float* d_conv_b_grad,
    float* d_fc_w_grad,
    float* d_fc_b_grad,
    float* d_conv_w_m, float* d_conv_w_v,
    float* d_conv_b_m, float* d_conv_b_v,
    float* d_fc_w_m,   float* d_fc_w_v,
    float* d_fc_b_m,   float* d_fc_b_v,
    float lr, int batch_size, int t, float beta1, float beta2, float eps,
    float* d_loss
){
    if(threadIdx.x == 0) {
        int conv_w_size = CONV_OUT_CHANNELS * IN_CHANNELS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE;
        for (int i = 0; i < conv_w_size; i++)
            d_conv_w_grad[i] = 0.0f;
        for (int i = 0; i < CONV_OUT_CHANNELS; i++)
            d_conv_b_grad[i] = 0.0f;
        int fc_w_size = FC_OUTPUT_SIZE * FC_INPUT_SIZE;
        for (int i = 0; i < fc_w_size; i++)
            d_fc_w_grad[i] = 0.0f;
        for (int i = 0; i < FC_OUTPUT_SIZE; i++)
            d_fc_b_grad[i] = 0.0f;
        d_loss[0] = 0.0f;
    }
    __syncthreads();

    int s = threadIdx.x;
    if (s < batch_size) {
        const float* input_sample = d_input + s * IN_CHANNELS * IN_HEIGHT * IN_WIDTH;
        float conv_out[CONV_OUT_CHANNELS * CONV_OUT_HEIGHT * CONV_OUT_WIDTH];
        int pool_argmax[CONV_OUT_CHANNELS * POOL_OUT_HEIGHT * POOL_OUT_WIDTH];
        float pool_out[FC_INPUT_SIZE];
        float fc_out[FC_OUTPUT_SIZE];
        float fc_grad[FC_OUTPUT_SIZE];
        float fc_in_grad[FC_INPUT_SIZE];
        float conv_grad[CONV_OUT_CHANNELS * CONV_OUT_HEIGHT * CONV_OUT_WIDTH];
        for (int i = 0; i < CONV_OUT_CHANNELS * CONV_OUT_HEIGHT * CONV_OUT_WIDTH; i++)
            conv_grad[i] = 0.0f;
        for (int oc = 0; oc < CONV_OUT_CHANNELS; oc++) {
            for (int oy = 0; oy < CONV_OUT_HEIGHT; oy++) {
                for (int ox = 0; ox < CONV_OUT_WIDTH; ox++) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < IN_CHANNELS; ic++) {
                        for (int ky = 0; ky < CONV_KERNEL_SIZE; ky++) {
                            for (int kx = 0; kx < CONV_KERNEL_SIZE; kx++) {
                                int in_x = ox + kx;
                                int in_y = oy + ky;
                                int input_idx = ic * (IN_HEIGHT * IN_WIDTH) + in_y * IN_WIDTH + in_x;
                                int weight_idx = oc * (IN_CHANNELS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE)
                                               + ic * (CONV_KERNEL_SIZE * CONV_KERNEL_SIZE)
                                               + ky * CONV_KERNEL_SIZE + kx;
                                sum += input_sample[input_idx] * d_conv_w[weight_idx];
                            }
                        }
                    }
                    sum += d_conv_b[oc];
                    if(sum < 0) sum = 0;
                    int out_idx = oc * (CONV_OUT_HEIGHT * CONV_OUT_WIDTH) + oy * CONV_OUT_WIDTH + ox;
                    conv_out[out_idx] = sum;
                }
            }
        }
        for (int c = 0; c < CONV_OUT_CHANNELS; c++) {
            for (int py = 0; py < POOL_OUT_HEIGHT; py++) {
                for (int px = 0; px < POOL_OUT_WIDTH; px++) {
                    int pool_index = c * (POOL_OUT_HEIGHT * POOL_OUT_WIDTH) + py * POOL_OUT_WIDTH + px;
                    float max_val = -1e20f;
                    int max_ind = 0;
                    int start_y = py * POOL_SIZE;
                    int start_x = px * POOL_SIZE;
                    for (int dy = 0; dy < POOL_SIZE; dy++) {
                        for (int dx = 0; dx < POOL_SIZE; dx++) {
                            int oy = start_y + dy;
                            int ox = start_x + dx;
                            int conv_index = c * (CONV_OUT_HEIGHT * CONV_OUT_WIDTH) + oy * CONV_OUT_WIDTH + ox;
                            float val = conv_out[conv_index];
                            if(val > max_val){
                                max_val = val;
                                max_ind = conv_index;
                            }
                        }
                    }
                    pool_out[pool_index] = max_val;
                    pool_argmax[pool_index] = max_ind;
                }
            }
        }
        for (int neuron = 0; neuron < FC_OUTPUT_SIZE; neuron++) {
            float sum = d_fc_b[neuron];
            for (int i = 0; i < FC_INPUT_SIZE; i++) {
                int weight_idx = neuron * FC_INPUT_SIZE + i;
                sum += pool_out[i] * d_fc_w[weight_idx];
            }
            fc_out[neuron] = sum;
        }
        float max_val = -1e20f;
        for (int i = 0; i < FC_OUTPUT_SIZE; i++) {
            if(fc_out[i] > max_val) max_val = fc_out[i];
        }
        float exp_sum = 0.0f;
        for (int i = 0; i < FC_OUTPUT_SIZE; i++) {
            float exp_val = expf(fc_out[i] - max_val);
            exp_sum += exp_val;
            fc_grad[i] = exp_val;
        }
        int label = d_labels[s];
        float p = fc_grad[label] / exp_sum;
        if(p < 1e-8f) p = 1e-8f;
        float loss = -logf(p);
        for (int i = 0; i < FC_OUTPUT_SIZE; i++)
            fc_grad[i] = fc_grad[i] / exp_sum;
        fc_grad[label] -= 1.0f;
        atomicAdd(&d_loss[0], loss);
        for (int i = 0; i < FC_OUTPUT_SIZE; i++) {
            atomicAdd(&d_fc_b_grad[i], fc_grad[i] / batch_size);
            for (int j = 0; j < FC_INPUT_SIZE; j++) {
                int idx_grad = i * FC_INPUT_SIZE + j;
                float grad_val = fc_grad[i] * pool_out[j] / batch_size;
                atomicAdd(&d_fc_w_grad[idx_grad], grad_val);
            }
        }
        for (int i = 0; i < FC_INPUT_SIZE; i++) {
            float sum = 0.0f;
            for (int neuron = 0; neuron < FC_OUTPUT_SIZE; neuron++) {
                int idx_w = neuron * FC_INPUT_SIZE + i;
                sum += d_fc_w[idx_w] * fc_grad[neuron];
            }
            fc_in_grad[i] = sum;
        }
        float pool_grad[CONV_OUT_CHANNELS * CONV_OUT_HEIGHT * CONV_OUT_WIDTH];
        for (int i = 0; i < CONV_OUT_CHANNELS * CONV_OUT_HEIGHT * CONV_OUT_WIDTH; i++)
            pool_grad[i] = 0.0f;
        for (int c = 0; c < CONV_OUT_CHANNELS; c++) {
            for (int p_idx = 0; p_idx < (POOL_OUT_HEIGHT * POOL_OUT_WIDTH); p_idx++) {
                int pool_index = c * (POOL_OUT_HEIGHT * POOL_OUT_WIDTH) + p_idx;
                float grad_val = fc_in_grad[pool_index];
                int argmax_idx = pool_argmax[pool_index];
                pool_grad[argmax_idx] = grad_val;
            }
        }
        for (int i = 0; i < CONV_OUT_CHANNELS * CONV_OUT_HEIGHT * CONV_OUT_WIDTH; i++) {
            if(conv_out[i] <= 0)
                pool_grad[i] = 0.0f;
        }
        for (int i = 0; i < CONV_OUT_CHANNELS * CONV_OUT_HEIGHT * CONV_OUT_WIDTH; i++)
            conv_grad[i] += pool_grad[i];
        for (int oc = 0; oc < CONV_OUT_CHANNELS; oc++) {
            for (int ic = 0; ic < IN_CHANNELS; ic++) {
                for (int ky = 0; ky < CONV_KERNEL_SIZE; ky++) {
                    for (int kx = 0; kx < CONV_KERNEL_SIZE; kx++) {
                        float sum = 0.0f;
                        for (int oy = 0; oy < CONV_OUT_HEIGHT; oy++) {
                            for (int ox = 0; ox < CONV_OUT_WIDTH; ox++) {
                                int in_y = oy + ky;
                                int in_x = ox + kx;
                                int input_idx = ic * (IN_HEIGHT * IN_WIDTH) + in_y * IN_WIDTH + in_x;
                                int conv_index = oc * (CONV_OUT_HEIGHT * CONV_OUT_WIDTH) + oy * CONV_OUT_WIDTH + ox;
                                sum += input_sample[input_idx] * conv_grad[conv_index];
                            }
                        }
                        int weight_idx = oc * (IN_CHANNELS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE)
                                         + ic * (CONV_KERNEL_SIZE * CONV_KERNEL_SIZE)
                                         + ky * CONV_KERNEL_SIZE + kx;
                        atomicAdd(&d_conv_w_grad[weight_idx], sum / batch_size);
                    }
                }
            }
            float sum_b = 0.0f;
            for (int i = 0; i < CONV_OUT_HEIGHT * CONV_OUT_WIDTH; i++) {
                int idx_conv = oc * (CONV_OUT_HEIGHT * CONV_OUT_WIDTH) + i;
                sum_b += conv_grad[idx_conv];
            }
            atomicAdd(&d_conv_b_grad[oc], sum_b / batch_size);
        }
    }
    __syncthreads();
    if(threadIdx.x == 0) {
        int conv_w_size = CONV_OUT_CHANNELS * IN_CHANNELS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE;
        for (int i = 0; i < conv_w_size; i++) {
            d_conv_w_m[i] = beta1 * d_conv_w_m[i] + (1 - beta1) * d_conv_w_grad[i];
            d_conv_w_v[i] = beta2 * d_conv_w_v[i] + (1 - beta2) * d_conv_w_grad[i] * d_conv_w_grad[i];
            float m_hat = d_conv_w_m[i] / (1 - powf(beta1, (float)t));
            float v_hat = d_conv_w_v[i] / (1 - powf(beta2, (float)t));
            d_conv_w[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
        }
        int conv_b_size = CONV_OUT_CHANNELS;
        for (int i = 0; i < conv_b_size; i++) {
            d_conv_b_m[i] = beta1 * d_conv_b_m[i] + (1 - beta1) * d_conv_b_grad[i];
            d_conv_b_v[i] = beta2 * d_conv_b_v[i] + (1 - beta2) * d_conv_b_grad[i] * d_conv_b_grad[i];
            float m_hat = d_conv_b_m[i] / (1 - powf(beta1, (float)t));
            float v_hat = d_conv_b_v[i] / (1 - powf(beta2, (float)t));
            d_conv_b[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
        }
        int fc_w_size = FC_OUTPUT_SIZE * FC_INPUT_SIZE;
        for (int i = 0; i < fc_w_size; i++) {
            d_fc_w_m[i] = beta1 * d_fc_w_m[i] + (1 - beta1) * d_fc_w_grad[i];
            d_fc_w_v[i] = beta2 * d_fc_w_v[i] + (1 - beta2) * d_fc_w_grad[i] * d_fc_w_grad[i];
            float m_hat = d_fc_w_m[i] / (1 - powf(beta1, (float)t));
            float v_hat = d_fc_w_v[i] / (1 - powf(beta2, (float)t));
            d_fc_w[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
        }
        int fc_b_size = FC_OUTPUT_SIZE;
        for (int i = 0; i < fc_b_size; i++) {
            d_fc_b_m[i] = beta1 * d_fc_b_m[i] + (1 - beta1) * d_fc_b_grad[i];
            d_fc_b_v[i] = beta2 * d_fc_b_v[i] + (1 - beta2) * d_fc_b_grad[i] * d_fc_b_grad[i];
            float m_hat = d_fc_b_m[i] / (1 - powf(beta1, (float)t));
            float v_hat = d_fc_b_v[i] / (1 - powf(beta2, (float)t));
            d_fc_b[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
        }
    }
}

void init_array(float *a, int n){
  for(int i = 0; i < n; i++){
    a[i] = (((float)rand() / RAND_MAX) - 0.5f) * 0.1f;
  }
}

int main(){
    int train_samples = TRAIN_SAMPLES;
    float *train_images = (float*)malloc(train_samples * IN_CHANNELS * IN_HEIGHT * IN_WIDTH * sizeof(float));
    int   *train_labels = (int*)malloc(train_samples * sizeof(int));
    FILE *fimg = fopen("mnist_train_images.bin", "rb");
    if(!fimg){ printf("Error opening mnist_train_images.bin\n"); exit(1); }
    fread(train_images, sizeof(float), train_samples * IN_CHANNELS * IN_HEIGHT * IN_WIDTH, fimg);
    fclose(fimg);
    FILE *flab = fopen("mnist_train_labels.bin", "rb");
    if(!flab){ printf("Error opening mnist_train_labels.bin\n"); exit(1); }
    fread(train_labels, sizeof(int), train_samples, flab);
    fclose(flab);
    for(int i = 0; i < train_samples * IN_CHANNELS * IN_HEIGHT * IN_WIDTH; i++){
      train_images[i] /= 255.0f;
    }

    int conv_w_elems = CONV_OUT_CHANNELS * IN_CHANNELS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE;
    int conv_b_elems = CONV_OUT_CHANNELS;
    int fc_w_elems = FC_OUTPUT_SIZE * FC_INPUT_SIZE;
    int fc_b_elems = FC_OUTPUT_SIZE;
    float *h_conv_w = (float*)malloc(conv_w_elems * sizeof(float));
    float *h_conv_b = (float*)malloc(conv_b_elems * sizeof(float));
    float *h_fc_w   = (float*)malloc(fc_w_elems * sizeof(float));
    float *h_fc_b   = (float*)malloc(fc_b_elems * sizeof(float));
    init_array(h_conv_w, conv_w_elems);
    init_array(h_conv_b, conv_b_elems);
    init_array(h_fc_w, fc_w_elems);
    init_array(h_fc_b, fc_b_elems);

    float *d_input, *d_conv_w, *d_conv_b, *d_fc_w, *d_fc_b;
    float *d_conv_w_grad, *d_conv_b_grad, *d_fc_w_grad, *d_fc_b_grad;
    float *d_conv_w_m, *d_conv_w_v, *d_conv_b_m, *d_conv_b_v;
    float *d_fc_w_m, *d_fc_w_v, *d_fc_b_m, *d_fc_b_v;
    int   *d_labels;
    float *d_loss;

    CUDA_CHECK(cudaMalloc(&d_input, BATCH_SIZE * IN_CHANNELS * IN_HEIGHT * IN_WIDTH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_conv_w, conv_w_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv_b, conv_b_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc_w,   fc_w_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc_b,   fc_b_elems * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_conv_w_grad, conv_w_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv_b_grad, conv_b_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc_w_grad,   fc_w_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc_b_grad,   fc_b_elems * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_conv_w_m, conv_w_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv_w_v, conv_w_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv_b_m, conv_b_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv_b_v, conv_b_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc_w_m,   fc_w_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc_w_v,   fc_w_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc_b_m,   fc_b_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc_b_v,   fc_b_elems * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_conv_w, h_conv_w, conv_w_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv_b, h_conv_b, conv_b_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fc_w,   h_fc_w,   fc_w_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fc_b,   h_fc_b,   fc_b_elems * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_conv_w_m, 0, conv_w_elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_conv_w_v, 0, conv_w_elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_conv_b_m, 0, conv_b_elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_conv_b_v, 0, conv_b_elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_fc_w_m,   0, fc_w_elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_fc_w_v,   0, fc_w_elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_fc_b_m,   0, fc_b_elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_fc_b_v,   0, fc_b_elems * sizeof(float)));

    int opt = OPTIMIZER_ADAM;
    float lr_val = 0.0001f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    int epochs = 10;
    int t = 1;

    for(int epoch = 0; epoch < epochs; epoch++){
        float epoch_loss = 0.0f;
        int batches = train_samples / BATCH_SIZE;
        for (int b = 0; b < batches; b++){
            float* h_batch_input = train_images + b * BATCH_SIZE * IN_CHANNELS * IN_HEIGHT * IN_WIDTH;
            int*   h_batch_labels = train_labels + b * BATCH_SIZE;
            CUDA_CHECK(cudaMemcpy(d_input, h_batch_input, BATCH_SIZE * IN_CHANNELS * IN_HEIGHT * IN_WIDTH * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_labels, h_batch_labels, BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice));

            fused_train_kernel<<<1, BATCH_SIZE>>>(d_input, d_labels,
                d_conv_w, d_conv_b, d_fc_w, d_fc_b,
                d_conv_w_grad, d_conv_b_grad, d_fc_w_grad, d_fc_b_grad,
                d_conv_w_m, d_conv_w_v, d_conv_b_m, d_conv_b_v,
                d_fc_w_m, d_fc_w_v, d_fc_b_m, d_fc_b_v,
                lr_val, BATCH_SIZE, t, beta1, beta2, eps, d_loss);
            CUDA_CHECK(cudaDeviceSynchronize());

            float loss_batch;
            CUDA_CHECK(cudaMemcpy(&loss_batch, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
            epoch_loss += loss_batch / BATCH_SIZE;

            t++;
        }
        printf("Epoch %d: Average Loss = %.4f\n", epoch, epoch_loss / batches);
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    }
    printf("Training complete.\n");

    cudaFree(d_input);
    cudaFree(d_labels);
    cudaFree(d_conv_w);
    cudaFree(d_conv_b);
    cudaFree(d_fc_w);
    cudaFree(d_fc_b);
    cudaFree(d_conv_w_grad);
    cudaFree(d_conv_b_grad);
    cudaFree(d_fc_w_grad);
    cudaFree(d_fc_b_grad);
    cudaFree(d_conv_w_m);
    cudaFree(d_conv_w_v);
    cudaFree(d_conv_b_m);
    cudaFree(d_conv_b_v);
    cudaFree(d_fc_w_m);
    cudaFree(d_fc_w_v);
    cudaFree(d_fc_b_m);
    cudaFree(d_fc_b_v);
    cudaFree(d_loss);
    free(train_images);
    free(train_labels);
    free(h_conv_w);
    free(h_conv_b);
    free(h_fc_w);
    free(h_fc_b);

    return 0;
}

