#include <cuda_runtime.h>
#include "cnn_mnist.h"

#define INPUT_WIDTH 28
#define INPUT_HEIGHT 28
#define FILTER_SIZE 5
#define NUM_FILTERS 16
#define CONV_OUTPUT_WIDTH (INPUT_WIDTH - FILTER_SIZE + 1)
#define CONV_OUTPUT_HEIGHT (INPUT_HEIGHT - FILTER_SIZE + 1)
#define POOL_SIZE 2
#define POOL_OUTPUT_WIDTH (CONV_OUTPUT_WIDTH / POOL_SIZE)
#define POOL_OUTPUT_HEIGHT (CONV_OUTPUT_HEIGHT / POOL_SIZE)
#define FC_INPUT_SIZE (NUM_FILTERS * POOL_OUTPUT_WIDTH * POOL_OUTPUT_HEIGHT)
#define FC_OUTPUT_SIZE 10

__global__ void cnnForwardPassKernel(CNNModel* model, float* input_batch, int batch_size, float* output_batch)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if(sample >= batch_size)return;
    float conv_out[NUM_FILTERS][CONV_OUTPUT_WIDTH * CONV_OUTPUT_HEIGHT];
    for(int f = 0; f < NUM_FILTERS; f++){
        for(int i = 0; i < CONV_OUTPUT_WIDTH * CONV_OUTPUT_HEIGHT; i++){
            conv_out[f][i] = 0.0f;
        }
    }
    for(int f = 0; f < NUM_FILTERS; f++){
        for(int y = 0; y < CONV_OUTPUT_HEIGHT; y++){
            for(int x = 0; x < CONV_OUTPUT_WIDTH; x++){
                float sum = 0.0f;
                for(int ky = 0; ky < FILTER_SIZE; ky++){
                    for(int kx = 0; kx < FILTER_SIZE; kx++){
                        int in_y = y + ky;
                        int in_x = x + kx;
                        sum += input_batch[sample * (INPUT_WIDTH * INPUT_HEIGHT) + in_y * INPUT_WIDTH + in_x] * model->d_weights[f * (FILTER_SIZE * FILTER_SIZE) + ky * FILTER_SIZE + kx];
                    }
                }
                sum += model->d_biases[f];
                conv_out[f][y * CONV_OUTPUT_WIDTH + x] = sum > 0.0f ? sum : 0.0f;
            }
        }
    }
    float pool_out[NUM_FILTERS][POOL_OUTPUT_WIDTH * POOL_OUTPUT_HEIGHT];
    for(int f = 0; f < NUM_FILTERS; f++){
        for(int py = 0; py < POOL_OUTPUT_HEIGHT; py++){
            for(int px = 0; px < POOL_OUTPUT_WIDTH; px++){
                float max_val = -1e30f;
                for(int j = 0; j < POOL_SIZE; j++){
                    for(int i = 0; i < POOL_SIZE; i++){
                        int idx_conv = (py * POOL_SIZE + j) * CONV_OUTPUT_WIDTH + (px * POOL_SIZE + i);
                        float val = conv_out[f][idx_conv];
                        if(val > max_val){
                            max_val = val;
                        }
                    }
                }
                pool_out[f][py * POOL_OUTPUT_WIDTH + px] = max_val;
            }
        }
    }
    float fc_in[FC_INPUT_SIZE];
    int idx_fc = 0;
    for(int f = 0; f < NUM_FILTERS; f++){
        for(int i = 0; i < POOL_OUTPUT_WIDTH * POOL_OUTPUT_HEIGHT; i++){
            fc_in[idx_fc++] = pool_out[f][i];
        }
    }
    float fc_out[FC_OUTPUT_SIZE];
    for (int o = 0; o < FC_OUTPUT_SIZE; o++){
        float sum = 0.0f;
        for (int i = 0; i < FC_INPUT_SIZE; i++){
            sum += fc_in[i] * model->d_weights[NUM_FILTERS * (FILTER_SIZE * FILTER_SIZE) + o * FC_INPUT_SIZE + i];
        }
        sum += model->d_biases[NUM_FILTERS + o];
        fc_out[o] = sum > 0.0f ? sum : 0.0f;
    }
    for (int o = 0; o < FC_OUTPUT_SIZE; o++){
        output_batch[sample * FC_OUTPUT_SIZE + o] = fc_out[o];
    }
}

__global__ void cnnBackpropagationKernel(CNNModel* model, float* input_batch, float* output_batch, unsigned char* labels, int batch_size)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if(sample >= batch_size)return;
    float fc_grad[FC_OUTPUT_SIZE];
    for (int o = 0; o < FC_OUTPUT_SIZE; o++){
        float pred = output_batch[sample * FC_OUTPUT_SIZE + o];
        float target = (labels[sample] == o) ? 1.0f : 0.0f;
        fc_grad[o] = pred - target;
    }
    float fc_in[FC_INPUT_SIZE];
    float conv_out[NUM_FILTERS][CONV_OUTPUT_WIDTH * CONV_OUTPUT_HEIGHT];
    for (int f = 0; f < NUM_FILTERS; f++){
        for (int y = 0; y < CONV_OUTPUT_HEIGHT; y++){
            for (int x = 0; x < CONV_OUTPUT_WIDTH; x++){
                float sum = 0.0f;
                for (int ky = 0; ky < FILTER_SIZE; ky++){
                    for (int kx = 0; kx < FILTER_SIZE; kx++){
                        int in_y = y + ky;
                        int in_x = x + kx;
                        sum += input_batch[sample * (INPUT_WIDTH * INPUT_HEIGHT) + in_y * INPUT_WIDTH + in_x] * model->d_weights[f * (FILTER_SIZE * FILTER_SIZE) + ky * FILTER_SIZE + kx];
                    }
                }
                sum += model->d_biases[f];
                conv_out[f][y * CONV_OUTPUT_WIDTH + x] = sum > 0.0f ? sum : 0.0f;
            }
        }
    }
    float pool_out[NUM_FILTERS][POOL_OUTPUT_WIDTH * POOL_OUTPUT_HEIGHT];
    int conv_index[NUM_FILTERS][POOL_OUTPUT_WIDTH * POOL_OUTPUT_HEIGHT];
    for (int f = 0; f < NUM_FILTERS; f++){
        for (int py = 0; py < POOL_OUTPUT_HEIGHT; py++){
            for (int px = 0; px < POOL_OUTPUT_WIDTH; px++){
                float max_val = -1e30f;
                int max_idx = 0;
                for (int j = 0; j < POOL_SIZE; j++){
                    for (int i = 0; i < POOL_SIZE; i++){
                        int idx_conv = (py * POOL_SIZE + j) * CONV_OUTPUT_WIDTH + (px * POOL_SIZE + i);
                        float val = conv_out[f][idx_conv];
                        if(val > max_val){
                            max_val = val;
                            max_idx = idx_conv;
                        }
                    }
                }
                pool_out[f][py * POOL_OUTPUT_WIDTH + px] = max_val;
                conv_index[f][py * POOL_OUTPUT_WIDTH + px] = max_idx;
            }
        }
    }
    int idx_fc = 0;
    for (int f = 0; f < NUM_FILTERS; f++){
        for (int i = 0; i < POOL_OUTPUT_WIDTH * POOL_OUTPUT_HEIGHT; i++){
            fc_in[idx_fc++] = pool_out[f][i];
        }
    }
    float fc_weights_grad[FC_OUTPUT_SIZE * FC_INPUT_SIZE];
    float fc_biases_grad[FC_OUTPUT_SIZE];
    for (int o = 0; o < FC_OUTPUT_SIZE; o++){
        fc_biases_grad[o] = fc_grad[o];
        for (int i = 0; i < FC_INPUT_SIZE; i++){
            fc_weights_grad[o * FC_INPUT_SIZE + i] = fc_grad[o] * fc_in[i];
        }
    }
    float fc_in_grad[FC_INPUT_SIZE];
    for (int i = 0; i < FC_INPUT_SIZE; i++){
        float sum = 0.0f;
        for (int o = 0; o < FC_OUTPUT_SIZE; o++){
            sum += fc_grad[o] * model->d_weights[NUM_FILTERS * (FILTER_SIZE * FILTER_SIZE) + o * FC_INPUT_SIZE + i];
        }
        fc_in_grad[i] = sum;
    }
    float pool_grad[NUM_FILTERS][POOL_OUTPUT_WIDTH * POOL_OUTPUT_HEIGHT];
    idx_fc = 0;
    for (int f = 0; f < NUM_FILTERS; f++){
        for (int i = 0; i < POOL_OUTPUT_WIDTH * POOL_OUTPUT_HEIGHT; i++){
            pool_grad[f][i] = fc_in_grad[idx_fc++];
        }
    }
    float conv_grad[NUM_FILTERS][CONV_OUTPUT_WIDTH * CONV_OUTPUT_HEIGHT];
    for (int f = 0; f < NUM_FILTERS; f++){
        for (int i = 0; i < CONV_OUTPUT_WIDTH * CONV_OUTPUT_HEIGHT; i++){
            conv_grad[f][i] = 0.0f;
        }
        for (int py = 0; py < POOL_OUTPUT_HEIGHT; py++){
            for (int px = 0; px < POOL_OUTPUT_WIDTH; px++){
                int idx = py * POOL_OUTPUT_WIDTH + px;
                int max_idx = conv_index[f][idx];
                conv_grad[f][max_idx] = pool_grad[f][idx];
            }
        }
    }
    for (int f = 0; f < NUM_FILTERS; f++){
        for (int ky = 0; ky < FILTER_SIZE; ky++){
            for (int kx = 0; kx < FILTER_SIZE; kx++){
                float grad_sum = 0.0f;
                for (int y = 0; y < CONV_OUTPUT_HEIGHT; y++){
                    for (int x = 0; x < CONV_OUTPUT_WIDTH; x++){
                        int idx = y * CONV_OUTPUT_WIDTH + x;
                        grad_sum += conv_grad[f][idx] * input_batch[sample * (INPUT_WIDTH * INPUT_HEIGHT) + (y + ky) * INPUT_WIDTH + (x + kx)];
                    }
                }
                atomicAdd(&model->d_weights_grad[f * (FILTER_SIZE * FILTER_SIZE) + ky * FILTER_SIZE + kx], grad_sum);
            }
        }
        float bias_grad = 0.0f;
        for (int y = 0; y < CONV_OUTPUT_HEIGHT; y++){
            for (int x = 0; x < CONV_OUTPUT_WIDTH; x++){
                bias_grad += conv_grad[f][y * CONV_OUTPUT_WIDTH + x];
            }
        }
        atomicAdd(&model->d_biases_grad[f], bias_grad);
    }
    for (int o = 0; o < FC_OUTPUT_SIZE; o++){
        for (int i = 0; i < FC_INPUT_SIZE; i++){
            atomicAdd(&model->d_weights_grad[NUM_FILTERS * (FILTER_SIZE * FILTER_SIZE) + o * FC_INPUT_SIZE + i], fc_weights_grad[o * FC_INPUT_SIZE + i]);
        }
        atomicAdd(&model->d_biases_grad[NUM_FILTERS + o], fc_biases_grad[o]);
    }
}

__global__ void sgdUpdateWeightsKernel(CNNModel* model, float learning_rate, int weights_size, int biases_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < weights_size){
        model->d_weights[idx] -= learning_rate * model->d_weights_grad[idx];
    }
    if(idx < biases_size){
        model->d_biases[idx] -= learning_rate * model->d_biases_grad[idx];
    }
}

