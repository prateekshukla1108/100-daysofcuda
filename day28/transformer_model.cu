// transformer_model.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>
#include "transformer_model.h"
#include "kernels.h"

// Helper: Initialize a device array with random values using the selected initialization method.
// If initMethod == INIT_NORMAL, we use a Gaussian distribution (mean 0, stddev 0.1)
// Otherwise, we use a uniform distribution in the range [-0.1, 0.1].
void initDeviceArray(float* d_array, int size, int initMethod) {
    float* h_array = (float*)malloc(size * sizeof(float));
    if (!h_array) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++) {
        if (initMethod == INIT_NORMAL) {
            // Box-Muller transform for Gaussian(0, 0.1)
            float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
            float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
            float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
            h_array[i] = z0 * 0.1f;
        } else {
            // Uniform distribution in range [-0.1, 0.1]
            h_array[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
        }
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_array, h_array, size * sizeof(float), cudaMemcpyHostToDevice));
    free(h_array);
}

void initModel(TransformerModel* model, int initMethod) {
    CHECK_CUDA_ERROR(cudaMalloc((void**)&(model->d_embedding), VOCAB_SIZE * EMBED_DIM * sizeof(float)));
    initDeviceArray(model->d_embedding, VOCAB_SIZE * EMBED_DIM, initMethod);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&(model->d_W_q), EMBED_DIM * EMBED_DIM * sizeof(float)));
    initDeviceArray(model->d_W_q, EMBED_DIM * EMBED_DIM, initMethod);
    CHECK_CUDA_ERROR(cudaMalloc((void**)&(model->d_W_k), EMBED_DIM * EMBED_DIM * sizeof(float)));
    initDeviceArray(model->d_W_k, EMBED_DIM * EMBED_DIM, initMethod);
    CHECK_CUDA_ERROR(cudaMalloc((void**)&(model->d_W_v), EMBED_DIM * EMBED_DIM * sizeof(float)));
    initDeviceArray(model->d_W_v, EMBED_DIM * EMBED_DIM, initMethod);
    CHECK_CUDA_ERROR(cudaMalloc((void**)&(model->d_W_o), EMBED_DIM * EMBED_DIM * sizeof(float)));
    initDeviceArray(model->d_W_o, EMBED_DIM * EMBED_DIM, initMethod);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&(model->d_W1), EMBED_DIM * FFN_HIDDEN * sizeof(float)));
    initDeviceArray(model->d_W1, EMBED_DIM * FFN_HIDDEN, initMethod);
    CHECK_CUDA_ERROR(cudaMalloc((void**)&(model->d_b1), FFN_HIDDEN * sizeof(float)));
    initDeviceArray(model->d_b1, FFN_HIDDEN, initMethod);
    CHECK_CUDA_ERROR(cudaMalloc((void**)&(model->d_W2), FFN_HIDDEN * EMBED_DIM * sizeof(float)));
    initDeviceArray(model->d_W2, FFN_HIDDEN * EMBED_DIM, initMethod);
    CHECK_CUDA_ERROR(cudaMalloc((void**)&(model->d_b2), EMBED_DIM * sizeof(float)));
    initDeviceArray(model->d_b2, EMBED_DIM, initMethod);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&(model->d_W_out), EMBED_DIM * VOCAB_SIZE * sizeof(float)));
    initDeviceArray(model->d_W_out, EMBED_DIM * VOCAB_SIZE, initMethod);
    CHECK_CUDA_ERROR(cudaMalloc((void**)&(model->d_b_out), VOCAB_SIZE * sizeof(float)));
    initDeviceArray(model->d_b_out, VOCAB_SIZE, initMethod);
}

void freeModel(TransformerModel* model) {
    CHECK_CUDA_ERROR(cudaFree(model->d_embedding));
    CHECK_CUDA_ERROR(cudaFree(model->d_W_q));
    CHECK_CUDA_ERROR(cudaFree(model->d_W_k));
    CHECK_CUDA_ERROR(cudaFree(model->d_W_v));
    CHECK_CUDA_ERROR(cudaFree(model->d_W_o));
    CHECK_CUDA_ERROR(cudaFree(model->d_W1));
    CHECK_CUDA_ERROR(cudaFree(model->d_b1));
    CHECK_CUDA_ERROR(cudaFree(model->d_W2));
    CHECK_CUDA_ERROR(cudaFree(model->d_b2));
    CHECK_CUDA_ERROR(cudaFree(model->d_W_out));
    CHECK_CUDA_ERROR(cudaFree(model->d_b_out));
}

// --- Original forward pass (host copy of logits) ---
void forwardPass(TransformerModel* model, const int* h_tokens, float* h_logits) {
    int* d_tokens;
    float* d_embedded;         // [SEQ_LENGTH x EMBED_DIM]
    float* d_attention;        // after self-attention (dummy)
    float* d_ffn_intermediate; // after first FFN layer
    float* d_ffn_output;       // final encoder output [SEQ_LENGTH x EMBED_DIM]
    float* d_logits;           // [SEQ_LENGTH x VOCAB_SIZE]

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tokens, SEQ_LENGTH * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_embedded, SEQ_LENGTH * EMBED_DIM * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_attention, SEQ_LENGTH * EMBED_DIM * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ffn_intermediate, SEQ_LENGTH * FFN_HIDDEN * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ffn_output, SEQ_LENGTH * EMBED_DIM * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_logits, SEQ_LENGTH * VOCAB_SIZE * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_tokens, h_tokens, SEQ_LENGTH * sizeof(int), cudaMemcpyHostToDevice));

    // --- Embedding Lookup ---
    dim3 blockDim_embed(256 / EMBED_DIM, EMBED_DIM);
    dim3 gridDim_embed((SEQ_LENGTH + blockDim_embed.x - 1) / blockDim_embed.x);
    embeddingLookupKernel<<<gridDim_embed, blockDim_embed>>>(d_tokens, model->d_embedding, d_embedded, SEQ_LENGTH);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // --- Positional Encoding ---
    dim3 blockDim_pos(256 / EMBED_DIM, EMBED_DIM);
    dim3 gridDim_pos((SEQ_LENGTH + blockDim_pos.x - 1) / blockDim_pos.x);
    addPositionalEncoding<<<gridDim_pos, blockDim_pos>>>(d_embedded, SEQ_LENGTH, EMBED_DIM);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // --- Dummy Self-Attention ---
    int threadsPerBlock_attn = 256;
    int blocks_attn = (SEQ_LENGTH * EMBED_DIM + threadsPerBlock_attn - 1) / threadsPerBlock_attn;
    dummySelfAttentionKernel<<<blocks_attn, threadsPerBlock_attn>>>(d_embedded, d_attention, SEQ_LENGTH);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();

    // --- Feed-Forward Network (FFN) ---
    dim3 blockDim_ffn(16, 16);
    dim3 gridDim_ffn((FFN_HIDDEN + blockDim_ffn.x - 1) / blockDim_ffn.x,
                     (SEQ_LENGTH + blockDim_ffn.y - 1) / blockDim_ffn.y);
    fusedGemmReLUKernel<<<gridDim_ffn, blockDim_ffn>>>(d_attention, model->d_W1, model->d_b1,
                                                       d_ffn_intermediate, SEQ_LENGTH, FFN_HIDDEN, EMBED_DIM);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();

    float* d_ffn_temp;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ffn_temp, SEQ_LENGTH * EMBED_DIM * sizeof(float)));
    dim3 gridDim_matmul((EMBED_DIM + blockDim_ffn.x - 1) / blockDim_ffn.x,
                        (SEQ_LENGTH + blockDim_ffn.y - 1) / blockDim_ffn.y);
    matmulKernel<<<gridDim_matmul, blockDim_ffn>>>(d_ffn_intermediate, model->d_W2, d_ffn_temp,
                                                  SEQ_LENGTH, EMBED_DIM, FFN_HIDDEN);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();

    int total = SEQ_LENGTH * EMBED_DIM;
    int threadsPerBlock_add = 256;
    int blocks_add = (total + threadsPerBlock_add - 1) / threadsPerBlock_add;
    addBiasAndResidualKernel<<<blocks_add, threadsPerBlock_add>>>(d_attention, d_ffn_temp, model->d_b2, SEQ_LENGTH, EMBED_DIM);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();

    CHECK_CUDA_ERROR(cudaMemcpy(d_ffn_output, d_ffn_temp, SEQ_LENGTH * EMBED_DIM * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaFree(d_ffn_temp));

    dim3 gridDim_linear((VOCAB_SIZE + blockDim_ffn.x - 1) / blockDim_ffn.x,
                         (SEQ_LENGTH + blockDim_ffn.y - 1) / blockDim_ffn.y);
    linearOutputKernel<<<gridDim_linear, blockDim_ffn>>>(d_ffn_output, model->d_W_out, model->d_b_out,
                                                         d_logits, SEQ_LENGTH);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();

    CHECK_CUDA_ERROR(cudaMemcpy(h_logits, d_logits, SEQ_LENGTH * VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_tokens));
    CHECK_CUDA_ERROR(cudaFree(d_embedded));
    CHECK_CUDA_ERROR(cudaFree(d_attention));
    CHECK_CUDA_ERROR(cudaFree(d_ffn_intermediate));
    CHECK_CUDA_ERROR(cudaFree(d_ffn_output));
    CHECK_CUDA_ERROR(cudaFree(d_logits));
}

// --- New: Forward pass that keeps intermediate results on device ---
void forwardPassDevice(TransformerModel* model, const int* h_tokens, float** d_logits_out, float** d_encoder_output) {
    int* d_tokens;
    float* d_embedded;
    float* d_attention;
    float* d_ffn_intermediate;
    float* d_ffn_output;
    float* d_logits;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tokens, SEQ_LENGTH * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_embedded, SEQ_LENGTH * EMBED_DIM * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_attention, SEQ_LENGTH * EMBED_DIM * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ffn_intermediate, SEQ_LENGTH * FFN_HIDDEN * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ffn_output, SEQ_LENGTH * EMBED_DIM * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_logits, SEQ_LENGTH * VOCAB_SIZE * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_tokens, h_tokens, SEQ_LENGTH * sizeof(int), cudaMemcpyHostToDevice));

    dim3 blockDim_embed(256 / EMBED_DIM, EMBED_DIM);
    dim3 gridDim_embed((SEQ_LENGTH + blockDim_embed.x - 1) / blockDim_embed.x);
    embeddingLookupKernel<<<gridDim_embed, blockDim_embed>>>(d_tokens, model->d_embedding, d_embedded, SEQ_LENGTH);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();

    dim3 blockDim_pos(256 / EMBED_DIM, EMBED_DIM);
    dim3 gridDim_pos((SEQ_LENGTH + blockDim_pos.x - 1) / blockDim_pos.x);
    addPositionalEncoding<<<gridDim_pos, blockDim_pos>>>(d_embedded, SEQ_LENGTH, EMBED_DIM);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();

    int threadsPerBlock_attn = 256;
    int blocks_attn = (SEQ_LENGTH * EMBED_DIM + threadsPerBlock_attn - 1) / threadsPerBlock_attn;
    dummySelfAttentionKernel<<<blocks_attn, threadsPerBlock_attn>>>(d_embedded, d_attention, SEQ_LENGTH);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();

    dim3 blockDim_ffn(16, 16);
    dim3 gridDim_ffn((FFN_HIDDEN + blockDim_ffn.x - 1) / blockDim_ffn.x,
                     (SEQ_LENGTH + blockDim_ffn.y - 1) / blockDim_ffn.y);
    fusedGemmReLUKernel<<<gridDim_ffn, blockDim_ffn>>>(d_attention, model->d_W1, model->d_b1,
                                                       d_ffn_intermediate, SEQ_LENGTH, FFN_HIDDEN, EMBED_DIM);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();

    float* d_ffn_temp;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ffn_temp, SEQ_LENGTH * EMBED_DIM * sizeof(float)));
    dim3 gridDim_matmul((EMBED_DIM + blockDim_ffn.x - 1) / blockDim_ffn.x,
                        (SEQ_LENGTH + blockDim_ffn.y - 1) / blockDim_ffn.y);
    matmulKernel<<<gridDim_matmul, blockDim_ffn>>>(d_ffn_intermediate, model->d_W2, d_ffn_temp,
                                                  SEQ_LENGTH, EMBED_DIM, FFN_HIDDEN);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();

    int total = SEQ_LENGTH * EMBED_DIM;
    int threadsPerBlock_add = 256;
    int blocks_add = (total + threadsPerBlock_add - 1) / threadsPerBlock_add;
    addBiasAndResidualKernel<<<blocks_add, threadsPerBlock_add>>>(d_attention, d_ffn_temp, model->d_b2, SEQ_LENGTH, EMBED_DIM);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();

    CHECK_CUDA_ERROR(cudaMemcpy(d_ffn_output, d_ffn_temp, SEQ_LENGTH * EMBED_DIM * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaFree(d_ffn_temp));

    dim3 gridDim_linear((VOCAB_SIZE + blockDim_ffn.x - 1) / blockDim_ffn.x,
                         (SEQ_LENGTH + blockDim_ffn.y - 1) / blockDim_ffn.y);
    linearOutputKernel<<<gridDim_linear, blockDim_ffn>>>(d_ffn_output, model->d_W_out, model->d_b_out,
                                                         d_logits, SEQ_LENGTH);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();

    CHECK_CUDA_ERROR(cudaFree(d_tokens));
    CHECK_CUDA_ERROR(cudaFree(d_embedded));
    CHECK_CUDA_ERROR(cudaFree(d_attention));
    CHECK_CUDA_ERROR(cudaFree(d_ffn_intermediate));

    *d_encoder_output = d_ffn_output;
    *d_logits_out = d_logits;
}

// --- Training Step: Backpropagation for the Output Layer ---
void trainStep(TransformerModel* model,
               const int* h_input_seq,
               const int* h_target_seq,
               float learning_rate) {
    float *d_logits, *d_encoder_output;
    forwardPassDevice(model, h_input_seq, &d_logits, &d_encoder_output);
    int* d_target;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_target, SEQ_LENGTH * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_target, h_target_seq, SEQ_LENGTH * sizeof(int), cudaMemcpyHostToDevice));

    float* d_lossGrad;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_lossGrad, SEQ_LENGTH * VOCAB_SIZE * sizeof(float)));

    int threads = 32;
    int blocks = (SEQ_LENGTH + threads - 1) / threads;
    softmaxAndLossGradientKernel<<<blocks, threads>>>(d_logits, d_target, d_lossGrad, SEQ_LENGTH, VOCAB_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();

    float* d_dW_out;
    float* d_db_out;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dW_out, EMBED_DIM * VOCAB_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_db_out, VOCAB_SIZE * sizeof(float)));

    dim3 blockDim_out(16, 16);
    dim3 gridDim_out((VOCAB_SIZE + blockDim_out.x - 1) / blockDim_out.x,
                     (EMBED_DIM + blockDim_out.y - 1) / blockDim_out.y);
    linearOutputBackwardKernel<<<gridDim_out, blockDim_out>>>(d_encoder_output, d_lossGrad,
                                                              d_dW_out, SEQ_LENGTH, EMBED_DIM, VOCAB_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    int threadsBias = 256;
    int blocksBias = (VOCAB_SIZE + threadsBias - 1) / threadsBias;
    linearBiasBackwardKernel<<<blocksBias, threadsBias>>>(d_lossGrad, d_db_out, SEQ_LENGTH, VOCAB_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    int size_W_out = EMBED_DIM * VOCAB_SIZE;
    int threadsUpdate = 256;
    int blocksUpdate = (size_W_out + threadsUpdate - 1) / threadsUpdate;
    gradientDescentUpdateKernel<<<blocksUpdate, threadsUpdate>>>(model->d_W_out, d_dW_out, learning_rate, size_W_out);
    CHECK_CUDA_ERROR(cudaGetLastError());

    int size_b_out = VOCAB_SIZE;
    blocksUpdate = (size_b_out + threadsUpdate - 1) / threadsUpdate;
    gradientDescentUpdateKernel<<<blocksUpdate, threadsUpdate>>>(model->d_b_out, d_db_out, learning_rate, size_b_out);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();

    cudaFree(d_target);
    cudaFree(d_lossGrad);
    cudaFree(d_dW_out);
    cudaFree(d_db_out);
    cudaFree(d_encoder_output);
    cudaFree(d_logits);
}

