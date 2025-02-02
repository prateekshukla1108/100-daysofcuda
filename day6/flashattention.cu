#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define CHECK_CUDA(call)                                       \
    {                                                          \
        cudaError_t err = call;                                \
        if (err != cudaSuccess) {                              \
            printf("CUDA error at %s:%d: %s\n",                \
                   __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    }

__global__ void flash_attention_forward(
    const float* __restrict__ Q, 
    const float* __restrict__ K, 
    const float* __restrict__ V, 
    float* __restrict__ O,       
    int M,                     
    int N,                     
    int D,                     
    float scale              
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i >= M) return;

    const float* q = Q + i * D;

    float out[128]; 
    for (int d = 0; d < D; d++) {
        out[d] = 0.0f;
    }
    float m_val = -FLT_MAX;
    float l_sum = 0.0f;

    for (int j = 0; j < N; j++) {
        const float* k = K + j * D;
        float s = 0.0f;
        for (int d = 0; d < D; d++) {
            s += q[d] * k[d];
        }
        s *= scale;  

        float m_old = m_val;
        float m_new = fmaxf(m_old, s);
        float exp_s = expf(s - m_new);
        float l_new = l_sum * expf(m_old - m_new) + exp_s;

        const float* v = V + j * D;
        for (int d = 0; d < D; d++) {
            out[d] = out[d] * expf(m_old - m_new) + exp_s * v[d];
        }
        m_val = m_new;
        l_sum = l_new;
    }
    for (int d = 0; d < D; d++) {
        out[d] /= l_sum;
    }
    float* o = O + i * D;
    for (int d = 0; d < D; d++) {
        o[d] = out[d];
    }
}

void cpu_flash_attention(const float* Q, const float* K, const float* V, float* O,
                         int M, int N, int D, float scale) {
    for (int i = 0; i < M; i++) {
        const float* q = Q + i * D;
        float* out = (float*)malloc(D * sizeof(float));
        for (int d = 0; d < D; d++) out[d] = 0.0f;
        float m_val = -FLT_MAX;
        float l_sum = 0.0f;
        for (int j = 0; j < N; j++) {
            const float* k = K + j * D;
            float s = 0.0f;
            for (int d = 0; d < D; d++) {
                s += q[d] * k[d];
            }
            s *= scale;
            float m_old = m_val;
            float m_new = fmaxf(m_old, s);
            float exp_s = expf(s - m_new);
            float l_new = l_sum * expf(m_old - m_new) + exp_s;
            const float* v = V + j * D;
            for (int d = 0; d < D; d++) {
                out[d] = out[d] * expf(m_old - m_new) + exp_s * v[d];
            }
            m_val = m_new;
            l_sum = l_new;
        }
        // Normalize and write output.
        for (int d = 0; d < D; d++) {
            O[i * D + d] = out[d] / l_sum;
        }
        free(out);
    }
}

int main() {
    int M = 128; 
    int N = 256; 
    int D = 64;  
    float scale = 1.0f / sqrtf((float)D);

    size_t sizeQ = M * D * sizeof(float);
    size_t sizeK = N * D * sizeof(float);
    size_t sizeV = N * D * sizeof(float);
    size_t sizeO = M * D * sizeof(float);

    float* h_Q = (float*)malloc(sizeQ);
    float* h_K = (float*)malloc(sizeK);
    float* h_V = (float*)malloc(sizeV);
    float* h_O_gpu = (float*)malloc(sizeO);
    float* h_O_cpu = (float*)malloc(sizeO);

    for (int i = 0; i < M * D; i++) {
        h_Q[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }
    for (int j = 0; j < N * D; j++) {
        h_K[j] = ((float)rand() / RAND_MAX) * 2 - 1;
        h_V[j] = ((float)rand() / RAND_MAX) * 2 - 1;
    }

    float *d_Q, *d_K, *d_V, *d_O;
    CHECK_CUDA(cudaMalloc((void**)&d_Q, sizeQ));
    CHECK_CUDA(cudaMalloc((void**)&d_K, sizeK));
    CHECK_CUDA(cudaMalloc((void**)&d_V, sizeV));
    CHECK_CUDA(cudaMalloc((void**)&d_O, sizeO));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, sizeQ, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, sizeK, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, sizeV, cudaMemcpyHostToDevice));

    int threadsPerBlock = 128;
    int blocks = (M + threadsPerBlock - 1) / threadsPerBlock;
    flash_attention_forward<<<blocks, threadsPerBlock>>>(d_Q, d_K, d_V, d_O, M, N, D, scale);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_O_gpu, d_O, sizeO, cudaMemcpyDeviceToHost));

    cpu_flash_attention(h_Q, h_K, h_V, h_O_cpu, M, N, D, scale);

    // Compare GPU and CPU results.
    float max_err = 0.0f;
    for (int i = 0; i < M * D; i++) {
        float err = fabs(h_O_gpu[i] - h_O_cpu[i]);
        if (err > max_err)
            max_err = err;
    }
    printf("Maximum error between GPU and CPU outputs: %f\n", max_err);

    free(h_Q); free(h_K); free(h_V); free(h_O_gpu); free(h_O_cpu);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);

    return 0;
}

