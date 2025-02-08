#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define HEAD_DIM 64

__global__ void flash_attn_backward_kernel(const float * __restrict__ Q,
                                        const float * __restrict__ K,
                                        const float * __restrict__ V,
                                        const float * __restrict__ dO,
                                        float * __restrict__ dQ,
                                        float * __restrict__ dK,
                                        float * __restrict__ dV,
                                        int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float *q = Q + i * d;
    const float *dO_i = dO + i * d;
    
    float scale = 1.0f / sqrtf((float)d);

    float m_val = -INFINITY;
    for (int j = 0; j < n; j++) {
        const float *k = K + j * d;
        float s = 0.0f;
        for (int x = 0; x < d; x++) {
            s += q[x] * k[x];
        }
        s *= scale;
        if (s > m_val) m_val = s;
    }

    float L = 0.0f;
    for (int j = 0; j < n; j++) {
        const float *k = K + j * d;
        float s = 0.0f;
        for (int x = 0; x < d; x++) {
            s += q[x] * k[x];
        }
        s *= scale;
        L += expf(s - m_val);
    }

    float A = 0.0f;
    for (int j = 0; j < n; j++) {
        const float *k = K + j * d;
        float s = 0.0f;
        for (int x = 0; x < d; x++) {
            s += q[x] * k[x];
        }
        s *= scale;
        float p = expf(s - m_val) / L;
        const float *v = V + j * d;
        float dot_v = 0.0f;
        for (int x = 0; x < d; x++) {
            dot_v += v[x] * dO_i[x];
        }
        A += p * dot_v;
    }

    float dQ_acc[HEAD_DIM];
    for (int x = 0; x < d; x++) {
        dQ_acc[x] = 0.0f;
    }

    for (int j = 0; j < n; j++) {
        const float *k = K + j * d;
        float s = 0.0f;
        for (int x = 0; x < d; x++) {
            s += q[x] * k[x];
        }
        s *= scale;
        float p = expf(s - m_val) / L;
        const float *v = V + j * d;
        float dot_v = 0.0f;
        for (int x = 0; x < d; x++) {
            dot_v += v[x] * dO_i[x];
        }
        float dS = p * (dot_v - A);
        
        for (int x = 0; x < d; x++) {
            dQ_acc[x] += scale * dS * k[x];
        }
        for (int x = 0; x < d; x++) {
            atomicAdd(&dK[j * d + x], scale * dS * q[x]);
        }
        for (int x = 0; x < d; x++) {
            atomicAdd(&dV[j * d + x], p * dO_i[x]);
        }
    }

    for (int x = 0; x < d; x++) {
        dQ[i * d + x] = dQ_acc[x];
    }
}

extern "C" void flash_attention_backward(const float *Q, const float *K, const float *V,
                                        const float *dO,
                                        float *dQ, float *dK, float *dV,
                                        int n, int d) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    flash_attn_backward_kernel<<<blocks, threads>>>(Q, K, V, dO, dQ, dK, dV, n, d);
    cudaDeviceSynchronize();
}
