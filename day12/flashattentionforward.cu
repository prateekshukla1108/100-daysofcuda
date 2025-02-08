#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define TILE_SIZE 128
#define HEAD_DIM 64

__global__ void flash_attn_forward_kernel(const float * __restrict__ Q,
                                        const float * __restrict__ K,
                                        const float * __restrict__ V,
                                        float * __restrict__ O,
                                        int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float* q = Q + i * d;

    float out[HEAD_DIM];
#pragma unroll
    for (int x = 0; x < d; x++) {
        out[x] = 0.0f;
    }

    float m = -INFINITY;
    float l = 0.0f;

    float scale = 1.0f / sqrtf((float)d);

    float s_tile[TILE_SIZE];

    for (int tile = 0; tile < n; tile += TILE_SIZE) {
        int tile_end = (tile + TILE_SIZE < n) ? (tile + TILE_SIZE) : n;

        float tile_max = -INFINITY;
        for (int j = tile; j < tile_end; j++) {
            float s = 0.0f;
            const float* k_ptr = K + j * d;
#pragma unroll
            for (int x = 0; x < d; x++) {
                s += q[x] * k_ptr[x];
            }
            s *= scale;
            s_tile[j - tile] = s;
            if (s > tile_max) {
                tile_max = s;
            }
        }

        float new_m = (m > tile_max) ? m : tile_max;

        float tile_sum = 0.0f;
        float tile_out[HEAD_DIM];
#pragma unroll
        for (int x = 0; x < d; x++) {
            tile_out[x] = 0.0f;
        }
        for (int j = tile; j < tile_end; j++) {
            float s = s_tile[j - tile];
            float p = expf(s - new_m);
            tile_sum += p;
            const float* v_ptr = V + j * d;
#pragma unroll
            for (int x = 0; x < d; x++) {
                tile_out[x] += p * v_ptr[x];
            }
        }

        float alpha = expf(m - new_m);
        l = l * alpha + tile_sum;
#pragma unroll
        for (int x = 0; x < d; x++) {
            out[x] = out[x] * alpha + tile_out[x];
        }
        m = new_m;
    }

#pragma unroll
    for (int x = 0; x < d; x++) {
        O[i * d + x] = out[x] / l;
    }
}

extern "C" void flash_attention_forward(const float *Q, const float *K,
                                        const float *V, float *O,
                                        int n, int d) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    flash_attn_forward_kernel<<<blocks, threads>>>(Q, K, V, O, n, d);
    cudaDeviceSynchronize();
}
