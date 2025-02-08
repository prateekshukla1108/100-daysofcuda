#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define MAX_FEATURE_DIM 128

__global__ void graph_conv_kernel_norm(const int *row_ptr, const int *col_idx,
                                     const float *in_features, float *out_features,
                                     const float *weight, const float *norm,
                                     int num_nodes, int feature_dim, int out_dim) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node < num_nodes) {
        float agg[MAX_FEATURE_DIM];
        for (int f = 0; f < feature_dim; f++) {
            agg[f] = 0.0f;
        }
        float self_scale = 1.0f / (norm[node] * norm[node]);
        for (int f = 0; f < feature_dim; f++) {
            agg[f] += in_features[node * feature_dim + f] * self_scale;
        }
        int start = row_ptr[node];
        int end   = row_ptr[node + 1];
        for (int idx = start; idx < end; idx++) {
            int nbr = col_idx[idx];
            float scale = 1.0f / (norm[node] * norm[nbr]);
            for (int f = 0; f < feature_dim; f++) {
                agg[f] += in_features[nbr * feature_dim + f] * scale;
            }
        }
        for (int j = 0; j < out_dim; j++) {
            float sum = 0.0f;
            for (int f = 0; f < feature_dim; f++) {
                sum += agg[f] * weight[f * out_dim + j];
            }
            out_features[node * out_dim + j] = sum;
        }
    }
}

int main() {
    int num_nodes = 4;
    int h_row_ptr[] = {0, 2, 5, 8, 10};
    int h_col_idx[] = {
        1, 2,
        0, 2, 3,
        0, 1, 3,
        1, 2
    };

    int feature_dim = 3;
    int out_dim = 2;
    float h_in_features[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f
    };

    float h_weight[] = {
        0.1f, 0.2f,
        0.3f, 0.4f,
        0.5f, 0.6f
    };

    float h_norm[4];
    for (int i = 0; i < num_nodes; i++) {
        int degree = h_row_ptr[i + 1] - h_row_ptr[i];
        h_norm[i] = sqrtf(degree + 1.0f);
    }

    float h_out_features[4 * 2];

    int *d_row_ptr, *d_col_idx;
    float *d_in_features, *d_out_features, *d_weight, *d_norm;
    cudaMalloc((void**)&d_row_ptr, sizeof(int) * (num_nodes + 1));
    cudaMalloc((void**)&d_col_idx, sizeof(int) * 10);
    cudaMalloc((void**)&d_in_features, sizeof(float) * num_nodes * feature_dim);
    cudaMalloc((void**)&d_out_features, sizeof(float) * num_nodes * out_dim);
    cudaMalloc((void**)&d_weight, sizeof(float) * feature_dim * out_dim);
    cudaMalloc((void**)&d_norm, sizeof(float) * num_nodes);

    cudaMemcpy(d_row_ptr, h_row_ptr, sizeof(int) * (num_nodes + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, sizeof(int) * 10, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_features, h_in_features, sizeof(float) * num_nodes * feature_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, sizeof(float) * feature_dim * out_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_norm, h_norm, sizeof(float) * num_nodes, cudaMemcpyHostToDevice);

    int threads_per_block = 128;
    int blocks = (num_nodes + threads_per_block - 1) / threads_per_block;
    graph_conv_kernel_norm<<<blocks, threads_per_block>>>(d_row_ptr, d_col_idx,
                                                         d_in_features, d_out_features,
                                                         d_weight, d_norm,
                                                         num_nodes, feature_dim, out_dim);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out_features, d_out_features, sizeof(float) * num_nodes * out_dim, cudaMemcpyDeviceToHost);

    printf("Output features from the normalized GCN layer:\n");
    for (int i = 0; i < num_nodes; i++) {
        printf("Node %d: ", i);
        for (int j = 0; j < out_dim; j++) {
            printf("%f ", h_out_features[i * out_dim + j]);
        }
        printf("\n");
    }

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_in_features);
    cudaFree(d_out_features);
    cudaFree(d_weight);
    cudaFree(d_norm);

    return 0;
}
