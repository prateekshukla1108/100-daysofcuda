#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>
using namespace cooperative_groups;
#define WARPS_PER_BLOCK 4
#define TILE_SIZE 64
#define WINDOW_SIZE 512
#define COMPRESSED_BLOCK 32
#define SELECTED_BLOCKS 16
#define WARP_SIZE 32

__inline__ __device__ float warp_reduce_sum(float val, int lane) {
    for(int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return __shfl_sync(0xffffffff, val, 0);
}

__global__ void nsa_kernel(const half* __restrict__ queries, const half* __restrict__ keys, const half* __restrict__ values, const bool* __restrict__ mask, const half* __restrict__ gates, half* output, int batch_size, int num_heads, int seq_len, int d_k, int d_v) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int t = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if(t >= seq_len) return;
    extern __shared__ half smem[];
    half* query_smem = smem;
    half* sum_key_cmp = smem + WARPS_PER_BLOCK * d_k;
    int query_base = b * num_heads * seq_len * d_k + h * seq_len * d_k;
    int key_base = b * num_heads * seq_len * d_k + h * seq_len * d_k;
    int gate_base = b * seq_len * 3 + t * 3;
    const half* q_ptr = queries + query_base + t * d_k;
    for(int i = lane; i < d_k; i += WARP_SIZE) {
        query_smem[warp_id * d_k + i] = q_ptr[i];
    }
    __syncthreads();
    int compressed_len = (seq_len + COMPRESSED_BLOCK - 1) / COMPRESSED_BLOCK;
    if(warp_id < (d_k / WARP_SIZE)) {
        int dim = warp_id * WARP_SIZE + lane;
        if(dim < d_k) {
            float sum = 0.0f;
            for(int k = 0; k < compressed_len; ++k) {
                int key_idx = k * COMPRESSED_BLOCK;
                if(key_idx < seq_len) {
                    sum += __half2float(keys[key_base + key_idx * d_k + dim]);
                }
            }
            sum_key_cmp[dim] = __float2half(sum);
        }
    }
    __syncthreads();
    float partial_cmp = 0.0f;
    float partial_win = 0.0f;
    float partial_slc = 0.0f;
    const half* qvec = &query_smem[warp_id * d_k];
    for(int i = lane; i < d_k; i += WARP_SIZE) {
        partial_cmp += __half2float(qvec[i]) * __half2float(sum_key_cmp[i]);
    }
    int window_start = (t - WINDOW_SIZE > 0) ? (t - WINDOW_SIZE) : 0;
    for(int k = window_start; k < t; ++k) {
        const half* key_ptr = keys + key_base + k * d_k;
        for(int i = lane; i < d_k; i += WARP_SIZE) {
            partial_win += __half2float(qvec[i]) * __half2float(key_ptr[i]);
        }
    }
    int selected_indices[SELECTED_BLOCKS];
    for(int i = 0; i < SELECTED_BLOCKS; ++i) {
        selected_indices[i] = (t + i * 64) % seq_len;
    }
    for(int sb = 0; sb < SELECTED_BLOCKS; ++sb) {
        int block_start = selected_indices[sb];
        for(int m = 0; m < 64; ++m) {
            int global_k = block_start + m;
            if(global_k < seq_len) {
                const half* key_ptr = keys + key_base + global_k * d_k;
                for(int i = lane; i < d_k; i += WARP_SIZE) {
                    partial_slc += __half2float(qvec[i]) * __half2float(key_ptr[i]);
                }
            }
        }
    }
    float attn_cmp = warp_reduce_sum(partial_cmp, lane);
    float attn_win = warp_reduce_sum(partial_win, lane);
    float attn_slc = warp_reduce_sum(partial_slc, lane);
    float gate_cmp = __half2float(gates[gate_base]);
    float gate_slc = __half2float(gates[gate_base + 1]);
    float gate_win = __half2float(gates[gate_base + 2]);
    float result = (gate_cmp * attn_cmp) + (gate_slc * attn_slc) + (gate_win * attn_win);
    int out_offset = b * num_heads * seq_len * d_v + h * seq_len * d_v + t * d_v;
    if(lane < d_v) {
        output[out_offset + lane] = __float2half(result);
    }
}

int main() {
    int batch_size = 1, num_heads = 1, seq_len = 1024, d_k = 64, d_v = 64;
    size_t q_size = batch_size * num_heads * seq_len * d_k;
    size_t k_size = q_size;
    size_t v_size = batch_size * num_heads * seq_len * d_v;
    size_t gates_size = batch_size * seq_len * 3;
    size_t output_size = batch_size * num_heads * seq_len * d_v;
    half *h_queries = new half[q_size];
    half *h_keys = new half[k_size];
    half *h_values = new half[q_size];
    half *h_gates = new half[gates_size];
    half *h_output = new half[output_size];
    bool *h_mask = new bool[q_size];
    for(size_t i = 0; i < q_size; i++) {
        h_queries[i] = __float2half(1.0f);
        h_keys[i] = __float2half(1.0f);
        h_values[i] = __float2half(1.0f);
        h_mask[i] = true;
    }
    for(size_t i = 0; i < gates_size; i++) {
        h_gates[i] = __float2half(0.33f);
    }
    half *d_queries, *d_keys, *d_values, *d_gates, *d_output;
    bool *d_mask;
    cudaMalloc(&d_queries, q_size * sizeof(half));
    cudaMalloc(&d_keys, k_size * sizeof(half));
    cudaMalloc(&d_values, q_size * sizeof(half));
    cudaMalloc(&d_gates, gates_size * sizeof(half));
    cudaMalloc(&d_output, output_size * sizeof(half));
    cudaMalloc(&d_mask, q_size * sizeof(bool));
    cudaMemcpy(d_queries, h_queries, q_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_keys, h_keys, k_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, q_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gates, h_gates, gates_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, q_size * sizeof(bool), cudaMemcpyHostToDevice);
    dim3 grid((seq_len + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, num_heads, batch_size);
    dim3 block(WARPS_PER_BLOCK * WARP_SIZE, 1, 1);
    size_t sharedMemSize = (WARPS_PER_BLOCK + 1) * d_k * sizeof(half);
    nsa_kernel<<<grid, block, sharedMemSize>>>(d_queries, d_keys, d_values, d_mask, d_gates, d_output, batch_size, num_heads, seq_len, d_k, d_v);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, output_size * sizeof(half), cudaMemcpyDeviceToHost);
    std::cout << "Output[0]: " << __half2float(h_output[0]) << "\n";
    cudaFree(d_queries);
    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_gates);
    cudaFree(d_output);
    cudaFree(d_mask);
    delete[] h_queries;
    delete[] h_keys;
    delete[] h_values;
    delete[] h_gates;
    delete[] h_output;
    delete[] h_mask;
    return 0;
}
