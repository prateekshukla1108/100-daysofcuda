#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Warp-level reduction helper.
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Fused transformer kernel using an online softmax update (FlashAttention style).
// Each warp processes one query vector.
// Inputs:
//   Q: [num_queries, d_head] in row-major (FP16)
//   K: [seq_len,    d_head] in row-major (FP16)
//   V: [seq_len,    d_head] in row-major (FP16)
// Output:
//   O: [num_queries, d_head] in row-major (FP16)
// scale = 1/sqrt(d_head)
__global__ void fused_transformer_flash_attention(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int num_queries,
    int seq_len,
    int d_head,
    float scale)
{
    // Each warp handles one query.
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int query_id = blockIdx.x * (blockDim.x / 32) + warp_id;
    if (query_id >= num_queries) return;

    // Determine the number of elements each thread will process.
    // (Assumes d_head is a multiple of 32 or nearly so.)
    int elems_per_thread = (d_head + 31) / 32;

    // Each thread loads its portion of the query vector into registers.
    // q_local[j] holds the query element for index (lane_id + j*32).
    float q_local[8];  // Adjust size if d_head is larger.
    #pragma unroll
    for (int j = 0; j < elems_per_thread; j++) {
        int idx = lane_id + j * 32;
        q_local[j] = (idx < d_head) ? __half2float(Q[query_id * d_head + idx]) : 0.0f;
    }

    // Initialize online softmax parameters.
    float m = -INFINITY;  // running maximum
    float l_sum = 0.0f;   // running sum of exponentials
    // Each thread holds part of the accumulator for the weighted sum of V.
    float accum[8];
    #pragma unroll
    for (int j = 0; j < elems_per_thread; j++) {
        accum[j] = 0.0f;
    }

    // Loop over all keys (and values) sequentially.
    for (int k = 0; k < seq_len; k++) {
        // --- Compute scaled dot-product: q dot k ---
        float local_dot = 0.0f;
        #pragma unroll
        for (int j = 0; j < elems_per_thread; j++) {
            int idx = lane_id + j * 32;
            if (idx < d_head) {
                float k_val = __half2float(K[k * d_head + idx]);
                local_dot += q_local[j] * k_val;
            }
        }
        // Reduce across warp to get full dot product.
        float dot = warpReduceSum(local_dot);
        // Broadcast the result so every lane has it.
        dot = __shfl_sync(0xffffffff, dot, 0);
        // Apply scaling.
        dot = dot * scale;

        // --- Update online softmax in a numerically stable way ---
        // New maximum.
        float new_m = fmaxf(m, dot);
        // Scale the previous sum.
        float exp_factor = __expf(m - new_m);
        // Compute exp for the current dot.
        float exp_dot = __expf(dot - new_m);
        // Update running sum.
        float new_l = l_sum * exp_factor + exp_dot;

        // --- Accumulate weighted V ---
        #pragma unroll
        for (int j = 0; j < elems_per_thread; j++) {
            int idx = lane_id + j * 32;
            float v_val = (idx < d_head) ? __half2float(V[k * d_head + idx]) : 0.0f;
            accum[j] = accum[j] * exp_factor + v_val * exp_dot;
        }
        // Update running parameters.
        m = new_m;
        l_sum = new_l;
    }

    // --- Write final normalized output: O = accum / l_sum ---
    #pragma unroll
    for (int j = 0; j < elems_per_thread; j++) {
        int idx = lane_id + j * 32;
        if (idx < d_head) {
            float out_val = accum[j] / l_sum;
            O[query_id * d_head + idx] = __float2half(out_val);
        }
    }
}

// Host launcher for the fused kernel.
// This example uses one warp per query.
void launch_fused_transformer_flash_attention(
    const half* Q, const half* K, const half* V, half* O,
    int num_queries, int seq_len, int d_head, cudaStream_t stream)
{
    // Each warp (32 threads) handles one query.
    const int warps_per_block = 4;  // Tune as needed.
    const int threads_per_block = warps_per_block * 32;
    // Compute the total number of warps required.
    int total_warps = num_queries;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    size_t shared_mem_size = 0;  // Not used in this version.

    float scale = 1.0f / sqrtf(static_cast<float>(d_head));
    fused_transformer_flash_attention<<<blocks, threads_per_block, shared_mem_size, stream>>>(
         Q, K, V, O, num_queries, seq_len, d_head, scale);
}

//
// Example main: allocate dummy data, launch the kernel, benchmark execution,
// and print a sample output.
//
int main()
{
    const int num_queries = 256;
    const int seq_len = 256;
    const int d_head = 64;  // Must be chosen so that (d_head + 31)/32 fits in the local array size.
    
    size_t q_size = num_queries * d_head * sizeof(half);
    size_t k_size = seq_len * d_head * sizeof(half);
    size_t v_size = seq_len * d_head * sizeof(half);
    size_t o_size = num_queries * d_head * sizeof(half);

    half* h_Q = (half*)malloc(q_size);
    half* h_K = (half*)malloc(k_size);
    half* h_V = (half*)malloc(v_size);
    half* h_O = (half*)malloc(o_size);

    // Initialize all inputs to 1.0 for testing.
    for (int i = 0; i < num_queries * d_head; i++) {
        h_Q[i] = __float2half(1.0f);
    }
    for (int i = 0; i < seq_len * d_head; i++) {
        h_K[i] = __float2half(1.0f);
        h_V[i] = __float2half(1.0f);
    }

    half *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, q_size);
    cudaMalloc(&d_K, k_size);
    cudaMalloc(&d_V, v_size);
    cudaMalloc(&d_O, o_size);

    cudaMemcpy(d_Q, h_Q, q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, k_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, v_size, cudaMemcpyHostToDevice);

    // Create CUDA events for benchmarking.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    launch_fused_transformer_flash_attention(d_Q, d_K, d_V, d_O, num_queries, seq_len, d_head, 0);
    cudaDeviceSynchronize();

    // Record the start event.
    cudaEventRecord(start);
    // Launch the kernel (for benchmarking).
    launch_fused_transformer_flash_attention(d_Q, d_K, d_V, d_O, num_queries, seq_len, d_head, 0);
    // Record the stop event.
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel execution time: %f ms\n", ms);

    // Copy the result from device to host.
    cudaMemcpy(h_O, d_O, o_size, cudaMemcpyDeviceToHost);

    printf("Output sample (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", __half2float(h_O[i]));
    }
    printf("\n");

    // Clean up.
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

