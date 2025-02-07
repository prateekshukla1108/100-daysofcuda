#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void mergeKernel(int *input, int *output, int width, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = idx * (2 * width);
    if (start >= size) return;

    int mid = min(start + width, size);
    int end = min(start + 2 * width, size);

    int i = start;
    int j = mid;
    int k = start;

    while (i < mid && j < end) {
        if (input[i] <= input[j]) {
            output[k++] = input[i++];
        } else {
            output[k++] = input[j++];
        }
    }
    while (i < mid) {
        output[k++] = input[i++];
    }
    while (j < end) {
        output[k++] = input[j++];
    }
}

void mergeSortCUDA(int *h_array, int size) {
    int *d_input, *d_output;
    size_t bytes = size * sizeof(int);

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_array, bytes, cudaMemcpyHostToDevice);

    for (int width = 1; width < size; width *= 2) {
        int numMerges = (size + (2 * width - 1)) / (2 * width);
        int threadsPerBlock = 256;
        int numBlocks = (numMerges + threadsPerBlock - 1) / threadsPerBlock;

        mergeKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, size);
        cudaDeviceSynchronize();

        int *temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    cudaMemcpy(h_array, d_input, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    int h_array[] = {9, 4, 7, 3, 2, 8, 5, 1, 6, 0};
    int size = sizeof(h_array) / sizeof(h_array[0]);

    printf("Before sorting:\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    mergeSortCUDA(h_array, size);

    printf("After sorting:\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    return 0;
}
