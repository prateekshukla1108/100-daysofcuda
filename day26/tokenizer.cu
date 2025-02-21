#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string.h>

__device__ inline bool is_whitespace(char c) {
    return (c == ' ' || c == '\n' || c == '\t');
}

__global__ void tokenizeKernel(const char* input, int inputLength, char* output, int* outputOffset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= inputLength) return;

    if (!is_whitespace(input[idx]) && (idx == 0 || is_whitespace(input[idx - 1]))) {
        int tokenStart = idx;
        int tokenEnd = idx;
        while (tokenEnd < inputLength && !is_whitespace(input[tokenEnd])) {
            tokenEnd++;
        }
        int tokenLength = tokenEnd - tokenStart;
        int pos = atomicAdd(outputOffset, tokenLength + 1);
        for (int j = 0; j < tokenLength; j++) {
            output[pos + j] = input[tokenStart + j];
        }
        output[pos + tokenLength] = '\n';
    }
}

int main() {
    const char inputFileName[] = "input.txt";
    FILE* fp = fopen(inputFileName, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file: %s\n", inputFileName);
        return 1;
    }

    fseek(fp, 0, SEEK_END);
    long fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char* h_input = (char*)malloc(fileSize * sizeof(char));
    if (!h_input) {
        fprintf(stderr, "Error allocating host memory.\n");
        fclose(fp);
        return 1;
    }

    size_t bytesRead = fread(h_input, sizeof(char), fileSize, fp);
    if (bytesRead != fileSize) {
        fprintf(stderr, "Error reading file content.\n");
        free(h_input);
        fclose(fp);
        return 1;
    }
    fclose(fp);

    char* d_input;
    cudaError_t err = cudaMalloc((void**)&d_input, fileSize * sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for input: %s\n", cudaGetErrorString(err));
        free(h_input);
        return 1;
    }

    err = cudaMemcpy(d_input, h_input, fileSize * sizeof(char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying input to device: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        free(h_input);
        return 1;
    }

    int outputBufferSize = fileSize * 2;
    char* d_output;
    err = cudaMalloc((void**)&d_output, outputBufferSize * sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for output: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        free(h_input);
        return 1;
    }

    int* d_outputOffset;
    err = cudaMalloc((void**)&d_outputOffset, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for output offset: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        free(h_input);
        return 1;
    }
    err = cudaMemset(d_outputOffset, 0, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error initializing output offset: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_outputOffset);
        free(h_input);
        return 1;
    }

    int threadsPerBlock = 256;
    int blocks = (fileSize + threadsPerBlock - 1) / threadsPerBlock;
    tokenizeKernel<<<blocks, threadsPerBlock>>>(d_input, fileSize, d_output, d_outputOffset);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_outputOffset);
        free(h_input);
        return 1;
    }
    cudaDeviceSynchronize();
    int h_outputSize;
    err = cudaMemcpy(&h_outputSize, d_outputOffset, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying output offset from device: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_outputOffset);
        free(h_input);
        return 1;
    }

    char* h_output = (char*)malloc(h_outputSize * sizeof(char));
    if (!h_output) {
        fprintf(stderr, "Error allocating host memory for output.\n");
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_outputOffset);
        free(h_input);
        return 1;
    }
    err = cudaMemcpy(h_output, d_output, h_outputSize * sizeof(char), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying output from device: %s\n", cudaGetErrorString(err));
        free(h_output);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_outputOffset);
        free(h_input);
        return 1;
    }
    FILE* fpOut = fopen("tokens.txt", "wb");
    if (!fpOut) {
        fprintf(stderr, "Error opening output file tokens.txt for writing.\n");
        free(h_output);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_outputOffset);
        free(h_input);
        return 1;
    }
    fwrite(h_output, sizeof(char), h_outputSize, fpOut);
    fclose(fpOut);
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_outputOffset);

    printf("Tokenization complete. Output written to tokens.txt\n");
    return 0;
}

