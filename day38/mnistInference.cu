#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <stdexcept>

// Reverse bytes for big-endian data in MNIST files
int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// Load MNIST images
void read_mnist_images(const std::string& path, float* images, int num_images) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open image file: " + path);
    }
    int magic, n, rows, cols;
    file.read((char*)&magic, sizeof(magic));
    magic = reverse_int(magic);
    file.read((char*)&n, sizeof(n));
    n = reverse_int(n);
    file.read((char*)&rows, sizeof(rows));
    rows = reverse_int(rows);
    file.read((char*)&cols, sizeof(cols));
    cols = reverse_int(cols);
    if (n != num_images || rows != 28 || cols != 28) {
        throw std::runtime_error("Invalid MNIST image file dimensions");
    }
    for (int i = 0; i < num_images * 28 * 28; i++) {
        unsigned char pixel;
        file.read((char*)&pixel, 1);
        images[i] = pixel / 255.0f; // Normalize to [0,1]
    }
    file.close();
}

// Load MNIST labels
void read_mnist_labels(const std::string& path, int* labels, int num_labels) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open label file: " + path);
    }
    int magic, n;
    file.read((char*)&magic, sizeof(magic));
    magic = reverse_int(magic);
    file.read((char*)&n, sizeof(n));
    n = reverse_int(n);
    if (n != num_labels) {
        throw std::runtime_error("Invalid MNIST label file dimensions");
    }
    for (int i = 0; i < num_labels; i++) {
        unsigned char label;
        file.read((char*)&label, 1);
        labels[i] = static_cast<int>(label);
    }
    file.close();
}

// CUDA Kernels

// Matrix multiplication: C = A @ B
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K, bool transA, bool transB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            int a_idx = transA ? k * M + row : row * K + k;
            int b_idx = transB ? col * K + k : k * N + col;
            sum += A[a_idx] * B[b_idx];
        }
        C[row * N + col] = sum;
    }
}

// Add bias to output
__global__ void add_bias_kernel(float* output, const float* bias, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        output[row * N + col] += bias[col];
    }
}

// ReLU activation
__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(data[idx], 0.0f);
    }
}

// Softmax (simple implementation, one block per row)
__global__ void softmax_kernel(const float* input, float* output, int M, int N) {
    int row = blockIdx.x;
    if (row < M) {
        float max_val = -INFINITY;
        for (int i = 0; i < N; i++) {
            max_val = fmaxf(max_val, input[row * N + i]);
        }
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            output[row * N + i] = expf(input[row * N + i] - max_val);
            sum += output[row * N + i];
        }
        for (int i = 0; i < N; i++) {
            output[row * N + i] /= sum;
        }
    }
}

// Compute gradient of softmax cross-entropy loss
__global__ void softmax_cross_entropy_grad_kernel(const float* p, const int* labels, float* grad, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        int label = labels[row];
        grad[row * N + col] = p[row * N + col] - (col == label ? 1.0f : 0.0f);
    }
}

// ReLU backward
__global__ void relu_grad_kernel(const float* hidden, const float* grad_hidden, float* grad_hidden_pre_relu, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_hidden_pre_relu[idx] = (hidden[idx] > 0) ? grad_hidden[idx] : 0.0f;
    }
}

// Sum rows for bias gradients
__global__ void sum_rows_kernel(const float* data, float* sum, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        float s = 0.0f;
        for (int row = 0; row < M; row++) {
            s += data[row * N + col];
        }
        sum[col] = s;
    }
}

// Update weights with SGD
__global__ void update_weights_kernel(float* W, const float* dW, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        W[idx] -= lr * dW[idx];
    }
}

int main() {
    // Constants
    const int train_size = 60000;
    const int test_size = 10000;
    const int input_size = 28 * 28; // 784
    const int hidden_size = 128;
    const int output_size = 10;
    const int batch_size = 64;
    const int epochs = 5;
    const float lr = 0.01f;

    // Load MNIST data on CPU
    float* train_images = new float[train_size * input_size];
    int* train_labels = new int[train_size];
    float* test_images = new float[test_size * input_size];
    int* test_labels = new int[test_size];

    read_mnist_images("train-images-idx3-ubyte", train_images, train_size);
    read_mnist_labels("train-labels-idx1-ubyte", train_labels, train_size);
    read_mnist_images("t10k-images-idx3-ubyte", test_images, test_size);
    read_mnist_labels("t10k-labels-idx1-ubyte", test_labels, test_size);

    // Allocate GPU memory for data
    float *d_train_images, *d_test_images;
    int *d_train_labels, *d_test_labels;
    cudaMalloc(&d_train_images, train_size * input_size * sizeof(float));
    cudaMalloc(&d_train_labels, train_size * sizeof(int));
    cudaMalloc(&d_test_images, test_size * input_size * sizeof(float));
    cudaMalloc(&d_test_labels, test_size * sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_train_images, train_images, train_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_labels, train_labels, train_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_images, test_images, test_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_labels, test_labels, test_size * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize weights on CPU
    float* W1 = new float[input_size * hidden_size];
    float* b1 = new float[hidden_size];
    float* W2 = new float[hidden_size * output_size];
    float* b2 = new float[output_size];
    srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < input_size * hidden_size; i++) W1[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.2f;
    for (int i = 0; i < hidden_size * output_size; i++) W2[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.2f;
    memset(b1, 0, hidden_size * sizeof(float));
    memset(b2, 0, output_size * sizeof(float));

    // Allocate and copy weights to GPU
    float *d_W1, *d_b1, *d_W2, *d_b2;
    cudaMalloc(&d_W1, input_size * hidden_size * sizeof(float));
    cudaMalloc(&d_b1, hidden_size * sizeof(float));
    cudaMalloc(&d_W2, hidden_size * output_size * sizeof(float));
    cudaMalloc(&d_b2, output_size * sizeof(float));
    cudaMemcpy(d_W1, W1, input_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1, hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2, hidden_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2, output_size * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory for intermediates and gradients
    float *d_hidden_pre_relu, *d_hidden, *d_output, *d_p;
    float *d_grad_output, *d_grad_hidden, *d_grad_hidden_pre_relu;
    float *d_dW1, *d_db1, *d_dW2, *d_db2;
    cudaMalloc(&d_hidden_pre_relu, batch_size * hidden_size * sizeof(float));
    cudaMalloc(&d_hidden, batch_size * hidden_size * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_size * sizeof(float));
    cudaMalloc(&d_p, batch_size * output_size * sizeof(float));
    cudaMalloc(&d_grad_output, batch_size * output_size * sizeof(float));
    cudaMalloc(&d_grad_hidden, batch_size * hidden_size * sizeof(float));
    cudaMalloc(&d_grad_hidden_pre_relu, batch_size * hidden_size * sizeof(float));
    cudaMalloc(&d_dW1, input_size * hidden_size * sizeof(float));
    cudaMalloc(&d_db1, hidden_size * sizeof(float));
    cudaMalloc(&d_dW2, hidden_size * output_size * sizeof(float));
    cudaMalloc(&d_db2, output_size * sizeof(float));

    // Training
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 blockDim(16, 16);
    dim3 gridDimFC1((hidden_size + blockDim.x - 1) / blockDim.x, (batch_size + blockDim.y - 1) / blockDim.y);
    dim3 gridDimFC2((output_size + blockDim.x - 1) / blockDim.x, (batch_size + blockDim.y - 1) / blockDim.y);
    int num_batches = train_size / batch_size;

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int batch = 0; batch < num_batches; batch++) {
            float* d_input_batch = d_train_images + batch * batch_size * input_size;
            int* d_labels_batch = d_train_labels + batch * batch_size;

            // Forward pass
            matmul_kernel<<<gridDimFC1, blockDim>>>(d_input_batch, d_W1, d_hidden_pre_relu, batch_size, hidden_size, input_size, false, false);
            add_bias_kernel<<<gridDimFC1, blockDim>>>(d_hidden_pre_relu, d_b1, batch_size, hidden_size);
            relu_kernel<<<(batch_size * hidden_size + 255) / 256, 256>>>(d_hidden_pre_relu, batch_size * hidden_size);
            cudaMemcpy(d_hidden, d_hidden_pre_relu, batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToDevice);
            matmul_kernel<<<gridDimFC2, blockDim>>>(d_hidden, d_W2, d_output, batch_size, output_size, hidden_size, false, false);
            add_bias_kernel<<<gridDimFC2, blockDim>>>(d_output, d_b2, batch_size, output_size);
            softmax_kernel<<<batch_size, 1>>>(d_output, d_p, batch_size, output_size);

            // Backward pass
            softmax_cross_entropy_grad_kernel<<<gridDimFC2, blockDim>>>(d_p, d_labels_batch, d_grad_output, batch_size, output_size);
            matmul_kernel<<<gridDimFC2, blockDim>>>(d_hidden, d_grad_output, d_dW2, hidden_size, output_size, batch_size, true, false);
            sum_rows_kernel<<<(output_size + 255) / 256, 256>>>(d_grad_output, d_db2, batch_size, output_size);
            matmul_kernel<<<gridDimFC1, blockDim>>>(d_grad_output, d_W2, d_grad_hidden, batch_size, hidden_size, output_size, false, true);
            relu_grad_kernel<<<(batch_size * hidden_size + 255) / 256, 256>>>(d_hidden, d_grad_hidden, d_grad_hidden_pre_relu, batch_size * hidden_size);
            matmul_kernel<<<gridDimFC1, blockDim>>>(d_input_batch, d_grad_hidden_pre_relu, d_dW1, input_size, hidden_size, batch_size, true, false);
            sum_rows_kernel<<<(hidden_size + 255) / 256, 256>>>(d_grad_hidden_pre_relu, d_db1, batch_size, hidden_size);

            // Update weights
            update_weights_kernel<<<(input_size * hidden_size + 255) / 256, 256>>>(d_W1, d_dW1, lr, input_size * hidden_size);
            update_weights_kernel<<<(hidden_size + 255) / 256, 256>>>(d_b1, d_db1, lr, hidden_size);
            update_weights_kernel<<<(hidden_size * output_size + 255) / 256, 256>>>(d_W2, d_dW2, lr, hidden_size * output_size);
            update_weights_kernel<<<(output_size + 255) / 256, 256>>>(d_b2, d_db2, lr, output_size);
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float train_time_ms;
    cudaEventElapsedTime(&train_time_ms, start, stop);
    float train_time = train_time_ms / 1000.0f;

    // Evaluation
    int correct = 0;
    num_batches = test_size / batch_size; // Process full batches only for simplicity
    for (int batch = 0; batch < num_batches; batch++) {
        float* d_input_batch = d_test_images + batch * batch_size * input_size;
        int* d_labels_batch = d_test_labels + batch * batch_size;

        matmul_kernel<<<gridDimFC1, blockDim>>>(d_input_batch, d_W1, d_hidden_pre_relu, batch_size, hidden_size, input_size, false, false);
        add_bias_kernel<<<gridDimFC1, blockDim>>>(d_hidden_pre_relu, d_b1, batch_size, hidden_size);
        relu_kernel<<<(batch_size * hidden_size + 255) / 256, 256>>>(d_hidden_pre_relu, batch_size * hidden_size);
        cudaMemcpy(d_hidden, d_hidden_pre_relu, batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToDevice);
        matmul_kernel<<<gridDimFC2, blockDim>>>(d_hidden, d_W2, d_output, batch_size, output_size, hidden_size, false, false);
        add_bias_kernel<<<gridDimFC2, blockDim>>>(d_output, d_b2, batch_size, output_size);

        float* h_output = new float[batch_size * output_size];
        int* h_labels = new int[batch_size];
        cudaMemcpy(h_output, d_output, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_labels, d_labels_batch, batch_size * sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < batch_size; i++) {
            int pred = 0;
            float max_val = h_output[i * output_size];
            for (int j = 1; j < output_size; j++) {
                if (h_output[i * output_size + j] > max_val) {
                    max_val = h_output[i * output_size + j];
                    pred = j;
                }
            }
            if (pred == h_labels[i]) correct++;
        }
        delete[] h_output;
        delete[] h_labels;
    }
    float accuracy = static_cast<float>(correct) / (num_batches * batch_size);

    // Inference time
    cudaEventRecord(start);
    for (int batch = 0; batch < num_batches; batch++) {
        float* d_input_batch = d_test_images + batch * batch_size * input_size;
        matmul_kernel<<<gridDimFC1, blockDim>>>(d_input_batch, d_W1, d_hidden_pre_relu, batch_size, hidden_size, input_size, false, false);
        add_bias_kernel<<<gridDimFC1, blockDim>>>(d_hidden_pre_relu, d_b1, batch_size, hidden_size);
        relu_kernel<<<(batch_size * hidden_size + 255) / 256, 256>>>(d_hidden_pre_relu, batch_size * hidden_size);
        cudaMemcpy(d_hidden, d_hidden_pre_relu, batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToDevice);
        matmul_kernel<<<gridDimFC2, blockDim>>>(d_hidden, d_W2, d_output, batch_size, output_size, hidden_size, false, false);
        add_bias_kernel<<<gridDimFC2, blockDim>>>(d_output, d_b2, batch_size, output_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float inference_time_ms;
    cudaEventElapsedTime(&inference_time_ms, start, stop);
    float inference_time = inference_time_ms / 1000.0f;

    // Print results
    std::cout << "Chad NN - Accuracy: " << accuracy << ", Training Time: " << train_time << "s" << std::endl;
    std::cout << "Inference Time: " << inference_time << "s" << std::endl;

    // Cleanup
    delete[] train_images; delete[] train_labels;
    delete[] test_images; delete[] test_labels;
    delete[] W1; delete[] b1; delete[] W2; delete[] b2;
    cudaFree(d_train_images); cudaFree(d_train_labels);
    cudaFree(d_test_images); cudaFree(d_test_labels);
    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_hidden_pre_relu); cudaFree(d_hidden); cudaFree(d_output); cudaFree(d_p);
    cudaFree(d_grad_output); cudaFree(d_grad_hidden); cudaFree(d_grad_hidden_pre_relu);
    cudaFree(d_dW1); cudaFree(d_db1); cudaFree(d_dW2); cudaFree(d_db2);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
