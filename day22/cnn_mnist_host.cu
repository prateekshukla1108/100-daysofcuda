#include "cnn_mnist.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cudnn.h>

#define INPUT_WIDTH 28
#define INPUT_HEIGHT 28
#define FILTER_SIZE 5
#define NUM_FILTERS 16
#define CONV_OUTPUT_WIDTH (INPUT_WIDTH - FILTER_SIZE + 1)
#define CONV_OUTPUT_HEIGHT (INPUT_HEIGHT - FILTER_SIZE + 1)
#define POOL_SIZE 2
#define POOL_OUTPUT_WIDTH (CONV_OUTPUT_WIDTH / POOL_SIZE)
#define POOL_OUTPUT_HEIGHT (CONV_OUTPUT_HEIGHT / POOL_SIZE)
#define FC_INPUT_SIZE (NUM_FILTERS * POOL_OUTPUT_WIDTH * POOL_OUTPUT_HEIGHT)
#define FC_OUTPUT_SIZE 10


static int readInt(FILE* fp) {
    unsigned char bytes[4];
    if (fread(bytes, 1, 4, fp) != 4) exit(1);
    return ((int)bytes[0] << 24) | ((int)bytes[1] << 16) | ((int)bytes[2] << 8) | ((int)bytes[3]);
}

MNISTImage* load_mnist_training_images(const char* images_path, int& num_images) {
    FILE* fp = fopen(images_path, "rb");
    if (!fp) { std::cerr << "Error opening training image file\n"; exit(1); }
    int magic = readInt(fp);
    num_images = readInt(fp);
    int rows = readInt(fp);
    int cols = readInt(fp);
    MNISTImage* images = new MNISTImage[num_images];
    for (int i = 0; i < num_images; i++) {
        images[i].width = cols;
        images[i].height = rows;
        images[i].pixels = new unsigned char[rows * cols];
        if (fread(images[i].pixels, 1, rows * cols, fp) != (size_t)(rows * cols)) { std::cerr << "Error reading training image data\n"; exit(1); }
    }
    fclose(fp);
    return images;
}

MNISTLabel* load_mnist_training_labels(const char* labels_path, int& num_labels) {
    FILE* fp = fopen(labels_path, "rb");
    if (!fp) { std::cerr << "Error opening training label file\n"; exit(1); }
    int magic = readInt(fp);
    num_labels = readInt(fp);
    MNISTLabel* labels = new MNISTLabel[num_labels];
    for (int i = 0; i < num_labels; i++) {
        if (fread(&labels[i].label, 1, 1, fp) != 1) { std::cerr << "Error reading training label data\n"; exit(1); }
    }
    fclose(fp);
    return labels;
}

MNISTImage* load_mnist_testing_images(const char* images_path, int& num_images) {
    FILE* fp = fopen(images_path, "rb");
    if (!fp) { std::cerr << "Error opening testing image file\n"; exit(1); }
    int magic = readInt(fp);
    num_images = readInt(fp);
    int rows = readInt(fp);
    int cols = readInt(fp);
    MNISTImage* images = new MNISTImage[num_images];
    for (int i = 0; i < num_images; i++) {
        images[i].width = cols;
        images[i].height = rows;
        images[i].pixels = new unsigned char[rows * cols];
        if (fread(images[i].pixels, 1, rows * cols, fp) != (size_t)(rows * cols)) { std::cerr << "Error reading testing image data\n"; exit(1); }
    }
    fclose(fp);
    return images;
}

MNISTLabel* load_mnist_testing_labels(const char* labels_path, int& num_labels) {
    FILE* fp = fopen(labels_path, "rb");
    if (!fp) { std::cerr << "Error opening testing label file\n"; exit(1); }
    int magic = readInt(fp);
    num_labels = readInt(fp);
    MNISTLabel* labels = new MNISTLabel[num_labels];
    for (int i = 0; i < num_labels; i++) {
        if (fread(&labels[i].label, 1, 1, fp) != 1) { std::cerr << "Error reading testing label data\n"; exit(1); }
    }
    fclose(fp);
    return labels;
}

float* create_training_batches(MNISTImage* images, MNISTLabel* labels, int num_samples, int batch_size, int& num_batches) {
    num_batches = num_samples / batch_size;
    int image_size = images[0].width * images[0].height;
    float* h_batches = new float[num_batches * batch_size * image_size];
    for (int i = 0; i < num_batches * batch_size; i++) {
        for (int j = 0; j < image_size; j++) {
            h_batches[i * image_size + j] = images[i].pixels[j] / 255.0f;
        }
    }
    float* d_batches;
    cudaMalloc((void**)&d_batches, num_batches * batch_size * image_size * sizeof(float));
    cudaMemcpy(d_batches, h_batches, num_batches * batch_size * image_size * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_batches;
    return d_batches;
}

CNNModel* initialize_cnn_model(cudnnHandle_t handle) {
    CNNModel* model = new CNNModel;
    model->cudnn_handle = handle;
    size_t conv_weights_size = NUM_FILTERS * FILTER_SIZE * FILTER_SIZE * sizeof(float);
    size_t fc_weights_size = FC_OUTPUT_SIZE * FC_INPUT_SIZE * sizeof(float);
    size_t total_weights_size = conv_weights_size + fc_weights_size;
    cudaMalloc((void**)&model->d_weights, total_weights_size);
    cudaMalloc((void**)&model->d_weights_grad, total_weights_size);
    size_t conv_biases_size = NUM_FILTERS * sizeof(float);
    size_t fc_biases_size = FC_OUTPUT_SIZE * sizeof(float);
    size_t total_biases_size = conv_biases_size + fc_biases_size;
    cudaMalloc((void**)&model->d_biases, total_biases_size);
    cudaMalloc((void**)&model->d_biases_grad, total_biases_size);
    return model;
}

void forward_pass(CNNModel* model, float* input_batch, int batch_size, float* output_batch) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    cnnForwardPassKernel<<<blocks, threads>>>(model, input_batch, batch_size, output_batch);
    cudaDeviceSynchronize();
}

float calculate_cross_entropy_loss(float* predictions, unsigned char* labels, int batch_size) {
    float loss = 0.0f;
    float* h_predictions = new float[batch_size * FC_OUTPUT_SIZE];
    cudaMemcpy(h_predictions, predictions, batch_size * FC_OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < batch_size; i++) {
        int target = labels[i];
        float p = h_predictions[i * FC_OUTPUT_SIZE + target];
        if (p < 1e-7f) p = 1e-7f;
        loss += -logf(p);
    }
    delete[] h_predictions;
    return loss / batch_size;
}

void perform_backpropagation(CNNModel* model, float* input_batch, float* output_batch, unsigned char* labels, int batch_size) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    cnnBackpropagationKernel<<<blocks, threads>>>(model, input_batch, output_batch, labels, batch_size);
    cudaDeviceSynchronize();
}

void update_weights_sgd(CNNModel* model, float learning_rate) {
    int total_weights = NUM_FILTERS * FILTER_SIZE * FILTER_SIZE + FC_OUTPUT_SIZE * FC_INPUT_SIZE;
    int total_biases = NUM_FILTERS + FC_OUTPUT_SIZE;
    int max_count = (total_weights > total_biases) ? total_weights : total_biases;
    int threads = 256;
    int blocks = (max_count + threads - 1) / threads;
    sgdUpdateWeightsKernel<<<blocks, threads>>>(model, learning_rate, total_weights, total_biases);
    cudaDeviceSynchronize();
}

float train_batch(CNNModel* model, float* batch_images, unsigned char* batch_labels, int batch_size, float learning_rate) {
    float* d_output;
    cudaMalloc((void**)&d_output, batch_size * FC_OUTPUT_SIZE * sizeof(float));
    forward_pass(model, batch_images, batch_size, d_output);
    float loss = calculate_cross_entropy_loss(d_output, batch_labels, batch_size);
    perform_backpropagation(model, batch_images, d_output, batch_labels, batch_size);
    update_weights_sgd(model, learning_rate);
    cudaFree(d_output);
    return loss;
}

void train_epochs(CNNModel* model, float* training_batches, MNISTLabel* training_labels, int num_batches, int batch_size, int epochs, float learning_rate) {
    int image_size = INPUT_WIDTH * INPUT_HEIGHT;
    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;
        for (int b = 0; b < num_batches; b++) {
            float* batch_ptr = training_batches + b * batch_size * image_size;
            unsigned char* batch_labels = new unsigned char[batch_size];
            for (int i = 0; i < batch_size; i++) {
                batch_labels[i] = training_labels[b * batch_size + i].label;
            }
            float loss = train_batch(model, batch_ptr, batch_labels, batch_size, learning_rate);
            epoch_loss += loss;
            delete[] batch_labels;
        }
        std::cout << "Epoch " << epoch + 1 << " Loss: " << epoch_loss / num_batches << std::endl;
    }
}

float evaluate_model(CNNModel* model, MNISTImage* test_images, MNISTLabel* test_labels, int num_test_samples, int batch_size) {
    int image_size = test_images[0].width * test_images[0].height;
    int num_batches = num_test_samples / batch_size;
    int correct = 0;
    for (int b = 0; b < num_batches; b++) {
        float* d_input;
        cudaMalloc((void**)&d_input, batch_size * image_size * sizeof(float));
        float* h_input = new float[batch_size * image_size];
        for (int i = 0; i < batch_size; i++) {
            int idx = b * batch_size + i;
            for (int j = 0; j < image_size; j++) {
                h_input[i * image_size + j] = test_images[idx].pixels[j] / 255.0f;
            }
        }
        cudaMemcpy(d_input, h_input, batch_size * image_size * sizeof(float), cudaMemcpyHostToDevice);
        float* d_output;
        cudaMalloc((void**)&d_output, batch_size * FC_OUTPUT_SIZE * sizeof(float));
        forward_pass(model, d_input, batch_size, d_output);
        float* h_output = new float[batch_size * FC_OUTPUT_SIZE];
        cudaMemcpy(h_output, d_output, batch_size * FC_OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < batch_size; i++) {
            int max_idx = 0;
            float max_val = h_output[i * FC_OUTPUT_SIZE];
            for (int j = 1; j < FC_OUTPUT_SIZE; j++) {
                float val = h_output[i * FC_OUTPUT_SIZE + j];
                if (val > max_val) { max_val = val; max_idx = j; }
            }
            if (max_idx == test_labels[b * batch_size + i].label) correct++;
        }
        delete[] h_input;
        delete[] h_output;
        cudaFree(d_input);
        cudaFree(d_output);
    }
    return (float)correct / (num_batches * batch_size);
}

int main(int argc, char** argv) {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    int num_train_images, num_train_labels, num_test_images, num_test_labels;
    MNISTImage* train_images = load_mnist_training_images("train-images.idx3-ubyte", num_train_images);
    MNISTLabel* train_labels = load_mnist_training_labels("train-labels.idx1-ubyte", num_train_labels);
    MNISTImage* test_images = load_mnist_testing_images("t10k-images.idx3-ubyte", num_test_images);
    MNISTLabel* test_labels = load_mnist_testing_labels("t10k-labels.idx1-ubyte", num_test_labels);
    int batch_size = 64;
    int num_batches;
    float* training_batches = create_training_batches(train_images, train_labels, num_train_images, batch_size, num_batches);
    CNNModel* model = initialize_cnn_model(cudnn);
    int epochs = 10;
    float learning_rate = 0.01f;
    train_epochs(model, training_batches, train_labels, num_batches, batch_size, epochs, learning_rate);
    float accuracy = evaluate_model(model, test_images, test_labels, num_test_images, batch_size);
    std::cout << "Test Accuracy: " << accuracy * 100 << "%" << std::endl;
    cudaFree(training_batches);
    for (int i = 0; i < num_train_images; i++) delete[] train_images[i].pixels;
    delete[] train_images;
    delete[] train_labels;
    for (int i = 0; i < num_test_images; i++) delete[] test_images[i].pixels;
    delete[] test_images;
    delete[] test_labels;
    cudnnDestroy(cudnn);
    return 0;
}

