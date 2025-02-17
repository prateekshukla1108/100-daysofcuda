#ifndef CNN_MNIST_H
#define CNN_MNIST_H

#include <cudnn.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct {
    unsigned char* pixels;
    int width;
    int height;
} MNISTImage;

typedef struct {
    unsigned char label;
} MNISTLabel;

typedef struct {
    cudnnHandle_t cudnn_handle;
    float* d_weights;
    float* d_biases;
    float* d_weights_grad; // Added gradient pointer
    float* d_biases_grad;  // Added gradient pointer
} CNNModel;

MNISTImage* load_mnist_training_images(const char* file_path, int* num_images);
MNISTLabel* load_mnist_training_labels(const char* file_path, int* num_labels);
MNISTImage* load_mnist_testing_images(const char* file_path, int* num_images);
MNISTLabel* load_mnist_testing_labels(const char* file_path, int* num_labels);
float* create_training_batches(MNISTImage* images, MNISTLabel* labels, int num_samples, int batch_size, int* num_batches);

CNNModel* initialize_cnn_model(cudnnHandle_t handle);
void forward_pass(CNNModel* model, float* input_batch, int batch_size, float* output_batch);
float calculate_cross_entropy_loss(float* predictions, unsigned char* labels, int batch_size);
void perform_backpropagation(CNNModel* model, float* input_batch, float* output_batch, unsigned char* labels, int batch_size);
void update_weights_sgd(CNNModel* model, float learning_rate);
float train_batch(CNNModel* model, float* batch_images, unsigned char* batch_labels, int batch_size, float learning_rate);
void train_epochs(CNNModel* model, float* training_batches, MNISTLabel* training_labels, int num_batches, int batch_size, int epochs, float learning_rate);
float evaluate_model(CNNModel* model, MNISTImage* test_images, MNISTLabel* test_labels, int num_test_samples, int batch_size);

#ifdef __cplusplus
extern "C" {
#endif

__global__ void cnnForwardPassKernel(CNNModel* model, float* input_batch, int batch_size, float* output_batch);
__global__ void cnnBackpropagationKernel(CNNModel* model, float* input_batch, float* output_batch, unsigned char* labels, int batch_size);
__global__ void sgdUpdateWeightsKernel(CNNModel* model, float learning_rate, int weights_size, int biases_size);

#ifdef __cplusplus
}
#endif

#endif

