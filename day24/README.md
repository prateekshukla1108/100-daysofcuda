# CUDA CNN

This project is implementation of a Convolutional Neural Network (CNN) built entirely with CUDA. It’s designed to train on the classic MNIST and showcases how to perform convolution, pooling, fully-connected layers, softmax loss, and backpropagation on the GPU.

---

## ✨ Overview

- **Convolution Layer:** A 5x5 kernel extracts fun features from our input images.
- **ReLU Activation:** Adds a spark of non-linearity to our network.
- **Max Pooling:** Downsamples the feature maps.
- **Fully Connected Layer:** Brings together all the learned features to make predictions.
- **Softmax Loss:** Computes probabilities and loss, guiding our network with a smile.
- **Backpropagation:** Carefully computes gradients and updates weights using either SGD or the ever-adaptive Adam optimizer.

Each CUDA kernel is written with love to execute efficiently on your GPU, making training fast and fun!

---

## 🌟 Features

- **CUDA Acceleration:** All the heavy lifting happens on the GPU for super-fast performance.
- **Multiple Optimizers:** Choose between classic SGD and the advanced Adam optimizer.
- **Batch Training:** Processes mini-batches of images, perfect for efficient learning.

---

## 🚀 Getting Started

### Prerequisites

- **CUDA Toolkit:** Ensure you have the CUDA toolkit installed.
- **C/C++ Compiler:** A compiler that supports CUDA (like `nvcc`).
- **MNIST Data:** You’ll need the MNIST dataset in binary format which you can download from given script


### Compilation

Compile the code using `nvcc` (the NVIDIA CUDA Compiler):

```bash
nvcc -o cuda_cnn cuda_cnn.cu -lm
```

### Running the Application

After compiling, run the executable:

```bash
./cuda_cnn
```

Watch as the network trains over 10 epochs, printing the average loss for each epoch and the final training loss!

---

## 📝 Code Structure

Here’s a quick tour of our happy code:

- **Main Function:**  
  - Loads the MNIST data.
  - Initializes weights and biases with a touch of randomness.
  - Manages the training loop and memory allocations.
  
- **Forward Pass Kernels:**
  - `conv_forward`: Performs convolution over input images.
  - `pool_forward`: Implements max pooling with argmax tracking.
  - `fc_forward`: Processes the fully connected layer.
  - `softmax_loss`: Computes the softmax probabilities, loss, and gradients.

- **Backward Pass Kernels:**
  - `fc_backward_w_new` & `fc_backward_b`: Compute gradients for the fully connected layer.
  - `pool_backward`: Propagates gradients through the pooling layer.
  - `relu_backward_conv`: Handles the ReLU gradient.
  - `conv_backward_w` & `conv_backward_b`: Compute gradients for convolution weights and biases.
  - `conv_grad_input`: Propagates gradients back to the input.

- **Parameter Update Kernel:**
  - `update_param`: Updates parameters using either SGD or Adam

---

## 🎯 Optimizer Options

In the code, you can switch between:
- **SGD (Stochastic Gradient Descent)**
- **Adam:** The default, adaptive optimizer that brings extra sparkle to the training process.

Simply change the `opt` variable in the source code if you want to try a different optimizer.


---

## 🌈 Final Thoughts

Thank you for exploring this CUDA CNN code. I hope it inspires you to experiment with GPU programming and deep learning. If you have any questions, suggestions, or just want to spread some positive vibes, don’t hesitate to reach out.

Happy coding and keep working! 
