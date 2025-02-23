# Project Report: CUDA-Based Transformer for Character-Level Text Generation

## Abstract

This project implements a simplified Transformer model using NVIDIA's CUDA for accelerated computations. The model is designed to perform character-level text generation and includes a forward pass along with backpropagation for the output layer. The implementation leverages custom CUDA kernels for core operations—embedding lookup, positional encoding, dummy self-attention, feed-forward network computations, and gradient descent updates. Additionally, support for configurable random initialization (uniform and normal) has been integrated to facilitate experiments with different weight initialization strategies. Although the model is minimal and primarily demonstrative, it lays the groundwork for further exploration into GPU-accelerated deep learning models.

## Introduction

Transformer architectures have revolutionized natural language processing by providing efficient mechanisms for sequence-to-sequence tasks without relying on recurrent structures. However, the high computational cost of training these models often necessitates the use of parallel computing platforms like GPUs. This project aims to:
- Implement a simplified Transformer model in CUDA.
- Demonstrate key components of the model including embedding, self-attention, and feed-forward networks.
- Integrate a simple training loop with backpropagation (focusing on the output layer) using gradient descent.
- Provide flexibility in weight initialization via both uniform and Gaussian distributions.

While the project does not encompass the full complexity of state-of-the-art Transformers, it offers valuable insights into low-level GPU programming and model training dynamics.

## Methodology

### System Architecture

The project is organized into several CUDA/C++ source files:

- **data_loader.cu**: Contains functions to load text data from a file. The file is read into memory for subsequent tokenization.
- **tokenizer.cu**: Implements a simple character-level tokenizer that converts each character into its corresponding ASCII code.
- **kernels.h / kernels.cu**: Define and implement the CUDA kernels for:
  - Fused matrix multiplication with ReLU activation.
  - Standard matrix multiplication.
  - Embedding lookup and positional encoding.
  - Dummy self-attention and final linear layer operations.
  - Backpropagation on the output layer, including softmax-based gradient computation and gradient descent updates.
- **transformer_model.h / transformer_model.cu**: Define the `TransformerModel` structure, model initialization (with support for random initialization modes), the forward pass (both host and device variants), and a training step that integrates forward and backward passes.
- **main.cu**: Serves as the project entry point. It loads the text, tokenizes it, initializes the model, and runs inference or a training step based on command-line arguments.

### Key Components

1. **Data Loading and Tokenization**  
   The project uses simple file I/O to read an input text file. The tokenizer converts text into tokens based on ASCII values, ensuring compatibility with a vocabulary size of 256.

2. **Model Architecture and Forward Pass**  
   The forward pass is divided into several stages:
   - **Embedding Lookup:** Converts token indices into embedding vectors.
   - **Positional Encoding:** Adds positional information using sine and cosine functions.
   - **Dummy Self-Attention:** A placeholder operation that copies the embeddings (simulating self-attention).
   - **Feed-Forward Network (FFN):** Consists of a fused GEMM with ReLU activation followed by a simple matrix multiplication.
   - **Final Linear Layer:** Computes the logits for each token in the vocabulary.

3. **Backpropagation and Gradient Descent**  
   While full backpropagation through the entire Transformer is non-trivial, the project demonstrates backpropagation for the final output layer:
   - A custom CUDA kernel computes the gradient of the loss using softmax and one-hot encoding.
   - Additional kernels compute gradients for the output weights and biases.
   - A generic gradient descent update kernel applies these gradients to update the model parameters.

4. **Random Initialization**  
   The model supports two initialization schemes:
   - **Uniform Initialization:** Weights are sampled uniformly from the range \([-0.1, 0.1]\).
   - **Normal Initialization:** Weights are generated from a Gaussian distribution with a mean of 0 and a standard deviation of 0.1 using the Box-Muller transform.
   
   This flexibility helps in exploring how different initializations affect model convergence and output quality.

## Experimental Results

After compiling and running the project, the model generates logits for the first token and produces text during inference. With random initialization and minimal training (or no training), the model tends to output repetitive text (e.g., a sequence of the same ASCII character). This behavior is expected due to the untrained state of the model and serves as a baseline demonstration.

When the training mode is activated (using the "train" argument), a single training step updates the weights of the output layer. Although this limited training does not yield significant improvements, it validates that the backpropagation and gradient descent kernels function as expected.

## Discussion

The project successfully demonstrates:
- Integration of multiple CUDA kernels for various deep learning operations.
- Implementation of a simplified Transformer model with modular components.
- Support for configurable random initialization schemes.
- Basic training loop integration with forward and backward passes.

However, there are limitations:
- Only the output layer is updated via backpropagation. Full model training would require gradients through all layers.
- The dummy self-attention and simplified FFN do not capture the full complexity of real Transformer networks.
- The training regimen is minimal; significant training on a larger dataset and multiple epochs would be required to generate meaningful text.

## Conclusion

This project presents a CUDA-based implementation of a simplified Transformer model aimed at character-level text generation. Through custom CUDA kernels and modular design, the project highlights key aspects of model initialization, forward pass computation, and backpropagation using gradient descent. While the model in its current state produces repetitive output, it serves as an excellent starting point for further research and development in GPU-accelerated deep learning.

## Future Work

Future enhancements could include:
- Extending backpropagation to all layers (e.g., self-attention, FFN layers).
- Incorporating more sophisticated training strategies (e.g., learning rate scheduling, momentum).
- Scaling the model to handle larger datasets and more complex tasks.
- Implementing additional features like dropout, layer normalization, and multi-head attention to more closely resemble full Transformer architectures.
- Experimenting with different sampling methods during inference (e.g., temperature sampling, nucleus sampling) to increase output diversity.

