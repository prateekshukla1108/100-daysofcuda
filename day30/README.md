## **Fused CUDA Kernels for Neural Network Training and Inference on Text Data**

## Abstract
This project demonstrates a CUDA-based implementation of a simple neural network for processing text data. The network is designed with a single hidden layer and employs fused CUDA kernels to perform the forward pass (including softmax and loss computation), backward pass (gradient computation), and parameter update in a single launch each. By fusing these operations, the code minimizes kernel launch overhead and improves data locality using dynamic shared memory. This report details the design decisions, implementation, testing, and potential avenues for further optimization.

## Introduction
Modern deep learning models often require high-performance computation, especially when handling large datasets and complex architectures. CUDA provides the tools necessary to harness the parallel processing power of GPUs. This project implements a compact neural network designed for text-based tasks such as next-token prediction. It processes an input text file, builds a vocabulary, creates embedding representations, and trains the network using three fused CUDA kernels. The design aims to illustrate both the advantages of kernel fusion in reducing runtime overhead and the careful management of shared memory for efficient computation.

## Background
The neural network in this project consists of:
- An **embedding layer** that maps tokens to dense vectors.
- A **hidden layer** computed as a matrix-vector product followed by a ReLU activation.
- An **output layer** that produces logits for each token in the vocabulary.
- A **softmax function** to convert logits into probabilities.
- A **cross-entropy loss** function to measure prediction error.

Traditional implementations separate the forward, backward, and parameter update operations into distinct kernels. However, this approach can incur significant overhead due to repeated kernel launches and memory transfers. Fusing these operations into fewer kernels not only reduces overhead but also improves performance by retaining intermediate data in shared memory.

## Methodology
The project uses three fused kernels:

1. **Fused Forward–Loss Kernel:**  
   - Computes the hidden layer output using a matrix–vector multiplication (with ReLU activation).
   - Computes the output logits by multiplying the hidden layer output with the output weight matrix.
   - Applies a numerically stable softmax function and computes the cross-entropy loss for the target token.
   - Utilizes dynamic shared memory to store both intermediate hidden activations and temporary values for reduction operations.

2. **Fused Backward Kernel:**  
   - Computes the gradient of the loss with respect to the logits.
   - Backpropagates these gradients through the output layer to compute gradients for the output parameters.
   - Propagates errors back through the hidden layer while applying the ReLU derivative.
   - Computes gradients for the input layer, which are later used to update the embedding.
   - Manages dynamic shared memory allocation by partitioning the available space for different gradient arrays.

3. **Fused Update Kernel:**  
   - Applies gradient descent updates to all trainable parameters (weights and biases) in both the hidden and output layers.
   - Uses a simple learning rate-based update rule.

Dynamic shared memory is carefully allocated in both the forward and backward kernels. The forward kernel reserves memory for the hidden layer outputs and a temporary reduction buffer, while the backward kernel allocates memory for both the logits gradients and the hidden layer gradients. This design avoids exceeding the allocated shared memory limits and prevents runtime errors.

## Implementation
The project is implemented in a single C++ file with CUDA extensions. Key aspects include:

- **Text Processing and Vocabulary Building:**  
  The code reads input text, tokenizes it, and constructs a vocabulary mapping tokens to unique identifiers. If the vocabulary size exceeds a defined limit (`MAX_VOCAB`), excess tokens are mapped to a default index.

- **Parameter Initialization:**  
  Model parameters (embedding matrix, weight matrices, and bias vectors) are initialized using Xavier/Glorot initialization. Random number generators are used to fill these parameters with appropriate values.

- **Memory Allocation:**  
  Both host and device memories are allocated for parameters, gradients, inputs, and outputs. Dynamic shared memory is allocated per kernel launch based on the needs of each fused kernel.

- **Training Loop:**  
  The training loop processes one sample at a time. For each token pair (input token and target token), the embedding is extracted and copied to device memory. The fused forward-loss kernel computes the network’s output and loss, the fused backward kernel computes gradients, and the fused update kernel adjusts the parameters. The gradient with respect to the input is then used to update the embedding vector.

- **Inference:**  
  After training, the network performs inference using a context derived from the last few tokens of the input. The code averages the embeddings of these tokens, performs a forward pass, and selects the token with the highest probability as the prediction.

## Testing and Results
The complete code compiles using `nvcc` and executes on CUDA-capable hardware. During training, the program prints the loss per sample and the average loss per epoch. After training, it demonstrates inference by outputting the input context and the predicted next token along with its probability. The fusion of kernels has effectively minimized kernel launch overhead, although further optimization (such as removing atomic operations where possible) is a potential future improvement.


