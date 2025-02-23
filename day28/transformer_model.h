#ifndef TRANSFORMER_MODEL_H
#define TRANSFORMER_MODEL_H

#include <cuda_runtime.h>

// Model hyperparameters (for demonstration)
#define VOCAB_SIZE 256   // Character-level
#define EMBED_DIM 32
#define SEQ_LENGTH 32
#define FFN_HIDDEN 64

// Random initialization modes.
#define INIT_UNIFORM 0
#define INIT_NORMAL  1

// Structure holding pointers to model parameters in device memory.
typedef struct {
    // Embedding table: [VOCAB_SIZE x EMBED_DIM]
    float* d_embedding;
    // Dummy multi-head attention weight matrices (for Q, K, V, and output)
    float* d_W_q; // [EMBED_DIM x EMBED_DIM]
    float* d_W_k; // [EMBED_DIM x EMBED_DIM]
    float* d_W_v; // [EMBED_DIM x EMBED_DIM]
    float* d_W_o; // [EMBED_DIM x EMBED_DIM]

    // Feed-forward network parameters:
    float* d_W1; // [EMBED_DIM x FFN_HIDDEN]
    float* d_b1; // [FFN_HIDDEN]
    float* d_W2; // [FFN_HIDDEN x EMBED_DIM]
    float* d_b2; // [EMBED_DIM]

    // Output layer parameters:
    float* d_W_out; // [EMBED_DIM x VOCAB_SIZE]
    float* d_b_out; // [VOCAB_SIZE]
} TransformerModel;

// Function prototypes.
void initModel(TransformerModel* model, int initMethod);
void freeModel(TransformerModel* model);
void forwardPass(TransformerModel* model, const int* h_tokens, float* h_logits);
// Forward pass that returns device pointers for further training.
void forwardPassDevice(TransformerModel* model, const int* h_tokens, float** d_logits, float** d_encoder_output);
// Training step: forward pass (on device), backpropagation for the output layer,
// and parameter updates using gradient descent.
void trainStep(TransformerModel* model,
               const int* h_input_seq,
               const int* h_target_seq,
               float learning_rate);

#endif // TRANSFORMER_MODEL_H

