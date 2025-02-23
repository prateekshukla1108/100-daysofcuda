#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "transformer_model.h"

// Forward declarations for data loading and tokenization.
char* loadText(const char* filename, size_t* length);
void tokenize(const char* text, int** tokens, size_t* tokenCount);

// For simplicity, we assume SEQ_LENGTH tokens per sample.
#define SEQ_LENGTH 32

// Inference: Given an initial sequence, run forward pass and pick next token by argmax.
int inferNextToken(TransformerModel* model, const int* input_seq) {
    float* h_logits = (float*)malloc(SEQ_LENGTH * VOCAB_SIZE * sizeof(float));
    if (!h_logits) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    forwardPass(model, input_seq, h_logits);
    int bestToken = 0;
    float bestLogit = h_logits[(SEQ_LENGTH - 1) * VOCAB_SIZE];
    for (int i = 1; i < VOCAB_SIZE; i++) {
        float logit = h_logits[(SEQ_LENGTH - 1) * VOCAB_SIZE + i];
        if (logit > bestLogit) {
            bestLogit = logit;
            bestToken = i;
        }
    }
    free(h_logits);
    return bestToken;
}

int main(int argc, char** argv) {
    // Optionally, set a random seed for reproducibility.
    srand(1234);

    // Load and tokenize text.
    size_t textLength = 0;
    char* text = loadText("input.txt", &textLength);
    int* tokens;
    size_t tokenCount;
    tokenize(text, &tokens, &tokenCount);
    free(text);

    if(tokenCount < SEQ_LENGTH) {
        fprintf(stderr, "Not enough tokens in input.txt\n");
        exit(EXIT_FAILURE);
    }
    int input_seq[SEQ_LENGTH];
    for (int i = 0; i < SEQ_LENGTH; i++) {
        input_seq[i] = tokens[i];
    }
    free(tokens);

    // Initialize model.
    TransformerModel model;
    // Choose the initialization method here: INIT_UNIFORM or INIT_NORMAL.
    initModel(&model, INIT_NORMAL);

    // If run with "train" argument, perform one training step.
    if(argc > 1 && strcmp(argv[1], "train") == 0) {
        // For demonstration, use the same sequence as input and target (shifted by one).
        int target_seq[SEQ_LENGTH];
        for (int i = 0; i < SEQ_LENGTH - 1; i++) {
            target_seq[i] = input_seq[i+1];
        }
        target_seq[SEQ_LENGTH - 1] = input_seq[0];
        float learning_rate = 0.01f;
        trainStep(&model, input_seq, target_seq, learning_rate);
        printf("Training step completed.\n");
    }

    // Run a forward pass and print logits for the first token.
    float* h_logits = (float*)malloc(SEQ_LENGTH * VOCAB_SIZE * sizeof(float));
    if (!h_logits) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    forwardPass(&model, input_seq, h_logits);
    printf("Logits for first token:\n");
    for (int i = 0; i < VOCAB_SIZE; i++) {
        printf("%f ", h_logits[i]);
    }
    printf("\n");
    free(h_logits);

    // Inference demo: generate 50 tokens.
    printf("Generated text (ASCII codes): ");
    int current_seq[SEQ_LENGTH];
    memcpy(current_seq, input_seq, SEQ_LENGTH * sizeof(int));
    for (int t = 0; t < 50; t++) {
        int nextToken = inferNextToken(&model, current_seq);
        printf("%c", (char)nextToken);
        for (int i = 0; i < SEQ_LENGTH - 1; i++) {
            current_seq[i] = current_seq[i+1];
        }
        current_seq[SEQ_LENGTH - 1] = nextToken;
    }
    printf("\n");

    freeModel(&model);
    return 0;
}

