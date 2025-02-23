#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

// For this demo, we use a simple character-level tokenizer.
// Each character's ASCII code (0-255) is its token.
void tokenize(const char* text, int** tokens, size_t* tokenCount) {
    size_t len = strlen(text);
    *tokenCount = len;
    *tokens = (int*)malloc(len * sizeof(int));
    if (!(*tokens)) {
        fprintf(stderr, "Token allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < len; i++) {
        // Optionally, you could lowercase letters:
        char ch = tolower(text[i]);
        (*tokens)[i] = (int)ch;  // token is ASCII code
    }
}

