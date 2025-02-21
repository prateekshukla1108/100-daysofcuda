# Day 26: CUDA-Based Text Tokenization

## 1. Project Overview
This project implements a parallel text tokenization system using NVIDIA CUDA. The program reads text from an input file ("input.txt"), processes it on the GPU to split it into tokens (words) separated by whitespace, and writes the resulting tokens to an output file ("tokens.txt"), with each token on a new line.

### Objectives
- Utilize GPU parallel processing to tokenize text efficiently.
- Demonstrate CUDA programming concepts including kernel design, memory management, and thread synchronization.
- Handle file I/O and error checking robustly.

### Key Features
- Parallel tokenization using a CUDA kernel.
- Dynamic memory allocation on both host and device.
- Atomic operations for thread-safe output buffer management.

---

## 2. System Design

### Components
1. **Host Code (main function)**:
   - File I/O: Reads input from "input.txt" and writes output to "tokens.txt".
   - Memory Management: Allocates and frees memory on both host and device.
   - CUDA Setup: Configures and launches the kernel, manages data transfers between host and device.
   - Error Handling: Comprehensive checks for file operations, memory allocation, and CUDA API calls.

2. **Device Code (tokenizeKernel)**:
   - A CUDA kernel that processes the input text in parallel to identify and extract tokens.
   - Uses atomic operations to safely write tokens to a shared output buffer.

### Workflow
1. Read the entire input file into host memory.
2. Allocate device memory for input, output, and an output offset counter.
3. Copy input data to the GPU.
4. Launch the CUDA kernel to tokenize the text.
5. Copy the tokenized output back to host memory.
6. Write the output to a file and clean up resources.

---

## 3. Implementation Details

### CUDA Kernel: `tokenizeKernel`
```c
__global__ void tokenizeKernel(const char* input, int inputLength, char* output, int* outputOffset)
```
- **Purpose**: Identifies tokens in the input string and writes them to the output buffer.
- **Thread Assignment**: Each thread processes one character of the input string based on its global index (`idx`).
- **Token Detection**:
  - A token starts when a non-whitespace character follows a whitespace character (or is at the beginning).
  - The token ends at the next whitespace character or the end of the input.
- **Whitespace Definition**: Defined in `is_whitespace` as space (`' '`), newline (`'\n'`), or tab (`'\t'`).
- **Output Writing**:
  - Uses `atomicAdd` to reserve space in the output buffer for each token, ensuring thread safety.
  - Writes the token characters followed by a newline (`'\n'`).
- **Optimization**: Inline `is_whitespace` function for performance.

### Host Code: `main`
- **Input Handling**: Reads "input.txt" into a host buffer using standard C file operations.
- **Memory Management**:
  - Host: Allocates buffers for input and output using `malloc`.
  - Device: Allocates buffers using `cudaMalloc` for input, output, and offset counter.
- **Kernel Launch**: Configures a grid with 256 threads per block and calculates the number of blocks based on input size.
- **Synchronization**: Uses `cudaDeviceSynchronize` to ensure kernel completion.
- **Output Retrieval**: Copies the final output size and content from the device to the host.

### Key Parameters
- **Thread Block Size**: 256 threads per block (configurable).
- **Output Buffer Size**: Initialized to twice the input size to accommodate tokens and newlines.

---

## 4. Performance Analysis
- **Parallelism**: The kernel leverages CUDA's massive parallelism by assigning each character to a thread, making it efficient for large texts.
- **Bottlenecks**:
  - Atomic operations (`atomicAdd`) may cause contention when many threads detect tokens simultaneously.
  - Memory transfers between host and device could be a limiting factor for small inputs.
- **Scalability**: Scales well with input size due to the grid-block-thread hierarchy.

### Theoretical Speedup
- Compared to a serial CPU implementation, the CUDA version benefits from processing thousands of characters in parallel, potentially achieving a speedup proportional to the number of CUDA cores available.

---

## 5. Error Handling
- **File Errors**: Checks for file opening and reading failures.
- **Memory Errors**: Validates host and device memory allocations.
- **CUDA Errors**: Checks all CUDA API calls (e.g., `cudaMalloc`, `cudaMemcpy`, kernel launch) and reports errors with descriptive messages.

---

## 6. Results
- **Input**: Text file "input.txt" containing arbitrary text.
- **Output**: File "tokens.txt" with one token per line.
- **Success Message**: "Tokenization complete. Output written to tokens.txt" printed upon completion.

### Sample Execution
For an input file "input.txt" containing:
```
Hello world this is a test
```
The output file "tokens.txt" would contain:
```
Hello
world
this
is
a
test
```

---

## 7. Limitations
- **Buffer Overflow**: The output buffer size (2x input size) may be insufficient for inputs with many small tokens.
- **Delimiter**: Only supports basic whitespace characters as delimiters.
- **Thread Efficiency**: Some threads may exit early if they don’t process a token start, reducing utilization.

---

## 8. Future Improvements
- **Dynamic Output Sizing**: Implement a two-pass approach to calculate exact output size first.
- **Custom Delimiters**: Allow user-defined token separators.
- **Optimization**: Reduce atomic operation contention using per-block buffers or prefix-sum techniques.
- **Error Recovery**: Add retry mechanisms for recoverable errors (e.g., file access).

---

## 9. Conclusion
This CUDA-based tokenization project successfully demonstrates GPU-accelerated text processing. It efficiently parallelizes token extraction, handles memory and errors robustly, and provides a foundation for further optimization. The implementation showcases key CUDA concepts like kernel design, memory management, and synchronization, making it a valuable educational and practical tool.


