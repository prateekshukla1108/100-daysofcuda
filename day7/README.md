
### **1. CUDA Kernel Execution Model**
- Each **block** processes one row (one instance) of the input data.
- Each **thread** within the block handles multiple elements in that row.

---

### **2. Shared Memory for Efficiency**
- We use **shared memory** for computing the sum and sum of squares, which helps speed up reduction (computing mean and variance).
- Each thread contributes to the computation and reduces memory access to slow **global memory**.

---

### **3. Parallel Reduction**
- The kernel performs a two-step process:
  1. **Summation across threads** to compute the mean and variance.
  2. **Normalization and affine transformation** using gamma and beta.

---

### **4. Error Checking (CUDA_CHECK Macro)**
- Every **CUDA function call** (e.g., `cudaMalloc`, `cudaMemcpy`, `cudaDeviceSynchronize`) is wrapped in a **macro** to check for errors.
- If an error occurs, it prints an error message and stops execution.
- This helps catch issues like:
  - Memory allocation failures.
  - Kernel launch problems.
  - Synchronization errors.

---

### **5. Kernel Launch Configuration**
- We determine the **number of threads per block** dynamically based on `feature_size`, but it defaults to 256 for efficiency.
- We allocate **shared memory** dynamically to store intermediate results.

---

### **6. Host Code Workflow**
1. Allocate memory on the **CPU (host)** and **GPU (device)**.
2. Copy input data and parameters (`gamma`, `beta`) to the GPU.
3. Launch the **CUDA kernel** with the right configuration.
4. Copy the results back to the **host**.
5. Free the allocated memory.

---

### **7. Key Benefits of This Implementation**
- **Optimized Performance**: Uses shared memory and parallel computation.
- **Robustness**: Error checking ensures smooth debugging.
- **Flexibility**: Supports varying batch and feature sizes.

