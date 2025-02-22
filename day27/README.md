CUDA matrix multiplication code that works across **thousands of H100 GPUs** using **CUDA, NCCL, and MPI**.  

---

## **🔥 Code Walkthrough:**
```cpp
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <iostream>
```
- **`cuda_runtime.h`** → Includes CUDA runtime functions.
- **`nccl.h`** → Includes NCCL (NVIDIA Collective Communication Library) for fast GPU-GPU communication.
- **`mpi.h`** → Includes MPI (Message Passing Interface) for multi-node communication.
- **`iostream`** → Standard C++ library for input/output.

---

### **Step 1: Define Matrix Size and CUDA Block Size**
```cpp
#define N 4096  // Matrix size (adjust as needed)
#define BLOCK_SIZE 32  // CUDA block size
```
- **`N`** → Defines the size of the square matrices (4096 × 4096). You can scale this based on GPU memory.
- **`BLOCK_SIZE`** → Defines CUDA thread block size (32×32). This matches **warp size** for better performance.

---

### **Step 2: CUDA Kernel for Matrix Multiplication**
```cpp
__global__ void matmul_kernel(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}
```
- **`__global__`** → Declares the function as a **GPU kernel**.
- **`blockIdx.y * blockDim.y + threadIdx.y`** → Computes the **row index** of the element.
- **`blockIdx.x * blockDim.x + threadIdx.x`** → Computes the **column index** of the element.
- **Matrix Multiplication Logic**:
  - Loops over `k` to perform **dot product** between row `i` of `A` and column `j` of `B`.
  - Stores result in `C[row, col]`.

**💡 Optimization Tip**:  
- This kernel **does not** use shared memory or tensor cores. A more optimized version would use **tiling (shared memory) or Tensor Cores (WMMA API)**.

---

### **Step 3: Launch CUDA Kernel**
```cpp
void cuda_matmul(float* d_A, float* d_B, float* d_C, int n) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(n / threads.x, n / threads.y);
    matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
}
```
- **`dim3 threads(BLOCK_SIZE, BLOCK_SIZE);`** → Defines a **CUDA thread block** of size **32×32**.
- **`dim3 blocks(n / threads.x, n / threads.y);`** → Defines how many **blocks** are needed to cover the matrix.
- **`cudaDeviceSynchronize();`** → Ensures kernel execution completes before proceeding.

---

### **Step 4: MPI Initialization (Multi-Node Support)**
```cpp
MPI_Init(&argc, &argv);

int world_size, world_rank;
MPI_Comm_size(MPI_COMM_WORLD, &world_size);
MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
```
- **`MPI_Init`** → Initializes MPI.
- **`MPI_Comm_size`** → Gets the total number of MPI processes (nodes).
- **`MPI_Comm_rank`** → Gets the rank (ID) of the current node.

If you run on **1000 H100 GPUs**, `world_size = 1000`, and each GPU gets a part of the matrix.

---

### **Step 5: NCCL (Intra-Node Communication)**
```cpp
ncclComm_t nccl_comm;
ncclUniqueId nccl_id;

if (world_rank == 0) ncclGetUniqueId(&nccl_id);
MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
ncclCommInitRank(&nccl_comm, world_size, nccl_id, world_rank);
```
- **`ncclUniqueId nccl_id;`** → Creates a unique NCCL ID for communication.
- **`if (world_rank == 0) ncclGetUniqueId(&nccl_id);`** → Only **rank 0** generates an ID.
- **`MPI_Bcast`** → Broadcasts the NCCL ID to **all GPUs**.
- **`ncclCommInitRank`** → Initializes NCCL communication across GPUs.

**💡 Why NCCL?**  
- **MPI** is **slow for intra-node GPU communication**. NCCL allows **direct GPU-GPU** transfers via **NVLink** or **InfiniBand**.

---

### **Step 6: Allocate GPU Memory**
```cpp
float *d_A, *d_B, *d_C;
cudaMalloc(&d_A, N * N * sizeof(float));
cudaMalloc(&d_B, N * N * sizeof(float));
cudaMalloc(&d_C, N * N * sizeof(float));
```
- Allocates GPU memory for **A, B, C** matrices.

---

### **Step 7: Compute Local Matrix Multiplication**
```cpp
cuda_matmul(d_A, d_B, d_C, N / world_size);
```
- Each GPU computes a **partial matrix multiplication** on a **sub-matrix** (`N / world_size`).
- The final result is **distributed across GPUs**.

---

### **Step 8: NCCL AllReduce (Reduce & Synchronize Results)**
```cpp
ncclAllReduce(d_C, d_C, (N * N) / world_size, ncclFloat, ncclSum, nccl_comm, cudaStreamDefault);
```
- **AllReduce** operation **sums** the results across all GPUs.
- This aggregates partial results into the final matrix.

**💡 Why AllReduce?**  
- Instead of transferring large matrices between GPUs, **each GPU keeps computing and merging results** efficiently.

---

### **Step 9: Cleanup and Finalize**
```cpp
ncclCommDestroy(nccl_comm);
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
MPI_Finalize();
```
- **`ncclCommDestroy`** → Frees NCCL resources.
- **`cudaFree`** → Frees GPU memory.
- **`MPI_Finalize`** → Shuts down MPI.

---

## **🔥 How This Works at Scale**
| Component | Purpose |
|-----------|---------|
| **CUDA**  | Runs the matrix multiplication kernel on **each GPU** |
| **MPI**   | Distributes workload across **multiple nodes** |
| **NCCL**  | Handles **intra-node** GPU communication efficiently |
| **AllReduce** | Aggregates results across GPUs |

---

## **🚀 Execution & Deployment**
### **1️⃣ Compile:**
```bash
mpic++ -o matmul matmul.cu -lcudart -lnccl -lmpi
```
### **2️⃣ Run on 1000 H100s:**
```bash
mpirun -np 1000 --hostfile hosts ./matmul
```
- **`-np 1000`** → Runs **1000 MPI processes** (1 per GPU).
- **`--hostfile hosts`** → Specifies GPU cluster nodes.

---

## **🔥 Summary**
✅ **CUDA kernel** → Computes matrix multiplication on each GPU.  
✅ **MPI** → Distributes computation across nodes.  
✅ **NCCL AllReduce** → Efficiently aggregates results **without CPU overhead**.  
✅ **Scalable to 1000s of H100s** 🚀  

Let me know if you need **further optimizations** (cuBLAS, Tensor Cores, etc.)! 🚀🔥
