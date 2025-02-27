# 100-daysofcuda
 Kernels Written for 100 days of CUDA Challenge

**Day 1:** Wrote a Naive Matmul kernel

**Day 2:** Wrote a 1D tiled matmul kernel

**Day 3:** Oversimplified 2D tiled matmul kernel

**Day 4:** 2D matmul kernel with CPU verification and great memory management

**Day 5:** 3D matmul tiling!

**Day 6:** Softmax flashattention online

**Day 7:** Layer Normalization

**Day 8:** Better layer normalization with lots of nice modifications

**Day 9:** Implimented a 2D tiled convlutional kernel


**Day 10:** Implimented Vectorized matmul !! Got first Batch

**Day 11:** Implimented merge sort algorithm using CUDA

**Day 12:** Wrote flash attention with backpropagation

**Day 13:** Learned about GNNs and wrote Kernel to impliment a layer of GNNs

**Day 14:** Wrote N body simulation kernel inspired by 1y33

**Day 15:** Random weight selection kernel for neural networks

**Day 16:** Full Fledged Transformer model

**Day 17:** Optimized the transformer by adding tiled matmul and shared memory in softmax !

**Day 18:** More Optimizations like loop unrolling and tiled transposed matrix B and Warp level reductions in Softmax

**Day 19:** Learned and integrated FP16, Fusion kernels, Online softmax, registers etc in the transformer kernel

**Day 20:** Wrote a full neural network with forward pass, SGD, backpropagation

**Day 21:** Tried writing full CNN For mnist but failed lots lots and lots of time

**Day 22:** Shifted focus from Implimenting everything from scratch to cuDNN. Reimplimented the CNN Mnist project in cuDNN and managed to compile it with some logical errors still there

**Day 23:** Again refocused on Implimenting everything from scratch as cuDNN version was pretty difficult to maintain. Did huge progress by integrating everything in one file and almost reaching excellence in doing it

**Day 24:** Reimagined the whole code and after 10 hours of tinkering around with nested loops wrote the perfect CUDA CNN kernel for MNIST

**Day 25:** Implimented Native sparse attention Paper by deepseek for an nvidia H100 GPU

**Day 26:** Tokenizer in CUDA

**Day 27:** Learned how to do an operation for 1000s of GPUs using MPI and NCCL

**Day 28:** Wrote kernel for Full LLM (don't know how if it works perfect)

**Day 29:** LLM Kernel code review

**Day 30:** Rethought the whole LLM kernel and fit that into a single file

**Day 30:** Debugged the LLM kernel a lot
