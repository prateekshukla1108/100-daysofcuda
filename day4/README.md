cuda kernel which uses 2D tiling with shared memory to improve memory access patterns

Tiles are loaded iteratively from input matrices to maximize data reuse

added CPU reference implementation for result verification

Square thread blocks for better shared memory utilization

it worked in P100(colab gpu)!

you need to use nvcc -O3 -arch=sm_70 -o matmul matmul.cu to complile it without it you might fall into bank conflict
