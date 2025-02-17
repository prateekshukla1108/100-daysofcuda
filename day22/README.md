Got the MNIST CNN compiling with cuDNN implimentation. Still there are some logic errors but I got the kernel compiling!

compile it with 

```bash
nvcc -o cnn_mnist_executable cnn_mnist_host.cu cnn_mnist_kernels.cu -lcudnn
```

run it with 

```bash
./cnn_mnist_executable
```
