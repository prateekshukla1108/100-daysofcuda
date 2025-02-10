 This kernel basically impliments something that can be alternative of backprop. Instead of calculating gradient and doing all that it just randomly takes lots of parameters and finds value of loss function and chooses the parameter which minimises loss and then takes values in neighbourhood of that and again calculate the loss function and find the weight with least loss function.

you can run it using nvcc neural_net_parallel_exploration_layerwise.cu -o neural_net_parallel_exploration_layerwise and then ./neural_net_parallel_exploration_layerwise
