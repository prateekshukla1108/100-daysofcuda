import tensorflow as tf
import numpy as np

def main():
    # Load the MNIST dataset from TensorFlow/Keras
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # The images are loaded as (num_samples, 28, 28) arrays of type uint8.
    # Flatten each 28x28 image into a 784-element 1D array.
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)

    # Optionally, verify the data types
    print("Training images:", x_train.shape, x_train.dtype)
    print("Training labels:", y_train.shape, y_train.dtype)

    # Save the data to binary files.
    # The CUDA code expects raw bytes (unsigned char values) for the images and labels.
    x_train.tofile("mnist_train_images.bin")
    y_train.tofile("mnist_train_labels.bin")
    x_test.tofile("mnist_test_images.bin")
    y_test.tofile("mnist_test_labels.bin")

    print("MNIST data saved as binary files.")

if __name__ == "__main__":
    main()


