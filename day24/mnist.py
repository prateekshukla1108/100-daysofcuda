import tensorflow as tf
import numpy as np

# Load MNIST dataset from TensorFlow
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize and flatten images, convert to float32
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0
train_images = train_images.reshape(train_images.shape[0], 784) # 28*28 = 784
test_images = test_images.reshape(test_images.shape[0], 784)

# Convert labels to int32
train_labels = train_labels.astype(np.int32)
test_labels = test_labels.astype(np.int32)


# Save the datasets to binary files
train_images.tofile("mnist_train_images.bin")
train_labels.tofile("mnist_train_labels.bin")
test_images.tofile("mnist_test_images.bin")
test_labels.tofile("mnist_test_labels.bin")

print("MNIST dataset downloaded and saved in binary format.")
print("Files saved:")
print("- mnist_train_images.bin")
print("- mnist_train_labels.bin")
print("- mnist_test_images.bin")
print("- mnist_test_labels.bin")
