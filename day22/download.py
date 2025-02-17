import struct
import numpy as np
from tensorflow.keras.datasets import mnist

def save_idx_images(filename, images):
    # images: numpy array of shape (num_images, rows, cols) in uint8
    num_images, rows, cols = images.shape
    with open(filename, 'wb') as f:
        # magic number 2051 (0x00000803)
        f.write(struct.pack('>I', 2051))
        f.write(struct.pack('>I', num_images))
        f.write(struct.pack('>I', rows))
        f.write(struct.pack('>I', cols))
        f.write(images.tobytes())

def save_idx_labels(filename, labels):
    # labels: numpy array of shape (num_labels,) in uint8
    num_labels = labels.shape[0]
    with open(filename, 'wb') as f:
        # magic number 2049 (0x00000801)
        f.write(struct.pack('>I', 2049))
        f.write(struct.pack('>I', num_labels))
        f.write(labels.tobytes())

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Ensure data is in uint8 format
    x_train = x_train.astype(np.uint8)
    x_test = x_test.astype(np.uint8)
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    save_idx_images('train-images.idx3-ubyte', x_train)
    save_idx_labels('train-labels.idx1-ubyte', y_train)
    save_idx_images('t10k-images.idx3-ubyte', x_test)
    save_idx_labels('t10k-labels.idx1-ubyte', y_test)

    print("MNIST IDX files have been created.")

if __name__ == '__main__':
    main()

