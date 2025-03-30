import numpy as np
import tensorflow as tf

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize images (scale pixel values between 0 and 1)
x_train = x_train.reshape(-1, 784) / 255.0  # Flatten 28x28 images to 1D
x_test = x_test.reshape(-1, 784) / 255.0

# Combine labels and pixel data
train_data = np.column_stack((y_train, x_train))  # [label, pixel1, pixel2, ..., pixel784]
test_data = np.column_stack((y_test, x_test))

# Save to CSV (no headers for fast C++ parsing)
np.savetxt("mnist_train.csv", train_data, delimiter=",", fmt="%.5f")
np.savetxt("mnist_test.csv", test_data, delimiter=",", fmt="%.5f")

print("MNIST dataset has been successfully saved as CSV!")
