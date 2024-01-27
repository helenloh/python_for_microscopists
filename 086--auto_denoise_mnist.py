#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# Objective :To create a denoise autoencoder. 
# Purpose: The autoencoder learn to separate noise from signal

# Operation: Read this to understand why we need to add noise to image
# In your script, x_train represents the original, clean images from the MNIST dataset. The MNIST dataset is a collection of handwritten digit images commonly used for training various image processing systems. These images are generally clean and free from noise, making them ideal as a baseline or ground truth for tasks like training a denoising autoencoder.

# In the context of your script, the process is as follows:

#     Original Clean Images (x_train): This dataset is the collection of original MNIST images. These images are used as the target or ground truth in the training process. The model learns to recreate these images as closely as possible.

#     Noisy Images (x_train_noisy): These are created by adding artificial noise to the original clean images from x_train. This step simulates real-world scenarios where images might be corrupted with noise.

# During the training of the denoising autoencoder:

#     The input to the model is the noisy images (x_train_noisy).
#     The target output the model is trying to achieve is the clean images (x_train).

# By doing this, the model learns how to filter out the noise and reconstruct the clean image from the noisy input. The effectiveness of the model is usually measured by how closely the output resembles the original clean images after being trained on a mixture of noisy and clean data.


# Import necessary libraries
from tensorflow.keras.datasets import mnist  # Import MNIST dataset from Keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D  # Import layers for CNN
from tensorflow.keras.models import Sequential  # Import Sequential model type from Keras

import numpy as np  # Import numpy for numerical calculations
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Load the MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()  # Load training and test sets (ignoring labels)

# Purpose of image scaling in range 0-1
# the MNIST images are scaled from 0-255 to 0-1. 
# This normalization simplifies the training process 
# and is particularly beneficial for image data 
# and models using gradient descent-based optimization.
# Preprocess the data
x_train = x_train.astype('float32') / 255.  # Normalize training images
x_test = x_test.astype('float32') / 255.  # Normalize test images
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # Reshape training images
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # Reshape test images

# Add noise to the images
# Adding noise to images is a crucial step in training a denoising autoencoder.
# The idea is to artificially create noisy versions of the training images,
# so the model can learn how to reconstruct the original, clean image from the noisy input.
# This process helps the model to learn to ignore the noise and focus on the key features of the image.
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Displaying images with noise
plt.figure(figsize=(20, 2))  # Set the figure size
for i in range(1, 10):
    ax = plt.subplot(1, 10, i)  # Create a subplot for each image
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap="binary")  # Display the noisy image
plt.show()  # Show the plot

# Build the model
model = Sequential()  # Create a Sequential model
# Add layers to the model
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))  # Convolutional layer
model.add(MaxPooling2D((2, 2), padding='same'))  # Max pooling layer
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))  # Convolutional layer
model.add(MaxPooling2D((2, 2), padding='same'))  # Max pooling layer
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))  # Convolutional layer
model.add(MaxPooling2D((2, 2), padding='same'))  # Max pooling layer
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))  # Convolutional layer
model.add(UpSampling2D((2, 2)))  # Upsampling layer
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))  # Convolutional layer
model.add(UpSampling2D((2, 2)))  # Upsampling layer
model.add(Conv2D(32, (3, 3), activation='relu'))  # Convolutional layer
model.add(UpSampling2D((2, 2)))  # Upsampling layer
model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))  # Convolutional layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')  # Compile the model using Adam optimizer and MSE loss function

# Print model summary
model.summary()  # Display the model's architecture

# Train the model
model.fit(x_train_noisy, x_train, epochs=10, batch_size=256, shuffle=True, 
          validation_data=(x_test_noisy, x_test))  # Fit the model to the noisy
model.evaluate(x_test_noisy, x_test)

model.save('denoising_autoencoder.model')

no_noise_img = model.predict(x_test_noisy)

plt.figure(figsize=(40, 4))
for i in range(10):
    # display original
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap="binary")
    
    # display reconstructed (after noise removed) image
    ax = plt.subplot(3, 20, 40 +i+ 1)
    plt.imshow(no_noise_img[i].reshape(28, 28), cmap="binary")

plt.show()
