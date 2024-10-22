import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt
import numpy as np

# Load a sample image
china = load_sample_image("china.jpg")
china = np.array(china, dtype=np.float32) / 255.0

# Reshape the image to add a batch dimension
china = china.reshape((1,) + china.shape)

# Create filters
gaussian_blur = np.array([[1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]]) / 16.0

edge_detection = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]])

sharpen = np.array([[0, -1,  0],
                    [-1, 5, -1],
                    [0, -1,  0]])

vertical_lines = np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]])

horizontal_lines = np.array([[-1, -1, -1],
                             [0, 0, 0],
                             [1, 1, 1]])

# Define the convolutional layers
conv_gaussian = Conv2D(filters=1, kernel_size=3, padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.Constant(gaussian_blur))
conv_edge = Conv2D(filters=1, kernel_size=3, padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.Constant(edge_detection))
conv_sharpen = Conv2D(filters=1, kernel_size=3, padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.Constant(sharpen))
conv_vertical = Conv2D(filters=1, kernel_size=3, padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.Constant(vertical_lines))
conv_horizontal = Conv2D(filters=1, kernel_size=3, padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.Constant(horizontal_lines))

# Apply the convolutional layers
gaussian_output = conv_gaussian(china)
edge_output = conv_edge(china)
sharpen_output = conv_sharpen(china)
vertical_output = conv_vertical(china)
horizontal_output = conv_horizontal(china)

# Define the pooling layer
pool_layer = MaxPooling2D(pool_size=2, strides=2, padding='valid')

# Apply the pooling layer
pool_output = pool_layer(edge_output)

# Function to display images
def show_images(images, titles):
    rows, cols = 2, 4
    fig, axs = plt.subplots(rows, cols, figsize=(20,10))
    for i, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 4:
            img = img[0]  # Remove batch dimension
        
        if img.shape[-1] == 1:
            img = np.squeeze(img) # Remove single-dimensional entries
        
        row, col = i//cols,i%cols        
        axs[row, col].imshow(img)
        axs[row, col].set_title(title)
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.savefig("Custom-filters.png")
    plt.close()

# Display results
show_images([china[0], gaussian_output[0], edge_output[0], sharpen_output[0], vertical_output[0], horizontal_output[0], pool_output[0]],
            ['Original', 'Gaussian Blur', 'Edge Detection', 'Sharpened', 'Vertical Lines', 'Horizontal Lines', 'Edge Detection + Pooling'])