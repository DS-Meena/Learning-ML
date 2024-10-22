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

# This filter is to detect sharp edges
kernel = np.array([[0, -1,  0],
                    [-1, 5, -1],
                    [0, -1,  0]])

# Define the convolutional layer
conv_layer = Conv2D(filters=1, kernel_size=3, padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.Constant(kernel))
conv_output = conv_layer(china)

# Define the pooling layer
pool_layer = MaxPooling2D(pool_size=2, strides=2, padding='valid')

# Apply the pooling layer
pool_output = pool_layer(conv_output)

# Function to display images
def show_images(images, titles):
    fig, axs = plt.subplots(1, len(images), figsize=(15,5))
    for i, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 4:
            img = img[0]  # Remove batch dimension
        
        if img.shape[-1] == 1:
            img = np.squeeze(img) # Remove single-dimensional entries
        
        # axs[i].imshow(img)  # color
        axs[i].imshow(img, cmap='gray') # gray

        axs[i].set_title(title)
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig("result-gray.png")
    plt.close()

# Display results
# show_images([china[0], conv_output[0], pool_output[0]], ['Original', 'After Convolution', 'After Pooling']) # color
show_images([china[0], conv_output[0, :, :, 0], pool_output[0, :, :, 0]], ['Original', 'After Convolution', 'After Pooling']) # gray