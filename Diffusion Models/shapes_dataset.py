import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def create_shape_dataset(num_samples=1000, image_size=28):
    """
    Creates a dataset of simple geometric shapes (circles, squares, triangles)
    """
    images = []
    
    for _ in range(num_samples):
        # Create a blank image
        image = np.zeros((image_size, image_size))
        
        # Randomly choose shape type
        shape_type = np.random.choice(['circle', 'square', 'triangle'])
        
        # Random position and size
        center_x = np.random.randint(image_size//4, 3*image_size//4)
        center_y = np.random.randint(image_size//4, 3*image_size//4)
        size = np.random.randint(image_size//6, image_size//3)
        
        if shape_type == 'circle':
            # Create circle
            y, x = np.ogrid[:image_size, :image_size]
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            circle = dist_from_center <= size
            image[circle] = 1.0
            
        elif shape_type == 'square':
            # Create square
            x_start = max(0, center_x - size//2)
            x_end = min(image_size, center_x + size//2)
            y_start = max(0, center_y - size//2)
            y_end = min(image_size, center_y + size//2)
            image[y_start:y_end, x_start:x_end] = 1.0
            
        else:  # triangle
            # Create triangle
            points = np.array([
                [center_x, center_y - size],
                [center_x - size, center_y + size],
                [center_x + size, center_y + size]
            ])
            
            # Create triangle mask
            y, x = np.mgrid[:image_size, :image_size]
            points = points.astype(int)
            from matplotlib.path import Path
            path = Path(points)
            mask = path.contains_points(np.vstack((x.flatten(), y.flatten())).T)
            image[mask.reshape(image_size, image_size)] = 1.0
        
        images.append(image)
    
    # Convert to numpy array and add channel dimension
    images = np.array(images)[..., np.newaxis]
    
    # Convert to TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.shuffle(1000).batch(32)
    
    return dataset

# Create dataset
dataset = create_shape_dataset()

# Visualize some examples
def plot_samples(dataset, num_samples=5):
    plt.figure(figsize=(15, 3))
    for i, image in enumerate(dataset.unbatch().take(num_samples)):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(image.numpy().squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()

# Show sample images
plot_samples(dataset)

