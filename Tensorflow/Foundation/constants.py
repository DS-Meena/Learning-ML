import tensorflow as tf

tensor_3d = tf.constant([[[1, 2]], [[3, 4]]], dtype=tf.int16)

print(f"Tensor: {tensor_3d}")
print(f"Shape: {tensor_3d.shape}")
print(f"Data type: {tensor_3d.dtype}")

tesnor_2d = tf.constant([[1, 2], [3, 4]])
scalar = tf.constant(42.0)

print(f"Tensor: {tesnor_2d} \n and shape: {tesnor_2d.shape}")
print(f"Tensor: {scalar}")