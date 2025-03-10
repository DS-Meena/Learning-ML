import tensorflow as tf
import tensorflow.experimental.numpy as tnp

x = tf.math.count_nonzero(tnp.ones([3, 3]))
y = tnp.count_nonzero(tf.ones([3, 3]))

print(x)
print(y)

ones = tnp.ones([1, 1, 2, 2, 3], dtype=tnp.float16)

# print(ones)