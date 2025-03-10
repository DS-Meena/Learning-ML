import tensorflow as tf

# Reassigning variables
a = tf.Variable([10, 3])
a.assign([7, 4])

print(a)