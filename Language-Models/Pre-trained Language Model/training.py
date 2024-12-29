import tensorflow as tf
from pretrained import model
from preprocessing import tf_train_dataset

# Implement gradient clipping
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# compile the model
# model.compile(optimizer=optimizer, loss=loss)
model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer])

# Fine-tune the model
# model.fit(tf_train_dataset, epochs=20)
history = model.fit(tf_train_dataset, epochs=10, verbose=1)
