from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, LayerNormalization, MultiHeadAttention
from preprocessing import vocab_size

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim), ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  

# Model architecture
class PoemGenerator(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim):
        super(PoemGenerator, self).__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.final_layer = Dense(vocab_size)
    
    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.transformer_block(x, training=training)
        return self.final_layer(x)

# Compile the model
embedding_dim = 64

model = PoemGenerator(vocab_size, embed_dim=embedding_dim, num_heads=2, ff_dim=32)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
