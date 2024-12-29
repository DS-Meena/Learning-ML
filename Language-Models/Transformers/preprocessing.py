import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Reading the Data 
my_data = pd.read_csv("all.csv")
poems = my_data.loc[my_data['type'] == 'Love']['content']
poems = list(poems)

# Clean and preprocess the text
def preprocess_text(text):
    # Remove extra whitespace and convert to lowercase
    text = ' '.join(text.split()).lower()
    return text

cleaned_poems = [preprocess_text(poem) for poem in poems]

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(cleaned_poems)
vocab_size = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
for poem in cleaned_poems:
    # numerical representation of poem
    token_list = tokenizer.texts_to_sequences([poem])[0]

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_seq_length = max([len(seq) for seq in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_length, padding='pre'))

# Create predictors and label
X, y = input_sequences[:, :-1], input_sequences[:, 1:]

# Convert y to categorical
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)