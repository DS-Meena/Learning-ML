from preprocessing import tokenizer
from model import seq_length
from model import model

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def generate_poem(model, seed_text, num_words):
    generated_text = seed_text
    for _ in range(num_words):
        # Tokenize and pad the input sequence
        sequence = tokenizer.texts_to_sequences([generated_text])
        padded_sequence = pad_sequences(sequence, maxlen=seq_length, padding='pre')
        
        # Predict the next word
        predicted = model.predict(padded_sequence)
        predicted_word_index = np.argmax(predicted)
        predicted_word = tokenizer.index_word[predicted_word_index]
        
        # Append the predicted word to the generated text
        generated_text += " " + predicted_word
    
    return generated_text

# Generate a poem
seed_text = "Love is"
poem = generate_poem(model, seed_text, 50)
print(poem)