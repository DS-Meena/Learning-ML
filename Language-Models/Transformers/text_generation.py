import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing import tokenizer, max_seq_length
from model import model

def generate_poem(model, start_string, num_generate=20, temperature=1.0):
    input_eval = tokenizer.texts_to_sequences([start_string])[0]
    input_eval = pad_sequences([input_eval], maxlen=max_seq_length-1, padding='post')
    text_generated = []
    
    for _ in range(num_generate):
        predicted_id = 0
        while predicted_id == 0:   
            predictions = model(input_eval, training=False)
            predictions = tf.squeeze(predictions, 0)
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        
        text_generated.append(tokenizer.index_word[predicted_id])
        
        input_eval = tf.concat([input_eval[:, 1:], [[predicted_id]]], axis=-1)
        
    return (start_string + ' ' + ' '.join(text_generated))

print(generate_poem(model, start_string="love is"))