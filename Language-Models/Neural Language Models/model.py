from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam

vocab_size = 7231 #  Size of vocabulary
seq_length = 2228 #  Length of sequences
embedding_dim = 100  # Dimension of the embedding layer

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=seq_length),
    LSTM(128, return_sequences=True),
    LSTM(128),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))