# Statistical Language Model

A SLM generates sentences based on the statistics or n-gram frequency.

## Building a Statistical Language Model (SLM) 🧠

1. **Data Collection** 📚:
    - Gather a large and relevant corpus of text data. This can be from books, articles, websites, etc.
2. **Preprocessing** 🧹:
    - Clean the text data by removing any non-text elements, punctuation (if necessary), and converting text to lowercase.
    - Tokenize the text into words or characters.
    - Optionally, stop words can be removed depending on your use case.
3. **Generate N-grams** 🔄:
    - Create N-grams from the tokenized text. N-grams are sequences of N items (words or characters).
    - For example, for bigrams (N=2), the sentence "I love coding" would be split into ["I love", "love coding"].
4. **Calculate Frequencies** 📊:
    - Count the frequency of each N-gram in the corpus.
    - Store these frequencies in a data structure like a dictionary or a Counter object from the `collections` module in Python.
5. **Calculate Probabilities** 🔢:
    - Calculate the probability of each N-gram based on its frequency.
    - For a bigram model, the probability of a word given the previous word can be calculated as:
    $P(w_n | w_{n-1}) = \frac{\text{Count}(w_{n-1}, w_n)}{\text{Count}(w_{n-1})}$

## Training the Model 🏋️

1. **Smoothing**:
    - Apply smoothing techniques to handle zero probabilities (unseen N-grams). Common techniques include Add-One (Laplace) Smoothing, Good-Turing Smoothing, etc.
2. **Model Evaluation**:
    - Split your data into training and validation sets.
    - Train your model on the training set and evaluate its performance on the validation set using metrics like perplexity.
3. **Iterate**:
    - Based on the evaluation, iterate on your preprocessing steps, N-gram size, and smoothing techniques to improve the model's performance.

# Results ❤️‍🔥

For a given corpus  ["I love coding", "I love programming", "I enjoy learning"], when I tried to generate sentence with starting word as "i" I get "i love" as result.
 
input: "i"

output: "i love"

```bash
tokenized_sentences: [['i', 'love', 'coding'], ['i', 'love', 'programming'], ['i', 'enjoy', 'learning']]
bigrams: [('i', 'love'), ('love', 'coding'), ('i', 'love'), ('love', 'programming'), ('i', 'enjoy'), ('enjoy', 'learning')]

bigrams_freq: Counter({('i', 'love'): 2, ('love', 'coding'): 1, ('love', 'programming'): 1, ('i', 'enjoy'): 1, ('enjoy', 'learning'): 1})

next_word_candidates and next_word_probabilities for "i" as current word:
['love', 'enjoy'] [0.3333333333333333, 0.2222222222222222]
```