import nltk
from collections import Counter, defaultdict
import random

corpus = ["I love coding", "I love programming", "I enjoy learning"]

# Preprocessing
tokenized_sentences = [nltk.word_tokenize(sentence.lower()) for sentence in corpus]

# Generate bigrams
bigrams = [list(nltk.bigrams(sentence)) for sentence in tokenized_sentences]
bigrams = [bigram for sentence in bigrams for bigram in sentence] # flatten list

# Calculate bigram frequencies
bigram_freq = Counter(bigrams)
unigram_freq = Counter([word for sentence in tokenized_sentences for word in sentence]) # unigram freq

# Calculate bigram probabilities with Add-One smoothing
bigram_prob = defaultdict(lambda: 0)
vocab_size = len(unigram_freq)

for (w1, w2), freq in bigram_freq.items():
    bigram_prob[(w1, w2)] = (freq+1) / (unigram_freq[w1] + vocab_size)  # conditional

# Function to generate text 
def generate_text(start_word, num_words):
    current_word = start_word
    sentence = [current_word]

    for _ in range(num_words):
        # bigram candidates
        next_word_candidates = [w2 for (w1, w2) in bigram_prob if w1 == current_word]
        if not next_word_candidates:
            break
    
        next_word_probs = [bigram_prob[(current_word, w2)] for w2 in next_word_candidates]

        # choose next word with high probability {main principle}
        next_word = random.choices(next_word_candidates, weights=next_word_probs, k=1)

        sentence.append(next_word[0])
        current_word = next_word
    
    return ' '.join(sentence)

print(generate_text("i", 5))

"""
tokenized_sentences: [['i', 'love', 'coding'], ['i', 'love', 'programming'], ['i', 'enjoy', 'learning']]
bigrams: [('i', 'love'), ('love', 'coding'), ('i', 'love'), ('love', 'programming'), ('i', 'enjoy'), ('enjoy', 'learning')]

bigrams_freq: Counter({('i', 'love'): 2, ('love', 'coding'): 1, ('love', 'programming'): 1, ('i', 'enjoy'): 1, ('enjoy', 'learning'): 1})

next_word_candidates and next_word_probabilities for "i" as current word:
['love', 'enjoy'] [0.3333333333333333, 0.2222222222222222]

sentence with "i" as initial word: i love
"""