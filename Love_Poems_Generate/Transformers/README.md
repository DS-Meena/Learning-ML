Let's create a transformer model from scratch using TensorFlow to generate love poems. This process involves four key steps:

1. Data Preparation
2. Model Architecture
3. Training
4. Poem Generation

Check the kaggle notebook here: [Love poems using Transformers](https://www.kaggle.com/code/dsmeena/love-poems-using-transformers/notebook)

# 1. Data Preparation

During data preparation, we need to convert the sentences into sequences of integers. For this purpose, we do encoding of the sentences.

## Word-Level vs Character-Level Encoding in Language Models 🔤

When it comes to encoding text for language models, we have two main approaches: word-level and character-level encoding. Each has its own strengths and use cases. Let's dive into both! 🏊‍♂️

### Word-Level Encoding 📚

Word-level encoding treats each word as a single unit. It's like looking at a sentence as a collection of building blocks, where each block is a complete word. 🧱

- **Advantages 👍**
    - Better at capturing semantic meaning 🧠
    - Typically results in shorter sequences, which can be computationally efficient 🚀
    - Works well for tasks that rely heavily on word meanings, like sentiment analysis or text classification 📊
- **Disadvantages 👎**
    - Limited to a fixed vocabulary, struggles with out-of-vocabulary words 📖
    - Can't handle misspellings or novel words easily ❌
    - Requires more memory to store word embeddings for large vocabularies 💾
- **When to use 🕰️**
    - For tasks where word meaning is crucial (e.g., translation, summarization) 🌐
    - When working with formal or well-structured text 📝
    - For larger datasets with a consistent vocabulary 📊

### Character-Level Encoding 🔤

Character-level encoding breaks down text into individual characters. It's like looking at a sentence as a string of letters, spaces, and punctuation marks. 🔍

- **Advantages 👍**
    - Can handle any word, including misspellings and novel words 🆕
    - Smaller vocabulary size (usually just ASCII characters) 🔡
    - Can learn subword patterns and morphology 🧬
- **Disadvantages 👎**
    - Sequences are much longer, which can be computationally expensive 🐢
    - May struggle to capture higher-level semantic meanings 🤔
    - Training can take longer due to the increased sequence length ⏳
- **When to use 🕰️**
    - For tasks involving noisy text or user-generated content (e.g., tweets, chat logs) 💬
    - When dealing with languages that have complex morphology 🌍
    - For creative text generation tasks, like poetry or song lyrics 🎵

Example:
```
text = """
Roses are red,
Violets are blue,
Sugar is sweet,
And so are you.
"""

Word-level encoding:
[1, 2, 3, 4, 2, 5, 6, 7, 8, 9, 10, 2, 11]
Length of word-level encoding: 13

Character-level encoding:
[1, 2, 3, 4, 3, 5, 6, 1, 4, 5, 1, 4, 5, 7, 8, 2, 9, 4, 10, 3, 5, 6, 1, 4, 5, 11, 9, 12, 4, 5, 13, 12, 14, 6, 1, 5, 8, 3, 5, 3, 15, 4, 4, 10, 5, 16, 17, 5, 3, 2, 5, 6, 1, 4, 5, 18, 2, 12]
Length of character-level encoding: 58
```

For our task, First I tried to use character level encoding, so that it can play with words better. But the sequences length become very large and it was computationally not possible to train model with such large sequences [120 GB was needed]. So, I had to settle with word level encoding. 🚀✨



input sequences:
```
[[141, 1000],
 [141, 1000, 28],
 [141, 1000, 28, 865],
 [141, 1000, 28, 865, 63],
 [141, 1000, 28, 865, 63, 9],
 [141, 1000, 28, 865, 63, 9, 676],
 [141, 1000, 28, 865, 63, 9, 676, 112],
 [141, 1000, 28, 865, 63, 9, 676, 112, 1],
 [141, 1000, 28, 865, 63, 9, 676, 112, 1, 76],
 [141, 1000, 28, 865, 63, 9, 676, 112, 1, 76, 15]]
```
Padded sequences
```
array([[   0,    0,    0, ...,    0,  141, 1000],
       [   0,    0,    0, ...,  141, 1000,   28],
       [   0,    0,    0, ..., 1000,   28,  865]], dtype=int32)
```


# 2. Model Architecture

Our model uses the transformer block to generate text. Transformer has 2 components, first is a encoder which encodes the input sequences and uses attention mechanism to focus on the important parts of the sequences. Second, is the decoder which uses the output of encoder and generates output sequence.

Our model returns logits not probabilities.

## Logits
Logits are an important concept in machine learning, particularly in neural networks and classification tasks. Here's a brief explanation of logits:

- **Definition:** Logits are the raw output values of a neural network's last layer before applying an activation function (like softmax). 🐯
- **Interpretation:** They represent the unnormalized predictions or "scores" for each class in a classification problem. 💯
- **Usage:** Logits are often used as input to loss functions (e.g., cross-entropy loss) during training. 📉
- **Conversion to probabilities:** Applying a softmax function to logits converts them into probabilities that sum to 1. 🟰

In the context of language models: 🗣️

- Logits represent the model's raw predictions for each token in the vocabulary.
- The higher the logit value for a token, the more likely the model thinks that token should be the next word in the sequence.


# 3. Training

I trained the model for 50 epochs.

**Predictor and labels:**

We've adjusted the training data preparation to use the entire sequence for X and y, shifting y by one position.

```
(array([  0,   0,   0, ...,   0,   0, 141], dtype=int32),
 array([   0,    0,    0, ...,    0,  141, 1000], dtype=int32))
```

# 4. Poem Generation

This generate_poem function does the following:

- It takes the model, tokenizer, a start string, number of words to generate, and temperature as inputs.
- It tokenizes the start string and uses it as the initial input to the model.
- It then enters a loop, generating one word at a time: ➿
- The model predicts the next word based on the current input.
- The prediction is adjusted by the temperature parameter to control randomness. 🌡️
- A word is selected based on the adjusted probabilities.
- The selected word is added to the generated text and used as the next input.
- Finally, it returns the complete generated poem.

Remember to adjust the temperature to control the creativity of the output. Lower values 👇 (e.g., 0.5) will make the output more focused and deterministic, while higher values (e.g., 1.0 or above) will make it more diverse and potentially more creative 🔅.

Remember, training a high-quality model requires a large dataset and significant computational resources. This example is simplified for demonstration purposes.