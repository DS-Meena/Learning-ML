## Building a LSTM model to generate poems

### 1. Finding Data 📚

To create a good model for generating love 💖 poems, we'll need a large dataset of poetry, preferably with a focus on love themes. 
I used [Poems from poetryfoundation.org](https://www.kaggle.com/datasets/ultrajack/modern-renaissance-poetry) dataset from kaggle.

Here are some other sources you can consider:

- Project Gutenberg (https://www.gutenberg.org/): A large collection of free e-books, including poetry collections.
- PoetryFoundation (https://www.poetryfoundation.org/): Offers a vast collection of poems, including many about love.
- Kaggle Datasets: Search for "poetry" or "love poems" on Kaggle for pre-compiled datasets.
- Reddit r/OCPoetry: A subreddit where users share original poetry, which you could scrape (following Reddit's API rules).

### 2. Preprocessing the Data

Before training, we'll need to preprocess your data:

1. Clean the text by removing any irrelevant information or formatting. 🧹
2. Tokenize the poems into words or subwords.
3. Create sequences of fixed length for training.
4. Encode the sequences into numerical format. 

You can try keeping the formatting, so that model generate poems with formatting.

### 3. Building the Model 🏗️

For this task, I created a LSTM based language model using tensorflow and keras.

### 4. Training the Model 💪

After compiling the model, train it on the poems dataset. Adjust the number of epochs and batch size as needed. You can also use early stopping and model checkpoints to prevent overfitting and save the best model.

### 5. Generating Poems 

After training, you can generate new poems by:

1. Starting with a seed text. 💬
2. Predicting the next word using the trained model.
3. Appending the predicted word to the seed text.
4. Repeating the process until you reach the desired length.

## How to Improve

- In future, we can try a larger dataset for better quality and diversity in generated poems.
- We can experiment with the model architecture, such as adding more LSTM layers or using bidirectional LSTMs.
- Fine-tune hyperparameters like learning rate, batch size, and sequence length.

In this model, I have not tried any fine-tuning because my aim was just to study different language models.

## Role of Each file

**preprocesssing.py**
- Reads the data as a list.

- Cleans the text by removing extra whitespace (also removes \n and \r from string) and converting to lowercase
- Tokenizes the poems using Keras' Tokenizer
- Creates input sequences for training
- Pads the sequences to ensure uniform length
- Prepares the data (X and y) for training a text generation model

We can later try to keep the formatting and try the model on data with \n and \r.

**text_generation.py**

This file has the `generate_poem` method, that generates a poem given a seed text and the number of words to generate.
It then iteratively predicts the next word based on the current sequence and appends it to the generated text.

Generated poem with trained model -> 
'Love is a sickness full of woes a race the colour that the size all which the prease of the purest sky for this a wishfull vow of the ground beneath her eyelids she or are times lord the world subdue both that that water with her eyes the fyre of woe'

**model.py**

In this, I create a sequential model with embedding and LSTM layers, which is a common architecture for text generation tasks.

**training.py**

This file has the code to train the model. I trained for 50 epochs and used 20% data for validation.

## Kaggle Notebook 📒

You can also check the corresponding notebook at kaggle: [Love Poems using LSTM](https://www.kaggle.com/code/dsmeena/love-poems-using-lstm/notebook)

