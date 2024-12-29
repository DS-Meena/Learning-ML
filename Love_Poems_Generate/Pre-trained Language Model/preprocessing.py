import pandas as pd
import numpy as np
import tensorflow as tf
from pretrained import tokenizer, model
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset

# Reading the Data 
my_data = pd.read_csv("all.csv")
poems = my_data.loc[my_data['type'] == 'Love']['content']
poems = list(poems)

# clean and preprocess the text
def preprocess_text(text):
    # remove extra whitespaces and convert to lowercase
    text = ' '.join(text.split()).lower()
    return text

cleaned_poems = [preprocess_text(poem) for poem in poems]

# Prepare the dataset
def load_dataset(poems_list, tokenizer):
    dataset = Dataset.from_dict({"text": poems_list})
#     dataset = dataset.map(lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128), batched=True)
    
    def tokenize_and_prepare(examples):
        tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128, return_tensors="tf")
        input_ids = tf.convert_to_tensor(tokenized["input_ids"])
        labels = tf.identity(input_ids)  # Create a copy of input_ids
        
        tokenized["labels"] = labels
        return tokenized
    
    dataset = dataset.map(tokenize_and_prepare, batched=True, remove_columns=dataset.column_names)
    return dataset

# use pad function to prepare dataset
train_dataset = load_dataset(cleaned_poems, tokenizer)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Create tensorflow dataset
tf_train_dataset = model.prepare_tf_dataset(
    train_dataset,
    shuffle=True,
    batch_size=4,
    collate_fn=data_collator,
)