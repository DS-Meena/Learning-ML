# Using pre-trained Lanuage Model for Love poem Generation

We can leavage pre-trained LM to gnerate love poems through fine-tuning. This involves following steps:
1. Choose a Pre-trained model: Select a suitable pre-trained model like GPT-2 or BERT. These models have been trained on vast amounts of text data and can generate coherent language.
2. Prepare your dataset
3. Fine-tune the model
4. Generate Poems

You can check the kaggle notebook here: [Kaggle notebok](https://www.kaggle.com/code/dsmeena/love-poems-using-pre-trained-lm/notebook#Generate-a-love-poem)

# Steps

## Pretrained.py 

Loads a pre-trained GPT-2 model and tokenizer using transformers.

## Preprocessing.py

In this step, we prepare our dataset of love poems for fine-tuning.

## training.py

Compiles and fine-tunes the model on the love poem dataset. 

## Text Generation.py 

Provides a function to generate new love poems using the fine-tuned model.

# Problem

This time, I couldn't get the model working as wanted. In current version of code, I am getting nan loss and "!" character only in text generation.

## Nan loss

Nan (Not a Number) loss typically indicates that the model's paramters have become unstable ⚖️ during training.

Usually this happens for following reasons:
- Learning rate too high: I tried reducing it from 5e-5 to 1e-5.
- Exploding gradients: We use gradient clipping to prevent this issue.
- Numerically instability: We need to ensure input data is properly normalized.

## Generated text contains only exclamation ‼️ marks

This issue directly suggests that the model hasn't learned meaningful patterns.

Following reasons could be behind this:
- Insufficient training: Not possible in our case, since getting nan loss.
- Small or low-quality dataset: We can try making the dataset large and ensure that it contains high-quality examples.
- Incorrect data preprocessing: Data preparation steps might contains some wrong encoding.

