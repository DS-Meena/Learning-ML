from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = TFGPT2LMHeadModel.from_pretrained(model_name)

# Set up tokenizer with padding
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Update model config
model.config.pad_token_id = tokenizer.pad_token_id