from pretrained import model, tokenizer

# Generate a love poem
def generate_poem(prompt, model, tokenizer, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="tf")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Generate a poem
prompt = "love is"
poem = generate_poem(prompt, model, tokenizer)
print(poem)