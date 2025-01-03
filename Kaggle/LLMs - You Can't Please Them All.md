# LLMs - You Can't Please Them All

## Trying to Generate Essay using GPT2

Below implementation requires internet, so I could not submit in competition.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class EssayGenerator:
    def __init__(self):
        # Load pre-trained model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def generate_essay(self, prompt, max_length=500, temperature=0.7):
        # Encode the input prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        # Generate text
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Decode and return the generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

generator = EssayGenerator()

prompt = "write an essay about how to fix crime?"
essay = generator.generate_essay(prompt)
print(essay)
```

Output was something like this:
```
It's not just about the crime, but about social issues, too. He said that he wants to be a political force for change in Canada. Not just in the media but in politics too, he said. They can say things about him that are un-Canadian. People can talk about his background or his family or the history of his country. It's all about what's right now. And it's about making things better for the whole country, for everyone, and for this country so that it can continue to grow."
```

## How to use Offline Model 📴

1. Search for correct model in kaggle.
2. Download it.
3. Load it and use it.

Downloading GPT2 model in input then using it worked.

```python
self.tokenizer = GPT2Tokenizer.from_pretrained("/kaggle/input/text-generation-gpt2-marketing/transformers/default/1/gpt2-finetuned")
self.model = GPT2LMHeadModel.from_pretrained("/kaggle/input/text-generation-gpt2-marketing/transformers/default/1/gpt2-finetuned")
```