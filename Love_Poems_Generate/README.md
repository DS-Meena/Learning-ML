# Language-Models

Language models are like the brain 🧠 of your favorite apps that predict text or understand your requests. They work by predicting the next word by analyzing the context provided by the preceding words or parts of the text.📊

In this project I have tried to generate Love Poems, by the means of Language models. I have used following language models:
1. **Neural Language Models** - built a LSTM-based Language Model from scratch.
2. **Pre-Trained Language Model** - used transfer learning to train GPT2 on poems dataset.
3. **Transformers** - Used transformer architecture to generate poems.
4. **Statistical Language Models** - Created a n-gram model (specifically bi-gram), to generate a sentence. There is no novelty, next word is predicted based on frequency.

# Results ❤️‍🔥

## Neural Language Models

**Input**: "Love is"

**Output**: "Love is a sickness full of woes a race the colour that the size all which the prease of the purest sky for this a wishfull vow of the ground beneath her eyelids she or are times lord the world subdue both that that water with her eyes the fyre of woe"

## Pre-trained Languaged Models

This should be fixable if you try again.

**Input**: "love is"

**Output**: "love is isisisisesisesizesizesisesiseiseisesizeiseizeizeizesizeizedizedizeizationizationizedizationizeizingizingizeizerizerizersizersizeriseriserisersisersizersisersersersererierierererrerrerrrrererrerrerrersrersrerrrsrsrrdrdrrararasrasrarrasrranranrronronronsronsronranronrryryyyrylylyylyryysysy"

## Transformers

**Input**: "love"

**Output**: "love love love love that on love that sickness brags itself in her name so that these worlds false bonds in"

## Statistical Langauge Models

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