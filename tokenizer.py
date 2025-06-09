
with open("as_a_man_thinketh.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))
print(raw_text[:99])

import re
text = "This is an example text."
result = re.split(r'([.,]|\s)', text)
result = [item for item in result if item.strip()]
print(result)

preprocessed = re.split(r'([.,;:!@#$%^&*"\'()-]|\s|--)', raw_text)
preprocessed = [item for item in preprocessed if item.strip()]
print(preprocessed[:30])

print(len(preprocessed))

# create list of all unique tokens and sort them alphabetically to determine to vocabulary size
# each unique token is mapped to a unique integer called token ID

all_words = sorted(set(preprocessed))
print(len(all_words))

# vocabulary is a dictionary of tokens and associated token IDs
# enumerate assigns integer to each word in alphabetical order

vocabulary = {token: integer for integer, token in enumerate(all_words)}
# print(vocabulary)

# vocabulary.items() retrieves the key-value pairs in the dictionary

for i, item in enumerate(vocabulary.items()):
  print(item)
  if (i >= 100):
    break

# token to token ID -> encoding
# decoder also needed to get back token given a token ID
# LLM will output token ID -> we need to get the corresponding word
# vocabulary is a mapping from token to token ID
# we also need a reverse mapping (decoder)

# we build a tokenizer class, it has 2 methods: encode and decode
# encode : takes token as input and gives ID as output
# decode : token ID as input and token as output

class SimpleTokenizerV1:
  def __init__(self, vocabulary):
    self.str_to_int = vocabulary
    self.int_to_str = {i:s for s, i in vocabulary.items()}

  def encode(self, text):
    preprocessed = re.split(r'([.,;:!@#$%^&*"\'()-]|\s|--)', text)
    # if you want to remove empty strings but keep original whitespace within strings
    # preprocessed = [item for item in preprocessed if item.strip()]

    # the following line removes empty strings and trims spaces from all items
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    ids = [self.str_to_int[s] for s in preprocessed]
    return ids

  def decode(self, ids):
    text = " ".join([self.int_to_str[i] for i in ids])
    text = re.sub(r'\s+([.,;?!:()"\'])', r'\1', text)
    return text

  # \s+ → Matches one or more spaces.
  # ([.,;?!:()"\']) → Captures a punctuation mark.
  # r'\1' → Replaces the matched space and punctuation with just the punctuation.

# instantiating a new tokenizer object from the SimpleTokenizerV1 class
tokenizer = SimpleTokenizerV1(vocabulary)
text = "Tempest-tossed souls, wherever ye may be, under whatsoever conditions ye may live, know this in the ocean of life the isles of Blessedness are smiling, and the sunny shore of your ideal awaits your coming. Keep your hand firmly upon the helm of thought. In the bark of your soul reclines the commanding Master; He does but sleep: wake Him. Self-control is strength; Right Thought is mastery; Calmness is power."
ids = tokenizer.encode(text)
print(ids)

tokenizer.decode(ids)

# what if some words we want to encode are not present in the vocabulary

# sample_text = "Cloves are spices"
# tokenizer.encode(sample_text)
# KeyError: 'Cloves'

# we get an error; this highlights the need to work with large and diverse vocabulary when working with LLMs
# we thus modify the tokenizer to handle unknown words

# we thus make a class SimpleTokenizerV2
# special features in this version of tokenizer is <|unk|> and <|endoftext|>
# we can modify the tokenizer to use an <|unk|> token if it encounters a word which is not in the vocabulary
# furthermore, we can add a tokne between unrelated texts; for example, when training GPT-like LLMs on multiple independent documents, it is common to insert a token before each document or book that follows a previous text source

# we add <|unk|> and <|endoftext|> to the list of unique tokens
# preprocessed was array of tokens from raw text

all_words = sorted(list(set(preprocessed)))
#all_words.append("<|unk|>")
#all_words.append("<|endoftext|>")
all_words.extend(["<|unk|>",  "<|endoftext|>"])

vocabulary = {token: integer for integer, token in enumerate(all_words)}
print(len(vocabulary))

for i, item in enumerate(list(vocabulary.items())[-5:]):
  print(item)

class SimpleTokenizerV2:
  def __init__(self, vocabulary):
    self.str_to_int = vocabulary
    self.int_to_str = {i:s for s, i in vocabulary.items()}

  def encode(self, text):
    preprocessed = re.split(r'([!@#$%^&*:;\',."()]|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    preprocessed = [
        item if item in vocabulary
        else "<|unk|>" for item in preprocessed
    ]
    ids = [self.str_to_int[s] for s in preprocessed]
    return ids

  def decode(self, ids):
    text = " ".join([self.int_to_str[i] for i in ids])
    text = re.sub(r'\s+([!@#$%^&*:;\',."()])', r'\1', text)
    return text

# use of SimpleTokenizerV2

tokenizer = SimpleTokenizerV2(vocabulary)
text1 = "Only he whose thoughts are controlled and purified, makes the winds and the storms of the soul obey him."
text2 = "This is sample text"
text = " <|endoftext|> ".join((text1, text2))
print(text)

print(tokenizer.encode(text))

print(tokenizer.decode(tokenizer.encode(text)))

'''So far, we have discussed tokenization as an essential step in processing text as input to
LLMs. Depending on the LLM, some researchers also consider additional special tokens such
as the following:

[BOS] (beginning of sequence): This token marks the start of a text. It
signifies to the LLM where a piece of content begins.

[EOS] (end of sequence): This token is positioned at the end of a text,
and is especially useful when concatenating multiple unrelated texts,
similar to <|endoftext|>. For instance, when combining two different
Wikipedia articles or books, the [EOS] token indicates where one article
ends and the next one begins.

[PAD] (padding): When training LLMs with batch sizes larger than one,
the batch might contain texts of varying lengths. To ensure all texts have
the same length, the shorter texts are extended or "padded" using the
[PAD] token, up to the length of the longest text in the batch.

Note that the tokenizer used for GPT models does not need any of these tokens mentioned
above but only uses an <|endoftext|> token for simplicity

the tokenizer used for GPT models also doesn't use an <|unk|> token for outof-vocabulary words. Instead, GPT models use a byte pair encoding tokenizer, which breaks
down words into subword units
'''

# BYTE PAIR ENCODING
!pip3 install tiktoken

import importlib
import tiktoken

print("tiktoken version:", importlib.metadata.version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)
# note in the output "<|endoftext|>" is assigned token ID 50256; there are 50257 tokens in ChatGPT (BPE tokenizer was used for GPT2 and GPT3); "<|endoftext|>" is the last token

strings = tokenizer.decode(integers)

print(strings)

# example to illustrate how BPE tokenizer deals with unknown words
integers = tokenizer.encode("Akwirw ier")
print(integers)

strings = tokenizer.decode(integers)
print(strings)
