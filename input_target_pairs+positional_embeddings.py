
!pip3 install tiktoken

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

# we will create a data loader which will fetch input-target pairs using sliding window approach

# first we tokenize the input text using the BPE tokenizer
with open("as_a_man_thinketh.txt", "r", encoding="utf-8") as f:
  raw_text = f.read()

encoded_text = tokenizer.encode(raw_text)
print(len(encoded_text))
# this returns 10654 which is the total number of tokens after encocding the raw text

# print(encoded_text)

# we remove first 50 tokens from the dataset (only for demonstration purposes)
# we can also proceed by skipping this step

encoded_sample = encoded_text[50:]

# easiest way to create input-target pairs is to create 2 variabbles x and y
# x contains input tokens and y contains the targets, which are the input shifted by 1
# example, if x is [1, 2, 3, 4] then y will be [2, 3, 4, 5]
# so if 1 is the input then 2 will be the output
# if [1, 2] is the input then 3 will be the output
# if [1, 2, 3] is the input then 4 will be the output
# if [1, 2, 3, 4] is the input then 5 will be the output

# context size - how many words given as input for the model to make prediction

context_size = 4 #length of the input
# The context_size of 4 means that the model is trained to look at a sequence of 4 words (or tokens) to predict the next word in the sequence.
# The input x is the first 4 tokens [1, 2, 3, 4], and the target y is the next 4 tokens [2, 3, 4, 5]

x = encoded_sample[:context_size]
y = encoded_sample[1:context_size+1]

print(f"x: {x}")
print(f"y:      {y}")

for i in range(1, context_size+1):
    context = encoded_sample[:i]
    desired = encoded_sample[i]

    print(context, "---->", desired)

# everything printed on left of the arrow (---->) refers to the input an LLM would receive, and the token ID on the right side of the arrow represents the target token ID that the LLM is supposed to predict.
# since context size is 4, there are 4 prediction tasks happening here

# for illustration purposes, let's repeat the previous code but convert the token IDs into text
for i in range(1, context_size+1):
    context = encoded_sample[:i]
    desired = encoded_sample[i]

    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

# we will implement an efficient data loader which will iterate over the input dataset
# this data loader will return the inputs and targets as PyTorch tensors

# we will return 2 tensors: 1. input tensor (the text that LLM sees) 2. target tensor (includes the targets for LLMs to predict)

# to implement efficient data loaders we collect inputs in a tensor x where each row represents one input context
# the second tensor y contains the corresponding predictions
# note : in every tensor pair (input-target pair) there will be context length number of prediction targets (next words)
# these prediction targets are created by shifting the input by 1 position

# step 1: tokenize the entire text
# step 2: use sliding window to chunk the book into overlapping sequences of max_length
# step 3: return the total number of rows in the dataset
# step 4: return a sinlge row from the dataset

from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
  def __init__(self, txt, tokenizer, max_length, stride):
    self.input_ids = []
    self.target_ids = []

    # tokenizing the entire dataset
    token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

    # using sliding window to chunk the book into overlapping sequences of max_length
    # here stride determines how much to slide to create the next input-target pair
    for i in range(0, len(token_ids) - max_length, stride):
      input_chunk = token_ids[i:i+max_length]
      target_chunk = token_ids[i+1:i+max_length+1]
      self.input_ids.append(torch.tensor(input_chunk))
      self.target_ids.append(torch.tensor(target_chunk))

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.target_ids[idx]


# GPTDatasetV1 class is based on the Dataset class in PyTorch
# it defines how individual rows are fetched from the dataset
# each row consists of a number of token IDs (based on the max_length) assigned to an input_chunk tensor
# target_chunk tensor contains the corresponding targets

# the following code will use GPTDatasetV1 to load the inputs in batches to the PyTorch DataLoader

# steps:
# step 1: initialize the tokenizer (because we need to pass it as an input when calling an instance of GPTDatasetV1)
# step 2: create the dataset
# step 3: drop_last=True drops the last batch if it is shorter than the specified batch_size to prevent loss spikes during training
# step 4: the number of CPU processes to use for processing

def create_dataloader_v1(txt, batch_size=4, max_length = 256, stride=128, shuffle=True, drop_last=True, num_workers=0):
  # initialize the tokenizer
  tokenizer = tiktoken.get_encoding("gpt2")

  # create the dataset
  dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

  return dataloader

# DataLoader accesses the getitem function (defined in the GPTDatasetV1) and get the input-target tensors
# it enables parallel processing and analyse multiple batches at one time
# batch_size is number of parameters analysed before updating parameters
# num_workers is for parallel processing on different heads of the CPU

# we already have raw_text defined
# so we proceed with the dataloader
# stride dictates the number of positions which the input shifts across batches emulating a sliding window approach (basically how much the sliding window slides for each input tensor)
import torch
print("PyTorch version: ", torch.__version__)
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
# first_batch variable contains 2 token IDs --> first tensor contains the input token IDs and the second tensor contains the output token IDs
# small batch sizes require less memory but lead to noisy model updates
# batch size is a trade-off and hyperparameter to experiment when working with LLMs
second_batch = next(data_iter)
print(second_batch)

# effect of large batch size
# in the above example, batch_size is 1 so when we print first_batch it contains input tensor with only one array and target tensor with only one array
# which means 4 predictions
# now we increase batch_size to 8 which means we have input tensor with 8 arrays and target tensor with 8 arrays
# so we have 4 * 8 predictions (since context length here is 4)

# the model will process this batch before making the parameter updates
# we increase the stride to 4 to utilize the data fully and avoid overlapping because more overlapping leads to overfitting
# thus usually stride length is kept equal to the context length
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
# batch_size 8 means 8 input-target pairs will be updated before making paramter updates
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
input, targets = next(data_iter)
print(input)
print(targets)

# we will convert each of these token IDs into a 256 dimension vector
# that will be the vector embedding
vocab_size = 50257 # we are using the gpt2 tokenization scheme, hence these many tokens
output_dim = 256 # gpt2 had 768 vector dimension
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# using the token_embedding_layer, if we sample from the dataloader we embed each token in each batch
# batch size of 8 with 4 tokens each, thus we have 8x4x256 tensor

token_embeddings = token_embedding_layer(input)
print(token_embeddings.shape)

# we need to create another embedding layer for the postional embeddings
# at each time, one 4 input tokens are processed to determine the target token
# so we need to encode only 4 positions in this case

# in case of token embeddings the number of rows was the vocab size
# in this case the number of rows will be 4 (only 4 input tokens processed at once)
# and number of cols will be 256 (because we need to add positional embedding vectors to token embeddings)
# here context length is 4

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# for each input the set of positional embeddings to be added are the same 4 vectors we geenrate
# they do not change because we just need to know in a particular input a token is first, second, third or fourth
# we will add the same 4 vecs to all input sequences for all batches

# we will thus generate 4 positional embedding vectors from the postional embedding matrix

pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)

'''
As shown in the preceding code example, the input to the pos_embeddings is usually a
placeholder vector torch.arange(context_length), which contains a sequence of
numbers 0, 1, ..., up to the maximum input length − 1.

The context_length is a variable
that represents the supported input size of the LLM.

Here, we choose it similar to the
maximum length of the input text.

In practice, input text can be longer than the supported
context length, in which case we have to truncate the text.
'''

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
