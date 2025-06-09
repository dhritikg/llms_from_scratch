
import torch

# we assume vocab size as 6 and dimension of vector embeddings as 3
# gt2 has 50257 tokens and vector embedding size is 768
# gpt3 has embedding size of 12288 dimensions
# we will create the embedding layer matrix

input_ids = torch.tensor([2, 3, 5, 1])

vocab_size = 6
output_dim = 3
torch.manual_seed(123)

embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# this initializes the wieghts of the embedding matrix in a random manner
print(embedding_layer.weight)
# every row corresponds to vector associated with the token ID

# the weight matrix of the embedding layer consists of small random values initially
# these values are optimised during LLM training as a part of the LLM optimization itself
# LLM optimization --> 1. embedding layer weights 2. weights for prediction of the next word

print(embedding_layer(torch.tensor([3])))
# the embedding layer is essentially a look-up operation which retrieves rows from the embedding layer matrix via the token IDs
print(embedding_layer(input_ids))
