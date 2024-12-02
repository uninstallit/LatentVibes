directories = [
    "aclImdb/train/pos",
    "aclImdb/train/neg",
    # "aclImdb/test/pos",
    # "aclImdb/test/neg",
]

path_to_glove_file = "glove.6B/glove.6B.100d.txt"

# special tokens
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"

vocab_size = 400000  # Only consider the top k words
latent_dim = 100  # Embedding dimensions
maxlen = 25  # Max sequence length

# diffusion
dt = 0.05

# training
epochs = 1000
batch_size = 64
learning_rate = 0.001

# Transformer parameters
num_transformer_blocks = 4
head_size = 64
num_heads = 4
ff_dim = 128
dropout = 0.1

# inference
start_prompt = "this movie is"
print_every_batch = 500