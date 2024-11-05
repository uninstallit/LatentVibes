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
epochs = 100
batch_size = 24
learning_rate = 0.001

# transformer block
num_heads = 8  # Number of attention heads
ff_dim = 128  # Feed-Forward Network dimension
num_layers = 4  # Number of Transformer Encoder layers

# inference
start_prompt = "this movie is"
print_every_batch = 10
