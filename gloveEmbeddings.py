import numpy as np
import tensorflow as tf
from keras import layers
import string

import os

filenames = []
directories = [
    "aclImdb/train/pos",
    "aclImdb/train/neg",
    # "aclImdb/test/pos",
    # "aclImdb/test/neg",
]

for dir in directories:
    for f in os.listdir(dir):
        filenames.append(os.path.join(dir, f))

text_ds = tf.data.TextLineDataset(filenames)
sentences = [line.numpy().decode("utf-8") for line in text_ds]

vocab_size = 20000  # Only consider the top 20k words
maxlen = 80  # Max sequence size


def custom_standardization(input_string):
    """Remove html line-break tags and handle punctuation"""
    lowercased = tf.strings.lower(input_string)
    stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
    return tf.strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")


def prepare_lm_tokens(text):
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:-1]
    return x


def prepare_lm_tokens_words(text_batch):
    tokenized_sentences = vectorize_layer(text_batch)
    tokens = tokenized_sentences[:, :-1]

    words = []
    for token_sequence in tokens:
        # Assuming vocab is a dictionary mapping token IDs to words
        word_sequence = [vocab[token_id.numpy()] for token_id in token_sequence]
        words.append(word_sequence)

    return tokens, words


vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size - 1,
    output_mode="int",
    output_sequence_length=maxlen + 1,
)
vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()  # To get words back from token indices
word_index = dict(zip(vocab, range(len(vocab))))

tokens, words = prepare_lm_tokens_words(sentences)

path_to_glove_file = "glove.6B/glove.6B.100d.txt"

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))


num_tokens = len(vocab) + 2
embedding_dim = 100
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

# needs to be implemented
# def get_embedding_matrix()

embedding_layer = layers.Embedding(
    num_tokens, embedding_dim, trainable=True, name="embedding"
)
embedding_layer.build((1,))
embedding_layer.set_weights([embedding_matrix])
