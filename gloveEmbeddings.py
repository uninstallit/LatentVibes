import string
import numpy as np
import tensorflow as tf
from keras import layers
from vibesConfig import (
    directories,
    path_to_glove_file,
    vocab_size,
    latent_dim,
    maxlen,
    UNK_TOKEN,
)

import os

filenames = []
for dir in directories:
    for f in os.listdir(dir):
        filenames.append(os.path.join(dir, f))

text_ds = tf.data.TextLineDataset(filenames)
sentences = [line.numpy().decode("utf-8") for line in text_ds]


def custom_standardization(input_string):
    """Remove html line-break tags and handle punctuation"""
    lowercased = tf.strings.lower(input_string)
    stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
    return tf.strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")


def prepare_lm_tokens(text):
    tokenized_sentences = vectorize_layer(text)
    return tokenized_sentences


vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=maxlen,
)
vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()


word_to_index = {}
for index, word in enumerate(vocab):
    word_to_index[word] = index

word_index = dict(zip(vocab, range(len(vocab))))

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

num_tokens = len(vocab)
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, latent_dim))

for word, i in word_index.items():
    if word == UNK_TOKEN:
        embedding_matrix[i] = np.random.uniform(-1.0, 1.0, latent_dim)
        misses += 1
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            embedding_matrix[i] = np.random.uniform(-1.0, 1.0, latent_dim)
            misses += 1  # OOV words other than [UNK]

print("\nConverted %d words (%d misses)" % (hits, misses))

embedding_layer = layers.Embedding(
    num_tokens, latent_dim, trainable=True, weights=embedding_matrix, name="embedding"
)

# print("initial embe matrix 0: ", embedding_matrix[0])
# print("initial embe matrix 0: ", embedding_matrix[1])

# embedding_layer.build((1,))
# embedding_layer.set_weights([embedding_matrix])
