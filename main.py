import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import string
import tensorflow as tf
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings

import keras
from keras import layers
from vibesDry import word_encoder, score_model, Vibes
from gloveEmbeddings import embedding_matrix

import numpy as np
from numpy.linalg import norm

import textwrap

# The dataset contains each review in a separate text file
# The text files are present in four different folders
# Create a list all files
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

epochs = 100
batch_size = 24
text_ds = tf_data.TextLineDataset(filenames)

sentences = [line.numpy().decode("utf-8") for line in text_ds]


norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
norms[norms == 0] = 1

embedding_matrix_norm = embedding_matrix / norms


def custom_standardization(input_string):
    """Remove html line-break tags and handle punctuation"""
    lowercased = tf_strings.lower(input_string)
    stripped_html = tf_strings.regex_replace(lowercased, "<br />", " ")
    return tf_strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")


vocab_size = 20000  # Only consider the top 20k words
maxlen = 80  # Max sequence size

# Create a vectorization layer and adapt it to the text
vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size - 1,
    output_mode="int",
    output_sequence_length=maxlen + 1,
)
vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()  # To get words back from token indices
# print(vocab)


def prepare_lm_inputs_labels(text):
    """
    Shift word sequences by 1 position so that the target for position (i) is
    word at position (i+1). The model will use all words up till position (i)
    to predict the next word.
    """
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y


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


# Apply the function to your sentences
normalized_tokens, words = prepare_lm_tokens_words(sentences)

text_ds_x_only = text_ds.map(prepare_lm_tokens, num_parallel_calls=tf_data.AUTOTUNE)
text_ds_x_only = text_ds_x_only.shuffle(buffer_size=256)
text_ds_x_only = text_ds_x_only.batch(batch_size, drop_remainder=True)
text_ds_x_only = text_ds_x_only.prefetch(tf_data.AUTOTUNE)

vae = Vibes(word_encoder, score_model, batch_size=batch_size, maxlen=80, dt=0.01)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))


class TextGenerator(keras.callbacks.Callback):
    def __init__(self, start_tokens, max_tokens, vocab):
        self.start_tokens = start_tokens
        self.max_tokens = max_tokens
        self.vocab = vocab
        self.word_index = {word: idx for idx, word in enumerate(vocab)}
        self.pad_token = 0
        self.unk_token = 1
        self.batch_count = 0

    def get_updated_embedding_matrix(self):
        embedding_layer = self.model.word_encoder.get_layer("embedding")
        embedding_matrix = embedding_layer.get_weights()

        embedding_matrix = embedding_layer.get_weights()[0]
        return embedding_matrix

    def find_closest_words_euclidean(self, new_embeddings):
        embedding_matrix = self.get_updated_embedding_matrix()

        distances = np.linalg.norm(
            new_embeddings[:, np.newaxis] - embedding_matrix, axis=2
        )
        closest_indices = np.argmin(distances, axis=1)
        closest_words = [self.vocab[idx] for idx in closest_indices]

        distance_scores = distances[np.arange(len(new_embeddings)), closest_indices]
        return closest_words, distance_scores

    def on_batch_end(self, batch, logs=None):
        self.batch_count += 1

        if (self.batch_count) % 50 == 0:
            start_index = len(self.start_tokens)

            # Step 1: Start with the initial tokens
            generated_tokens = self.start_tokens.copy()

            # Step 2: Pad the tokens to max_tokens if necessary
            if len(generated_tokens) < self.max_tokens:
                padding_length = self.max_tokens - len(generated_tokens)
                generated_tokens += [self.pad_token] * padding_length
            else:
                generated_tokens = generated_tokens[: self.max_tokens]

            # Step 3: Generate embeddings for the input tokens
            input_tokens = tf.convert_to_tensor([generated_tokens], dtype=tf.int32)
            embeddings = self.model.word_encoder.predict(input_tokens, verbose=0)
            embeddings_tr = tf.transpose(embeddings, perm=[1, 0, 2])

            # Step 4: Perform diffusion to generate new embeddings
            sequence = self.model.diffusion_generate(
                embeddings_tr, batch_size=1, start_index=start_index
            )
            sequence = tf.squeeze(sequence, axis=0).numpy()

            # Step 5: Find the closest words to the generated embeddings
            closest_words, similarity_scores = self.find_closest_words_euclidean(
                sequence
            )

            # Step 6: Construct and display the generated text
            generated_text = " ".join(closest_words)
            wrapped_text = textwrap.fill(generated_text, width=70)
            border = "*" * 80
            print("\n" + border)
            print(f"\nGenerated Text (Batch #{self.batch_count}):")
            print(f"\n- {wrapped_text}")
            print("\n" + border + "\n")


# Tokenize starting prompt
word_to_index = {}
for index, word in enumerate(vocab):
    word_to_index[word] = index

max_tokens = 80
start_prompt = "this movie is"
start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]

text_gen_callback = TextGenerator(start_tokens, max_tokens, vocab)
# text_gen_callback.on_batch_end(batch=1)

vae.fit(
    text_ds_x_only, epochs=epochs, batch_size=batch_size, callbacks=[text_gen_callback]
)
