import keras
import numpy as np
import tensorflow as tf
import textwrap


class TextGenerator(keras.callbacks.Callback):
    def __init__(self, start_tokens, max_tokens, vocab, print_every_batch=50):
        self.start_tokens = start_tokens
        self.max_tokens = max_tokens
        self.vocab = vocab
        self.print_every_batch = print_every_batch

        self.word_index = {word: idx for idx, word in enumerate(vocab)}
        self.pad_token = 0
        self.unk_token = 1
        self.batch_count = 0

    def get_updated_embedding_matrix(self):
        embedding_layer = self.model.word_encoder.get_layer("embedding")
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

        if (self.batch_count) % self.print_every_batch == 0:
            start_index = len(self.start_tokens)

            # Step 1: Start with the initial tokens
            generated_tokens = self.start_tokens.copy()

            # Step 2: Truncate tokens if necessary
            if len(generated_tokens) > self.max_tokens:
                generated_tokens = generated_tokens[: self.max_tokens]

            # Step 3: Generate embeddings for the input tokens
            input_tokens = tf.convert_to_tensor([generated_tokens], dtype=tf.int32)
            embeddings = self.model.word_encoder.predict(input_tokens, verbose=0)

            # Step 3: Pad with zeros up to maxlen
            current_len = tf.shape(input_tokens)[1]
            padding_len = self.max_tokens - current_len
            paddings = tf.stack(
                [
                    tf.constant([0, 0], dtype=tf.int32),
                    tf.stack([tf.constant(0, dtype=tf.int32), padding_len]),
                    tf.constant([0, 0], dtype=tf.int32),
                ]
            )
            padded_embeddings = tf.pad(
                embeddings, paddings, mode="CONSTANT", constant_values=0.0
            )
            embeddings_tr = tf.transpose(padded_embeddings, perm=[1, 0, 2])

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
