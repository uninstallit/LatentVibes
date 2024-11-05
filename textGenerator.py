import keras
import numpy as np
import tensorflow as tf
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


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

    def plot_embeddings(self, true_emb, pred_emb, batch=None):
        true_emb = true_emb.numpy() if hasattr(true_emb, "numpy") else true_emb
        pred_emb = pred_emb.numpy() if hasattr(pred_emb, "numpy") else pred_emb

        # Extract x and y coordinates
        pred_x = pred_emb[:, 0]
        pred_y = pred_emb[:, 1]
        true_x = true_emb[0]
        true_y = true_emb[1]

        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.plot(pred_x, pred_y, marker="o", label="Predicted Trajectory")
        plt.scatter(true_x, true_y, color="red", label="True Embedding", zorder=5)

        # Annotate the start and end points
        plt.scatter(pred_x[0], pred_y[0], color="green", label="Start Point", zorder=5)
        plt.scatter(pred_x[-1], pred_y[-1], color="blue", label="End Point", zorder=5)

        # Add title and labels
        title = "Embedding Trajectory"

        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(f"images/embedding_trajectory_{batch}.png")
        plt.close()

    def get_updated_embedding_matrix(self):
        embedding_layer = self.model.word_encoder.get_layer("embedding")
        embedding_matrix = embedding_layer.get_weights()[0]

        #  (105333, 100)
        # print(" embedding_matrix: ", embedding_matrix)
        ukn_embeddings = embedding_matrix[1, :]

        # embedding_0 = embedding_matrix[0, :]
        # embedding_1 = embedding_matrix[1, :]

        # print("0: ", embedding_0)
        # print("1: ", embedding_1)
        # print("\n\n")

        return embedding_matrix, ukn_embeddings

    def find_closest_words_euclidean(self, new_embeddings):
        # shape of ukn_embeddings = (100, )
        embedding_matrix, ukn_embeddings = self.get_updated_embedding_matrix()

        # print("embedding_matrix: ", embedding_matrix)

        # distance = np.linalg.norm(embedding_matrix - ukn_embeddings, axis=1)
        # closest_index = np.argmin(distance)
        # closest_word = self.vocab[closest_index]

        # print("\n\n *** closest_index: ", closest_index)
        # print("*** losest_word_to_ukn: ", closest_word)
        # print("*** vocab top 3: ", self.vocab[:3])
        # print("\n\n")

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

            # true_emb = self.model.true_emb.numpy()
            # pred_emb = self.model.pred_emb.numpy()

            # print("\n\n >>> starting coordinate: ", pred_emb[0, :])
            # print("\n\n >>> pred_emb coordinate: ", pred_emb)

            # print("true_emb: ", true_emb.numpy())
            # print("pred_emb: ", pred_emb.numpy())
            # exit()
            # self.plot_embeddings(true_emb, pred_emb, batch=self.batch_count)
