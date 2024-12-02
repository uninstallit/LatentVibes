import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
import tensorflow as tf
from keras import losses
from gloveEmbeddings import embedding_layer

# =========================
# Encoder
# =========================

word_encoder_inputs = keras.Input(shape=(1,), name="word_input")
embedded_sequences = embedding_layer(word_encoder_inputs)
word_encoder = keras.Model(
    inputs=word_encoder_inputs, outputs=embedded_sequences, name="word_encoder"
)
word_encoder.summary()


class Vibes(keras.Model):
    def __init__(
        self, word_encoder, score_model, batch_size, maxlen, latent_dim, dt, **kwargs
    ):
        super().__init__(**kwargs)
        self.word_encoder = word_encoder
        self.score_fn = score_model

        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.maxlen = maxlen
        self.dt = dt

        self.true_emb = None
        self.pred_emb = None

        self.step_loss_tracker = keras.metrics.Mean(name="step_loss")
        self.sequence_loss_tracker = keras.metrics.Mean(name="sequence_loss")

    @property
    def metrics(self):
        return [self.step_loss_tracker, self.sequence_loss_tracker]

    def get_maxlen(self):
        return self.maxlen

    def apply_word_encoder(self, inputs):
        word_element = tf.expand_dims(inputs, axis=-1)
        word_embedding = self.word_encoder(word_element)
        return word_embedding

    def update_slice_by_gather(self, dx, xi, i, batch_size):
        batch_indices = tf.range(batch_size)
        step_indices = tf.fill([batch_size], i)

        reshaped_indices = batch_indices * self.maxlen + step_indices
        reshaped_indices = tf.expand_dims(reshaped_indices, axis=1)

        updates = tf.gather(dx, reshaped_indices[:, 0])
        xi_updated = tf.tensor_scatter_nd_add(xi, reshaped_indices, updates)
        return xi_updated

    def update_slice_by_tile(self, dx, xi, i, batch_size):
        batch_indices = tf.range(batch_size)
        step_indices = tf.fill([batch_size], i)

        indices = batch_indices * self.maxlen + step_indices
        indices = tf.expand_dims(indices, axis=1)

        updates = tf.tile(dx, [batch_size, 1])
        xi_updated = tf.tensor_scatter_nd_add(xi, indices, updates)
        return xi_updated

    def gather_diagonal_slices(self, xt):
        indices = tf.range(self.maxlen)
        indices = tf.stack([indices, indices], axis=1)
        diagonal = tf.gather_nd(xt, indices)
        return diagonal

    def vectorized_masking(
        self, embeddings_tr, batch_size, include=True, mask_with_random=False
    ):
        indices = tf.range(self.maxlen)
        time_steps = tf.expand_dims(tf.range(self.maxlen), axis=1)

        if include:
            masks = tf.less(indices, time_steps)
        else:
            masks = tf.less_equal(indices, time_steps)

        masks = tf.cast(masks, tf.float32)
        masks = tf.expand_dims(masks, axis=2)
        masks = tf.expand_dims(masks, axis=3)
        masks = tf.tile(masks, [1, 1, batch_size, self.latent_dim])

        embeddings_expanded = tf.expand_dims(embeddings_tr, axis=0)
        embeddings_expanded = tf.tile(embeddings_expanded, [self.maxlen, 1, 1, 1])

        if mask_with_random:
            random_values = tf.random.uniform(
                tf.shape(embeddings_expanded), minval=0.0, maxval=1.0
            )
            masked_embeddings = embeddings_expanded * masks + random_values * (
                1 - masks
            )
        else:
            masked_embeddings = embeddings_expanded * masks

        return masked_embeddings

    def diffusion(self, state, time):
        xt, labels, error, batch_size = state

        time = tf.fill([batch_size, 1], time)
        time = tf.tile(time, [1, self.maxlen])
        time = tf.reshape(time, (batch_size * self.maxlen, 1))

        score = self.score_fn([xt, time])

        # print("*** score ***")
        score2 = tf.reshape(score, shape=(batch_size, self.maxlen, self.latent_dim))
        # print(score2[1, :, :])

        xt2 = tf.reshape(
            xt, shape=(self.maxlen, self.maxlen, batch_size, self.latent_dim)
        )
        xt2 = self.gather_diagonal_slices(xt2)
        xt2 = tf.reshape(xt2, shape=(batch_size, self.maxlen, self.latent_dim))
        labels2 = tf.reshape(labels, shape=(batch_size, self.maxlen, self.latent_dim))

        diff = labels2 - xt2
        error += losses.mean_squared_error(diff, score2)

        score_norm = tf.norm(score)

        dW = tf.random.normal(
            shape=(self.maxlen * batch_size, self.latent_dim),
            mean=0.0,
            stddev=tf.sqrt(self.dt),
        )
        dx = (score * self.dt) + score_norm * dW

        xt = tf.transpose(xt, perm=[1, 0, 2])
        indexes = tf.range(self.maxlen, dtype=tf.int32)

        xt_updated = tf.map_fn(
            lambda args: self.update_slice_by_gather(dx, args[0], args[1], batch_size),
            elems=(xt, indexes),
            fn_output_signature=tf.TensorSpec(
                shape=(None, self.latent_dim), dtype=tf.float32
            ),
        )
        xt_updated = tf.transpose(xt_updated, perm=[1, 0, 2])
        return (xt_updated, labels, error, batch_size)

    def diffusion_loss(self, embeddings, batch_size):
        embeddings_tr = tf.transpose(embeddings, perm=[1, 0, 2])

        embeddings_tr_tiled = self.vectorized_masking(
            embeddings_tr, batch_size=batch_size, mask_with_random=True
        )
        score_input = tf.reshape(
            embeddings_tr_tiled,
            (self.maxlen, self.maxlen * batch_size, self.latent_dim),
        )
        score_input = tf.transpose(score_input, perm=[1, 0, 2])

        embeddings_tr_tiled_with_target = self.vectorized_masking(
            embeddings_tr, batch_size=batch_size, include=False
        )
        labels = self.gather_diagonal_slices(embeddings_tr_tiled_with_target)

        steps = tf.range(0, 1, delta=self.dt, dtype=tf.float32)
        step_loss = tf.zeros(shape=(batch_size, self.maxlen))

        initial_state = (score_input, labels, step_loss, batch_size)
        xt, _, error, _ = tf.scan(self.diffusion, steps, initializer=initial_state)

        scan_out = tf.reshape(
            xt,
            (int(1 / self.dt), self.maxlen, self.maxlen, batch_size, self.latent_dim),
        )

        # random_batch = np.random.randint(0, batch_size - 1)
        random_step = np.random.randint(0, self.maxlen - 1)
        self.true_emb = labels[random_step, 1, :2]
        self.pred_emb = scan_out[:, random_step, random_step, 1, :2]

        xt = tf.reshape(xt[-1], (self.maxlen, self.maxlen, batch_size, self.latent_dim))
        xt = self.gather_diagonal_slices(xt)
        xt = tf.transpose(xt, perm=[1, 0, 2])

        step_loss = tf.reduce_sum(error)
        sequence_loss = tf.reduce_mean(tf.abs(embeddings - xt))
        return (xt, step_loss, sequence_loss)

    def generate(self, state, time):
        xt, target_step, batch_size = state

        score_input = tf.transpose(xt, perm=[1, 0, 2])
        time = tf.expand_dims(time, axis=-1)

        score = self.score_fn([score_input, time])
        # score_norm = tf.norm(score)
        dx = score * self.dt  # + score_norm

        xt = tf.transpose(xt, perm=[1, 0, 2])
        xt = tf.reshape(xt, (batch_size * self.maxlen, self.latent_dim))

        xt_updated = self.update_slice_by_tile(dx, xt, target_step, batch_size)
        xt_updated = tf.reshape(xt_updated, (self.maxlen, batch_size, self.latent_dim))
        return (xt_updated, target_step, batch_size)

    def geretate_step(self, state, index):
        embeddings_tr, batch_size = state
        time = tf.range(0, 1, delta=self.dt, dtype=tf.float32)

        initial_state = (embeddings_tr, index, batch_size)
        updated_embeddings_tr, _, _ = tf.scan(
            self.generate, time, initializer=initial_state
        )

        embeddings_tr_final = updated_embeddings_tr[-1]
        return (embeddings_tr_final, batch_size)

    def diffusion_generate(self, embeddings_tr, batch_size, start_index):
        indexes = tf.range(start=start_index, limit=self.maxlen)

        initial_state = (embeddings_tr, batch_size)
        updated_embeddings_tr, _ = tf.scan(self.geretate_step, indexes, initial_state)

        embeddings_tr_final = updated_embeddings_tr[-1]
        embeddings_final = tf.transpose(embeddings_tr_final, perm=[1, 0, 2])
        return embeddings_final

    def train_step(self, data):

        with tf.GradientTape() as tape:
            embeddings = self.word_encoder(data)

            _, step_loss, sequence_loss = self.diffusion_loss(
                embeddings, batch_size=self.batch_size
            )

        grads = tape.gradient(step_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.step_loss_tracker.update_state(step_loss)
        self.sequence_loss_tracker.update_state(sequence_loss)

        return {
            "loss": self.step_loss_tracker.result(),
            "sequence": self.sequence_loss_tracker.result(),
        }
