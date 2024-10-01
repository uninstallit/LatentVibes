import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
from keras import ops
from keras import layers


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


latent_dim = 10
maxlen = 80
vocab_size = 20000
embedding_dim = 16

word_encoder_inputs = keras.Input(shape=(1,), name="word_input")
time_inputs = keras.Input(shape=(1,), name="time_input")

word_embedding = layers.Embedding(
    input_dim=vocab_size, output_dim=embedding_dim, input_length=1
)(word_encoder_inputs)
word_embedding = layers.Flatten()(word_embedding)

time_embedding = layers.Embedding(
    input_dim=maxlen, output_dim=embedding_dim, input_length=1
)(time_inputs)
time_embedding = layers.Flatten()(time_embedding)

x = layers.concatenate([word_embedding, time_embedding])

x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(32, activation="relu")(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

word_encoder = keras.Model(
    inputs=[word_encoder_inputs, time_inputs],
    outputs=[z_mean, z_log_var, z],
    name="word_encoder",
)
word_encoder.summary()

# Word Decoder
word_latent_inputs = keras.Input(shape=(latent_dim,), name="latent_input")
time_inputs = keras.Input(shape=(1,), name="time_input")

time_embedding = layers.Embedding(
    input_dim=maxlen, output_dim=embedding_dim, input_length=1
)(time_inputs)
time_embedding = layers.Flatten()(time_embedding)

x = layers.concatenate([word_latent_inputs, time_embedding])
x = layers.Dense(32, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)

word_decoder_outputs = layers.Dense(1, activation="linear")(x)
word_decoder = keras.Model(
    inputs=[word_latent_inputs, time_inputs],
    outputs=word_decoder_outputs,
    name="word_decoder",
)
word_decoder.summary()


# # Score function
# xt_inputs = keras.Input(shape=(latent_dim,), name="xt_input")
# time_inputs = keras.Input(shape=(1,), name="time_input")

# # Time embedding
# time_embedding = layers.Embedding(
#     input_dim=maxlen, output_dim=embedding_dim, input_length=1
# )(time_inputs)
# time_embedding = layers.Flatten()(time_embedding)

# x = layers.concatenate([xt_inputs, time_embedding])
# x = layers.Dense(32, activation="relu")(x)
# x = layers.Dense(64, activation="relu")(x)

# score_output = layers.Dense(
#     latent_dim, activation="linear", name="log_gradient_output"
# )(x)
# score_model = keras.Model(
#     inputs=[xt_inputs, time_inputs], outputs=score_output, name="score_model"
# )
# score_model.summary()


# Score function
xt_inputs = keras.Input(shape=(maxlen, latent_dim), name="xt_input")
time_inputs = keras.Input(shape=(1,), name="time_input")  # Shape: (1,)

time_inputs_expanded = layers.RepeatVector(maxlen)(time_inputs)
x = layers.concatenate([xt_inputs, time_inputs_expanded], axis=-1)
x = layers.Flatten()(x)

x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)

score_output = layers.Dense(
    latent_dim, activation="linear", name="log_gradient_output"
)(x)
score_model = keras.Model(
    inputs=[xt_inputs, time_inputs], outputs=score_output, name="score_model"
)
score_model.summary()


class VAE(keras.Model):
    def __init__(
        self,
        word_encoder,
        word_decoder,
        score_model,
        batch_size,
        maxlen=80,
        dt=0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.word_encoder = word_encoder
        self.word_decoder = word_decoder
        self.score_fn = score_model
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.dt = dt

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def get_maxlen(self):
        return self.maxlen

    def apply_word_encoder(self, inputs):
        word_element, time_index = inputs
        time_index = tf.reshape(time_index, [])
        time_index = tf.fill([self.batch_size], time_index)
        z_mean, z_log_var, z = self.word_encoder([word_element, time_index])
        return z_mean, z_log_var, z

    def apply_word_decoder(self, inputs):
        z_element, time_index = inputs
        time_index = tf.reshape(time_index, [])
        time_index = tf.fill([self.batch_size], time_index)
        reconstruction = self.word_decoder([z_element, time_index])
        return reconstruction

    def mu(self, mu1, mu2):
        delta_mu = mu2 - mu1
        return delta_mu

    def sigma(self, z_log_var1, z_log_var2):
        max_z_log_var = tf.maximum(z_log_var1, z_log_var2)
        exp_diff1 = tf.exp(z_log_var1 - max_z_log_var)
        exp_diff2 = tf.exp(z_log_var2 - max_z_log_var)
        sum_exp = exp_diff1 + exp_diff2
        lse = max_z_log_var + tf.math.log(sum_exp)
        return lse

    def mask_only_target_step(self, input_tensor_tr, target_time_step):
        one_hot_mask = tf.one_hot(indices=target_time_step, depth=self.maxlen)
        one_hot_mask = tf.reshape(one_hot_mask, shape=(self.maxlen, 1, 1))
        mask = tf.tile(one_hot_mask, multiples=[1, self.batch_size, latent_dim])
        output_tensor = input_tensor_tr * mask
        return output_tensor

    def mask_below_target_step(self, input_tensor_tr, target_time_step):
        indices = tf.range(self.maxlen)
        mask = tf.less(indices, target_time_step)
        mask = tf.reshape(mask, (self.maxlen, 1, 1))
        mask = tf.cast(mask, tf.float32)
        mask = tf.tile(mask, [1, self.batch_size, latent_dim])
        output_tensor = input_tensor_tr * mask
        return output_tensor

    def diffusion(self, state, step, masked_mu_tr, masked_sigma_tr):
        xt, mu_tr, sigma_tr, error = state
        time = step

        time = tf.fill([self.batch_size, 1], time)
        sigma_std = tf.exp(0.5 * sigma_tr)

        t_index = tf.squeeze(tf.math.floor(time[0]))
        t_index = tf.cast(t_index, tf.int32)
        one_hot = tf.one_hot(indices=t_index, depth=80)
        one_hot = tf.reshape(one_hot, (self.maxlen, 1, 1))

        delta = one_hot * tf.expand_dims(xt, axis=0)
        score_input = masked_mu_tr + delta
        score_input = tf.transpose(score_input, perm=[1, 0, 2])

        score = self.score_fn([score_input, time])
        logdx = -(xt - mu_tr) / tf.math.abs(sigma_tr)

        error = tf.math.abs(logdx - score)

        dW = tf.random.normal(shape=tf.shape(mu_tr), mean=0.0, stddev=tf.sqrt(self.dt))
        dx = (0.5 * tf.square(sigma_std) * score) + sigma_std * dW
        xt = xt + dx
        return (xt, mu_tr, sigma_tr, error)

    def diffusion_step(
        self, index, mu_tr, sigma_tr, full_mu_tr, full_sigma_tr, full_index
    ):
        masked_mu_tr = self.mask_below_target_step(full_mu_tr, target_time_step=index)
        masked_sigma_tr = self.mask_below_target_step(
            full_sigma_tr, target_time_step=index
        )

        steps = int(1 / self.dt)
        time = (
            tf.cast(index, dtype=tf.float32)
            + tf.range(steps, dtype=tf.float32) * self.dt
        )

        x0 = mu_tr
        err0 = tf.zeros_like(x0)
        initial_state = (x0, mu_tr, sigma_tr, err0)

        xt, _, _, error = tf.scan(
            lambda state, t: self.diffusion(state, t, masked_mu_tr, masked_sigma_tr),
            time,
            initializer=initial_state,
        )

        xt_final = xt[-1]
        mean_error = tf.reduce_mean(error, axis=0)
        return xt_final, mean_error

    def diffusion_loss(self, z_mean, z_log_var):
        mu_tr = tf.transpose(z_mean, perm=[1, 0, 2])
        sigma_tr = tf.transpose(z_log_var, perm=[1, 0, 2])
        index = tf.range(tf.shape(mu_tr)[0])

        output_signature = (
            tf.TensorSpec(shape=(self.batch_size, latent_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(self.batch_size, latent_dim), dtype=tf.float32),
        )

        sequence, score_error = tf.map_fn(
            fn=lambda elems: self.diffusion_step(
                elems[0], elems[1], elems[2], mu_tr, sigma_tr, index
            ),
            elems=(index, mu_tr, sigma_tr),
            fn_output_signature=output_signature,
            parallel_iterations=10,
        )

        sequence = tf.transpose(sequence, perm=[1, 0, 2])
        score_error = tf.transpose(score_error, perm=[1, 0, 2])
        return sequence, score_error

    def train_step(self, data):

        with tf.GradientTape() as tape:
            # sequence steps
            sequence_length = tf.shape(data)[1]

            time_indices = tf.range(sequence_length)
            transposed_data = tf.transpose(data, perm=[1, 0])

            z_mean, z_log_var, z = tf.map_fn(
                fn=self.apply_word_encoder,
                elems=(transposed_data, time_indices),
                dtype=(tf.float32, tf.float32, tf.float32),
                parallel_iterations=10,
            )

            z_mean = tf.transpose(z_mean, perm=[1, 0, 2])
            z_log_var = tf.transpose(z_log_var, perm=[1, 0, 2])

            reconstruction = tf.map_fn(
                fn=self.apply_word_decoder,
                elems=(z, time_indices),
                dtype=tf.float32,
                parallel_iterations=10,
            )
            z = tf.transpose(z, perm=[1, 0, 2])

            reconstruction = tf.transpose(tf.squeeze(reconstruction, axis=-1))
            reconstruction_loss = ops.mean(
                ops.sum(
                    keras.losses.mean_absolute_error(data, reconstruction),
                    axis=-1,
                )
            )

            kl_loss = -0.5 * ops.sum(
                1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=-1
            )
            kl_loss = ops.mean(kl_loss)

            ez_log_var = tf.math.exp(z_log_var)
            sequence, score_error = self.diffusion_loss(z_mean, ez_log_var)
            score_loss = ops.mean(ops.sum(score_error, axis=-1))

            total_loss = (reconstruction_loss + kl_loss) + score_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "recon_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "loss": self.total_loss_tracker.result(),
        }


# Define the checkpoint callback to save only the best model
checkpoint_filepath = "/best_vae_model/ckpt/checkpoint.model.keras"
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,  # Save the entire model
    monitor="loss",  # Monitor the loss for determining the best model
    mode="min",  # Save the model with the minimum loss
    save_best_only=True,  # Save only when the monitored metric improves
    save_freq="epoch",  # Save at the end of every epoch
)
