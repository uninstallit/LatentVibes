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
embedding_dim = 32

# Word input
word_encoder_inputs = keras.Input(shape=(1,), name="word_input")

# Word Embedding
word_embedding = layers.Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    # input_length=1,
    name="word_embedding",
)(word_encoder_inputs)
word_embedding = layers.Flatten(name="flatten_word_embedding")(word_embedding)

x = layers.Dense(128, activation="relu", name="dense_128")(word_embedding)
# x = layers.BatchNormalization(name="bn_128")(x)
x = layers.LayerNormalization()(x)

x = layers.Dense(64, activation="relu", name="dense_64")(x)
x = layers.Dense(32, activation="relu", name="dense_32")(x)

# Latent space
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling(name="sampling")([z_mean, z_log_var])

# Define the encoder model
word_encoder = keras.Model(
    inputs=word_encoder_inputs, outputs=[z_mean, z_log_var, z], name="word_encoder"
)
word_encoder.summary()

word_latent_inputs = keras.Input(shape=(latent_dim,), name="latent_input")

x = layers.Dense(128, activation="relu", name="dense_128")(word_latent_inputs)
# x = layers.BatchNormalization(name="bn_128")(x)
x = layers.LayerNormalization()(x)

x = layers.Dense(64, activation="relu", name="dense_64")(x)
x = layers.Dense(32, activation="relu", name="dense_32")(x)
word_decoder_outputs = layers.Dense(1, activation="linear", name="decoder_output")(x)

word_decoder = keras.Model(
    inputs=word_latent_inputs,
    outputs=word_decoder_outputs,
    name="word_decoder",
)
word_decoder.summary()


# Score function
xt_inputs = keras.Input(shape=(maxlen, latent_dim), name="xt_input")
time_inputs = keras.Input(shape=(1,), name="time_input")

time_inputs_expanded = layers.RepeatVector(maxlen)(time_inputs)
x = layers.concatenate([xt_inputs, time_inputs_expanded], axis=-1)
x = layers.TimeDistributed(layers.Dense(64, activation="relu"))(x)

x = layers.Flatten()(x)
x = layers.LayerNormalization()(x)

x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(32, activation="relu")(x)

score_output = layers.Dense(
    latent_dim, activation="linear", name="log_gradient_output"
)(x)
score_model = keras.Model(
    inputs=[xt_inputs, time_inputs], outputs=score_output, name="score_model"
)
score_model.summary()

# xt_inputs = keras.Input(shape=(maxlen, latent_dim), name="xt_input")
# time_inputs = keras.Input(shape=(1,), name="time_input")

# time_inputs_expanded = layers.RepeatVector(maxlen)(time_inputs)
# x = layers.concatenate([xt_inputs, time_inputs_expanded], axis=-1)

# x = layers.TimeDistributed(layers.Dense(64, activation="relu"))(x)
# attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)

# x = layers.Add()([x, attention_output])
# x = layers.LayerNormalization()(x)

# x = layers.Flatten()(x)
# x = layers.Dense(64, activation="relu")(x)
# x = layers.Dense(64, activation="relu")(x)
# x = layers.Dense(64, activation="relu")(x)

# score_output = layers.Dense(
#     latent_dim, activation="linear", name="log_gradient_output"
# )(x)
# score_model = keras.Model(
#     inputs=[xt_inputs, time_inputs],
#     outputs=score_output,
#     name="score_model_with_attention",
# )
# score_model.summary()


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
        word_element = tf.expand_dims(word_element, axis=-1)
        z_mean, z_log_var, z = self.word_encoder(word_element)
        return z_mean, z_log_var, z

    def apply_word_decoder(self, inputs):
        z_element, time_index = inputs
        reconstruction = self.word_decoder(z_element)
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

    def mask_below_target_step(self, input_tensor_tr, target_time_step, batch_size):
        indices = tf.range(self.maxlen)
        mask = tf.less(indices, target_time_step)
        mask = tf.reshape(mask, (self.maxlen, 1, 1))
        mask = tf.cast(mask, tf.float32)
        mask = tf.tile(mask, [1, batch_size, latent_dim])
        output_tensor = input_tensor_tr * mask
        return output_tensor

    def shift_tensor_forward(self, x):
        zeros = tf.zeros_like(x[:, 0:1, :])
        shifted_x = tf.concat([zeros, x[:, :-1, :]], axis=1)
        return shifted_x

    def add_to_diagonal_simple(self, tensorA, tensorB, batch_size):

        # Create batch indices
        batch_indices = tf.range(batch_size * self.maxlen)  # Shape: (batch_steps,)

        # Compute step indices (diagonal positions)
        step_indices = tf.math.mod(batch_indices, self.maxlen)  # Shape: (batch_steps,)

        # Create a mask for the diagonal elements
        mask = tf.one_hot(step_indices, self.maxlen)  # Shape: (batch_steps, steps)

        # Expand dimensions to match A and B
        mask_expanded = tf.expand_dims(mask, axis=-1)  # Shape: (batch_steps, steps, 1)
        tensorB_expanded = tf.expand_dims(
            tensorB, axis=1
        )  # Shape: (batch_steps, 1, features)

        # Compute the updates
        updates = (
            mask_expanded * tensorB_expanded
        )  # Shape: (batch_steps, steps, features)

        # Add the updates to A
        tensorA_updated = tensorA + updates

        return tensorA_updated

    def diffusion(self, state, time):
        # shape=(80, 80, 10)
        xt, batch_size = state

        time = tf.fill([batch_size, 1], time)
        time = tf.tile(time, [1, self.maxlen])
        time = tf.reshape(time, (batch_size * self.maxlen, 1))

        score = self.score_fn([xt, time])

        dW = tf.random.normal(shape=tf.shape(score), mean=0.0, stddev=tf.sqrt(self.dt))
        dx = (0.5 * score) + dW
        xt = self.add_to_diagonal_simple(xt, dx, batch_size)

        return (xt, batch_size)

    def diffusion_loss(self, z_mean, z_log_var, batch_size):

        mu = tf.expand_dims(z_mean, axis=1)
        mu = tf.tile(mu, multiples=[1, self.maxlen, 1, 1])

        # mask upper triangle
        mu = tf.linalg.band_part(mu, -1, 0)

        diagonal = tf.linalg.diag_part(mu)
        score_input = tf.linalg.set_diag(mu, tf.zeros(tf.shape(diagonal)))

        x0 = tf.reshape(
            score_input, (self.maxlen * batch_size, self.maxlen, latent_dim)
        )

        steps = tf.range(0, 1, delta=self.dt, dtype=tf.float32)

        initial_state = (x0, batch_size)

        xt, _ = tf.scan(self.diffusion, steps, initializer=initial_state)

        xt_final = tf.reshape(
            xt[-1], shape=(batch_size, self.maxlen, self.maxlen, latent_dim)
        )

        loss = tf.reduce_mean(tf.abs(mu - xt_final))

        return loss



    def add_tensor_slice(self, tensorA, tensorB, target_step, batch_size):
        time_indices = tf.fill([batch_size], target_step)
        batch_indices = tf.range(batch_size, dtype=tf.int32)
        indices = tf.stack([time_indices, batch_indices], axis=1)
        tensorA = tf.tensor_scatter_nd_add(tensorA, indices, tensorB)
        return tensorA

    def update_tensor_slice(self, tensorA, tensorB, target_step, batch_size):
        time_indices = tf.fill([batch_size], target_step)
        batch_indices = tf.range(batch_size, dtype=tf.int32)
        indices = tf.stack([time_indices, batch_indices], axis=1)
        tensorA = tf.tensor_scatter_nd_update(tensorA, indices, tensorB)
        return tensorA

    def generate(self, state, time):
        xt, masked_mu_tr, masked_sigma_tr, target_step, batch_size = state

        score_input = tf.transpose(masked_mu_tr, perm=[1, 0, 2])
        time = tf.expand_dims(time, axis=-1)
        score = self.score_fn([score_input, time])

        dW = tf.random.normal(shape=tf.shape(xt), mean=0.0, stddev=tf.sqrt(self.dt))
        dx = (0.5 * score) + dW
        xt = xt + dx

        masked_mu_tr = self.update_tensor_slice(
            masked_mu_tr, xt, target_step, batch_size
        )
        return (xt, masked_mu_tr, masked_sigma_tr, target_step, batch_size)

    def geretate_step(self, state, index):
        masked_mu_tr, masked_sigma_tr, batch_size = state

        steps = int(1 / self.dt)
        time = tf.range(0, steps, dtype=tf.float32) * self.dt

        x0 = masked_mu_tr[index, :, :]
        initial_state = (x0, masked_mu_tr, masked_sigma_tr, index, batch_size)

        _, updated_mu_tr, updated_sigma_tr, _, _ = tf.scan(
            self.generate, time, initializer=initial_state
        )

        mu_tr_final = updated_mu_tr[-1]
        sigma_tr_final = updated_sigma_tr[-1]
        return (mu_tr_final, sigma_tr_final, batch_size)

    def diffusion_generate(self, z_mean, z_log_var, batch_size, start_index):
        mu_tr = tf.transpose(z_mean, perm=[1, 0, 2])
        sigma_tr = tf.transpose(z_log_var, perm=[1, 0, 2])

        index = tf.range(start=start_index, limit=tf.shape(mu_tr)[0])

        masked_mu_tr = self.mask_below_target_step(
            mu_tr, target_time_step=start_index, batch_size=batch_size
        )
        masked_sigma_tr = self.mask_below_target_step(
            sigma_tr, target_time_step=start_index, batch_size=batch_size
        )

        initial_state = (masked_mu_tr, masked_sigma_tr, batch_size)
        updated_mu_tr, _, _ = tf.scan(self.geretate_step, index, initial_state)

        mu_tr_final = updated_mu_tr[-1]
        mu_final = tf.transpose(mu_tr_final, perm=[1, 0, 2])
        return mu_final

    def train_step(self, data):

        with tf.GradientTape(persistent=True) as tape:
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

            reconstruction_loss = keras.losses.mean_squared_error(data, reconstruction)
            reconstruction_loss = tf.reduce_mean(reconstruction_loss)

            kl_loss = -0.5 * tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
            )
            kl_loss = tf.reduce_mean(kl_loss)

            ez_log_var = tf.math.exp(z_log_var)

            score_loss = self.diffusion_loss(
                z_mean, ez_log_var, batch_size=self.batch_size
            )

            # score_loss = ops.mean(ops.sum(score_error, axis=-1))

            beta = 0.8
            total_loss = (
                beta * (reconstruction_loss + kl_loss) + (1.0 - beta) * score_loss
            )

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
# checkpoint_filepath = "/best_vae_model/ckpt/checkpoint.model.keras"
# checkpoint_callback = keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=False,  # Save the entire model
#     monitor="loss",  # Monitor the loss for determining the best model
#     mode="min",  # Save the model with the minimum loss
#     save_best_only=True,  # Save only when the monitored metric improves
#     save_freq="epoch",  # Save at the end of every epoch
# )
