import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
from keras import layers

from vibesConfig import (
    latent_dim,
    maxlen,
    num_heads,
    ff_dim,
    batch_size,
    num_transformer_blocks,
    head_size,
    num_heads,
    ff_dim,
    dropout,
)

xt_inputs = keras.Input(
    shape=(maxlen, latent_dim), batch_size=batch_size, name="xt_input"
)
time_inputs = keras.Input(shape=(1,), batch_size=batch_size, name="time_input")

time_dense = layers.Dense(latent_dim, activation="relu")(
    time_inputs
)  # (batch_size, latent_dim)

time_dense = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(
    time_dense
)  # (batch_size, 1, latent_dim)

time_dense = layers.Lambda(lambda x: tf.tile(x, [1, maxlen, 1]))(
    time_dense
)  # (batch_size, maxlen, latent_dim)

x = layers.Concatenate()(
    [xt_inputs, time_dense]
)  # (batch_size, maxlen, latent_dim * 2)


# Build a stack of transformer encoder blocks
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)
    x = x + res
    return x


# Apply transformer blocks
for _ in range(num_transformer_blocks):
    x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

x = layers.GlobalAveragePooling1D()(x)  # (batch_size, latent_dim * 2)
x = layers.Dense(latent_dim, activation="relu")(x)  # (batch_size, latent_dim)

score_output = layers.Dense(
    latent_dim, activation="linear", name="log_gradient_output"
)(
    x
)  # (batch_size, latent_dim)

score_model = keras.Model(
    inputs=[xt_inputs, time_inputs], outputs=score_output, name="score_model"
)
score_model.summary()
