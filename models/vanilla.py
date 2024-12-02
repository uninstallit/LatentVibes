import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers

from vibesConfig import (
    latent_dim,
    maxlen,
)

# =========================
# Score Function
# =========================

xt_inputs = keras.Input(shape=(maxlen, latent_dim), name="xt_input")
time_inputs = keras.Input(shape=(1,), name="time_input")

time_inputs_expanded = layers.RepeatVector(maxlen)(time_inputs)
x = layers.concatenate([xt_inputs, time_inputs_expanded], axis=-1)

x = layers.TimeDistributed(layers.Dense(32, activation="relu"))(x)
x = layers.TimeDistributed(layers.Dense(32, activation="relu"))(x)
x = layers.TimeDistributed(layers.Dense(32, activation="relu"))(x)

x = layers.Flatten()(x)
x = layers.LayerNormalization()(x)

x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)

score_output = layers.Dense(
    latent_dim, activation="linear", name="log_gradient_output"
)(x)
score_model = keras.Model(
    inputs=[xt_inputs, time_inputs], outputs=score_output, name="score_model"
)
score_model.summary()