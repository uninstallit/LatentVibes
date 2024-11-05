import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow.data as tf_data

import keras
from vibes import word_encoder, score_model, Vibes
from textGenerator import TextGenerator

from gloveEmbeddings import (
    prepare_lm_tokens,
    # prepare_lm_tokens_words,
    word_to_index,
    sentences,
    text_ds,
    vocab,
)

from vibesConfig import (
    latent_dim,
    maxlen,
    dt,
    epochs,
    batch_size,
    learning_rate,
    start_prompt,
    print_every_batch,
)

# normalized_tokens, words = prepare_lm_tokens_words(sentences)

text_ds_x_only = text_ds.map(prepare_lm_tokens, num_parallel_calls=tf_data.AUTOTUNE)
text_ds_x_only = text_ds_x_only.shuffle(buffer_size=256)
text_ds_x_only = text_ds_x_only.batch(batch_size, drop_remainder=True)
text_ds_x_only = text_ds_x_only.prefetch(tf_data.AUTOTUNE)

vae = Vibes(
    word_encoder,
    score_model,
    batch_size=batch_size,
    maxlen=maxlen,
    latent_dim=latent_dim,
    dt=dt,
)
vae.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate)  # , run_eagerly=True
)

start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
text_gen_callback = TextGenerator(start_tokens, maxlen, vocab, print_every_batch)
# troubleshoot
# text_gen_callback.on_batch_end(batch=1)

vae.fit(
    text_ds_x_only, epochs=epochs, batch_size=batch_size, callbacks=[text_gen_callback]
)
