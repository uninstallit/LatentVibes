import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import random
import string
import tensorflow as tf
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings

import keras
from keras.layers import TextVectorization
from vibes import word_encoder, word_decoder, score_model, VAE

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import textwrap

# The dataset contains each review in a separate text file
# The text files are present in four different folders
# Create a list all files
filenames = []
directories = [
    # "aclImdb/train/pos",
    # "aclImdb/train/neg",
    "aclImdb/test/pos",
    # "aclImdb/test/neg",
]
for dir in directories:
    for f in os.listdir(dir):
        filenames.append(os.path.join(dir, f))

epochs = 10
batch_size = 24
text_ds = tf_data.TextLineDataset(filenames)

sentences = [line.numpy().decode("utf-8") for line in text_ds]


def custom_standardization(input_string):
    """Remove html line-break tags and handle punctuation"""
    lowercased = tf_strings.lower(input_string)
    stripped_html = tf_strings.regex_replace(lowercased, "<br />", " ")
    return tf_strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")


vocab_size = 20000  # Only consider the top 20k words
maxlen = 80  # Max sequence size

# Create a vectorization layer and adapt it to the text
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size - 1,
    output_mode="int",
    output_sequence_length=maxlen + 1,
)
vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()  # To get words back from token indices

# print(vocab)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_label_clusters(vae, data, labels, level="sentence"):
    import numpy as np
    import matplotlib.pyplot as plt
    import textwrap
    import tensorflow as tf

    batch_size = data.shape[0]
    sequence_length = data.shape[1]

    # Reshape data and get word-level latent representations
    data_reshaped = tf.reshape(data, [-1, 1])
    z_mean_word, _, _ = vae.word_encoder.predict(data_reshaped, verbose=0)
    latent_dim_word = z_mean_word.shape[-1]

    if level == "word":
        # Word-level latent representations
        z_mean_word = tf.reshape(
            z_mean_word, [batch_size * sequence_length, latent_dim_word]
        )
        x = z_mean_word[:, 0]
        y = z_mean_word[:, 1] if latent_dim_word > 1 else np.zeros_like(x)

        labels_flat = [word for seq in labels for word in seq]
    elif level == "sentence":
        # Sentence-level latent representations
        z_mean_word = tf.reshape(
            z_mean_word, [batch_size, sequence_length, latent_dim_word]
        )
        sz_mean, _, _ = vae.sequence_encoder.predict(z_mean_word, verbose=0)
        x = sz_mean[:, 0]
        y = sz_mean[:, 1] if sz_mean.shape[1] > 1 else np.zeros_like(x)
        labels_flat = labels
    else:
        raise ValueError("Invalid level. Choose 'word' or 'sentence'.")

    # Filter out padding tokens or zero vectors
    valid_indices = ~(np.isnan(x) | np.isnan(y) | (x == 0) & (y == 0))
    x = x[valid_indices]
    y = y[valid_indices]
    labels_filtered = [
        label for idx, label in enumerate(labels_flat) if valid_indices[idx]
    ]

    # Prepare values for coloring
    color_values = y

    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(
        x,
        y,
        c=color_values,
        cmap="viridis",
        alpha=0.6,
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Value for Coloring")

    # Annotations
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    annot.set_visible(False)

    # Function to update the annotation
    def update_annot(ind):
        idx = ind["ind"][0]
        pos = scatter.get_offsets()[idx]
        annot.xy = pos

        if level == "word":
            label_text = labels_filtered[idx]
        else:
            label_text = " ".join(labels_filtered[idx])

        wrapped_text = textwrap.fill(label_text, width=50)

        annot.set_text(wrapped_text)
        annot.get_bbox_patch().set_facecolor("yellow")
        annot.get_bbox_patch().set_alpha(0.8)

    # Function to handle mouse motion
    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = scatter.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title(f"{level.capitalize()}-Level Latent Space Visualization")
    plt.show()


def plot_running_sum(data, vae, maxlen=None):

    if not isinstance(data, tf.Tensor):
        data = tf.convert_to_tensor(data)

    samples, steps, _ = data.shape
    data_reshaped = tf.reshape(data, [-1, 1])

    # Get the mean from the VAE's word encoder
    z_mean, _, _ = vae.word_encoder.predict(data_reshaped, verbose=0)
    _, feat = z_mean.shape
    z_mean_reshaped = tf.reshape(z_mean, (samples, steps, feat))

    sequence = vae.diffusion(z_mean_reshaped, axis=1)
    sequence_np = sequence.numpy()

    plt.figure(figsize=(12, 7))

    # Plot each path without labels to reduce legend clutter
    for path in sequence_np:
        plt.plot(range(maxlen), path[:, 0], alpha=0.5)

    # Add labels and title
    plt.xlabel("Time Step", fontsize=14)
    plt.ylabel("Cumulative Sum", fontsize=14)
    plt.title("Running Sum over Time (Multiple Paths)", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


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

vae = VAE(
    word_encoder, word_decoder, score_model, batch_size=batch_size, maxlen=80, dt=0.01
)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

# example
sample_text = "For a movie that gets no respect there sure are a lot of memorable quotes listed for this gem."
sample_tokens, sample_words = prepare_lm_tokens_words(sentences)

sample_tokens = tf.expand_dims(sample_tokens, -1)


class TextGenerator(keras.callbacks.Callback):
    def __init__(self, start_tokens, max_tokens, vocab):
        self.start_tokens = start_tokens
        self.max_tokens = max_tokens
        self.vocab = vocab

        self.pad_token = 0  # Assuming 0 is the padding token index
        self.unk_token = 1  # Assuming 1 is the <UNK> token index

    def on_epoch_end(self, epoch, logs=None):
        # Step 1: Start with the initial tokens
        generated_tokens = self.start_tokens.copy()

        # Step 2: Pad the tokens to maxlen if necessary
        if len(generated_tokens) < self.max_tokens:
            padding_length = self.max_tokens - len(generated_tokens)
            generated_tokens += [self.pad_token] * padding_length
        else:
            generated_tokens = generated_tokens[: self.max_tokens]

        # Step 3: Perform diffusion step-by-step to generate the latent path
        input_tokens = tf.convert_to_tensor(generated_tokens, dtype=tf.float32)
        z_mean, z_log_var, _ = self.model.word_encoder.predict(input_tokens, verbose=0)

        z_mean = tf.expand_dims(z_mean, axis=0)
        z_log_var = tf.expand_dims(z_log_var, axis=0)

        ez_log_var = tf.math.exp(z_log_var)
        # sequence shape: sequence:  (1, 80, 10)
        sequence, _ = self.model.diffusion_loss(z_mean, ez_log_var, batch_size=1)

        # Step 4: Decode each latent vector to obtain tokens
        sequence = tf.squeeze(sequence, axis=0)
        decoded_sequence = self.model.word_decoder.predict(sequence, verbose=0)

        token_ids = np.rint(decoded_sequence).astype(int).flatten()

        # Step 5: Convert tokens to words and combine into single string
        generated_words = [
            self.vocab[token_id] if 0 <= token_id < len(self.vocab) else "<UNK>"
            for token_id in token_ids
        ]

        generated_text = " ".join(generated_words)

        wrapped_text = textwrap.fill(generated_text, width=70)
        border = "*" * 80
        print("\n" + border)
        print(f"Generated Text (Epoch #{epoch}):")
        print(f"\n- {wrapped_text}")
        print(border + "\n")


# Tokenize starting prompt
word_to_index = {}
for index, word in enumerate(vocab):
    word_to_index[word] = index

max_tokens = 80
start_prompt = "this movie is"
start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]

text_gen_callback = TextGenerator(start_tokens, max_tokens, vocab)
# text_gen_callback.on_epoch_end(epoch=1)

vae.fit(
    text_ds_x_only, epochs=epochs, batch_size=batch_size, callbacks=[text_gen_callback]
)

# plot after training
plot_label_clusters(vae, sample_tokens, sample_words, level="word")
