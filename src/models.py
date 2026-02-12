"""
Model architecture definitions for GCT-GAN.

- Transformer encoder block (Multi-Head Attention + FFN with skip connections)
- Generator: Transformer-based autoencoder that produces (reconstruction, latent_z)
- Discriminator: 1-D CNN binary classifier (real vs. fake)
"""

from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    MultiHeadAttention,
    Reshape,
)
from tensorflow.keras.models import Model


def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    """
    Single Transformer encoder block.

    Consists of Multi-Head Self-Attention followed by a position-wise
    Feed-Forward Network, each with layer normalization and residual
    connections.

    Args:
        inputs:    Tensor of shape (batch, seq_len, d_model).
        head_size: Dimensionality of each attention head.
        num_heads: Number of attention heads.
        ff_dim:    Hidden size in the feed-forward network.
        dropout:   Dropout rate.

    Returns:
        Output tensor with the same shape as *inputs*.
    """
    # --- Multi-Head Self-Attention ---
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(
        x, x
    )
    res = x + inputs  # skip connection

    # --- Feed-Forward Network ---
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res  # skip connection


def build_generator(input_shape, latent_dim, num_heads, ff_dim, n_layers, dropout=0.1):
    """
    Build the GCT-GAN Generator (Transformer Autoencoder).

    The encoder stacks ``n_layers`` transformer blocks, pools the output to a
    latent vector **Z**, then decodes back to the original sequence shape.

    Args:
        input_shape: Tuple (window_size, n_features).
        latent_dim:  Dimensionality of the latent vector Z.
        num_heads:   Number of attention heads per block.
        ff_dim:      FFN hidden size per block.
        n_layers:    Number of stacked transformer encoder blocks.
        dropout:     Dropout rate.

    Returns:
        Keras ``Model`` with inputs → [reconstruction, z].
    """
    inputs = Input(shape=input_shape)

    # --- Encoder ---
    x = inputs
    for _ in range(n_layers):
        x = transformer_encoder_block(x, latent_dim, num_heads, ff_dim, dropout)

    # --- Latent space ---
    x_pooled = GlobalAveragePooling1D()(x)
    z = Dense(latent_dim, activation="relu", name="latent_vector")(x_pooled)

    # --- Decoder ---
    x = Dense(input_shape[0] * input_shape[1], activation="relu")(z)
    x = Reshape((input_shape[0], input_shape[1]))(x)
    reconstruction = Dense(input_shape[1], activation="linear")(x)

    return Model(inputs=inputs, outputs=[reconstruction, z], name="GCT_Generator")


def build_discriminator(input_shape):
    """
    Build the GCT-GAN Discriminator (1-D CNN).

    A simple convolutional classifier that outputs the probability that the
    input sequence is *real* (as opposed to reconstructed).

    Args:
        input_shape: Tuple (window_size, n_features).

    Returns:
        Keras ``Model`` with inputs → sigmoid probability.
    """
    inputs = Input(shape=input_shape)

    x = Conv1D(32, 5, activation="relu", padding="same")(inputs)
    x = Conv1D(64, 5, activation="relu", padding="same")(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)

    output = Dense(1, activation="sigmoid", name="real_vs_fake")(x)

    return Model(inputs=inputs, outputs=output, name="GCT_Discriminator")
