"""
GCT-GAN training loop.

Alternating adversarial training: the Discriminator learns to distinguish
real from reconstructed sequences, while the Generator optimizes a
multi-objective loss (reconstruction + adversarial + contrastive).
"""

import numpy as np
import tensorflow as tf

from . import config
from .losses import geometric_masking, triplet_loss


def _make_train_step(generator, discriminator, generator_optimizer, cfg):
    """
    Create a compiled ``@tf.function`` training step.

    This factory pattern avoids issues with tracing closures over Python
    objects that change between calls.
    """
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    mse_loss = tf.keras.losses.MeanSquaredError()

    lambda_rec = cfg["lambda_rec"]
    lambda_adv = cfg["lambda_adv"]
    lambda_con = cfg["lambda_con"]
    mask_ratio = tf.constant(cfg["mask_ratio"], dtype=tf.float32)

    @tf.function
    def train_step(real_sequences):
        batch_size = tf.shape(real_sequences)[0]

        # --- Prepare augmented & negative samples ---
        augmented_sequences = geometric_masking(real_sequences, mask_ratio)
        negative_sequences = tf.roll(real_sequences, shift=1, axis=0)

        # --- Train Discriminator ---
        with tf.GradientTape() as tape:
            reconstruction, _ = generator(augmented_sequences, training=True)
            d_real = discriminator(real_sequences, training=True)
            d_fake = discriminator(reconstruction, training=True)

            real_labels = tf.ones((batch_size, 1)) * 0.9  # label smoothing
            fake_labels = tf.zeros((batch_size, 1)) + 0.1

            d_loss_real = bce_loss(real_labels, d_real)
            d_loss_fake = bce_loss(fake_labels, d_fake)
            d_loss = d_loss_real + d_loss_fake

        d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
        discriminator.optimizer.apply_gradients(
            zip(d_grads, discriminator.trainable_variables)
        )

        # --- Train Generator ---
        with tf.GradientTape() as tape:
            reco_a, z_a = generator(real_sequences, training=True)
            _, z_p = generator(augmented_sequences, training=True)
            _, z_n = generator(negative_sequences, training=True)

            rec_loss = mse_loss(real_sequences, reco_a) * lambda_rec
            d_reco_a = discriminator(reco_a, training=False)
            adv_loss = bce_loss(tf.ones_like(d_reco_a), d_reco_a) * lambda_adv
            con_loss = triplet_loss(z_a, z_p, z_n) * lambda_con

            g_loss = rec_loss + adv_loss + con_loss

        g_grads = tape.gradient(g_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(
            zip(g_grads, generator.trainable_variables)
        )

        return d_loss, rec_loss, adv_loss, con_loss, g_loss

    return train_step


def train(generator, discriminator, X_train, epochs=None, batch_size=None):
    """
    Run the GCT-GAN training loop.

    Args:
        generator:     Keras Model (autoencoder).
        discriminator: Keras Model (classifier), must already be compiled.
        X_train:       Training data, float32 array of shape (N, W, F).
        epochs:        Number of training epochs (default: ``config.EPOCHS``).
        batch_size:    Batch size (default: ``config.BATCH_SIZE``).

    Returns:
        list[dict] — per-epoch loss history with keys
        ``d_loss``, ``rec_loss``, ``adv_loss``, ``con_loss``.
    """
    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE

    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE
    )

    # Build TF dataset
    dataset = (
        tf.data.Dataset.from_tensor_slices(X_train)
        .shuffle(len(X_train))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    cfg = {
        "lambda_rec": config.LAMBDA_REC,
        "lambda_adv": config.LAMBDA_ADV,
        "lambda_con": config.LAMBDA_CON,
        "mask_ratio": config.MASK_RATIO,
    }
    train_step = _make_train_step(
        generator, discriminator, generator_optimizer, cfg
    )

    print("\n--- Starting GCT-GAN Training ---")
    history = []

    for epoch in range(epochs):
        d_losses, rec_losses, adv_losses, con_losses, g_losses = (
            [],
            [],
            [],
            [],
            [],
        )

        for x_batch in dataset:
            d_loss, rec_loss, adv_loss, con_loss, g_loss = train_step(x_batch)
            d_losses.append(d_loss.numpy())
            rec_losses.append(rec_loss.numpy())
            adv_losses.append(adv_loss.numpy())
            con_losses.append(con_loss.numpy())
            g_losses.append(g_loss.numpy())

        avg = {
            "d_loss": np.mean(d_losses),
            "rec_loss": np.mean(rec_losses),
            "adv_loss": np.mean(adv_losses),
            "con_loss": np.mean(con_losses),
        }
        history.append(avg)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"D_Loss: {avg['d_loss']:.4f} | "
            f"G_Total: {np.mean(g_losses):.4f} "
            f"(REC: {avg['rec_loss']:.4f}, "
            f"ADV: {avg['adv_loss']:.4f}, "
            f"CON: {avg['con_loss']:.4f})"
        )

    print("\n--- Training Complete ---")
    return history
