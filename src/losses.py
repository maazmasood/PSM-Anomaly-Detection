"""
Custom loss functions and data augmentation for GCT-GAN.

- Geometric masking: random zero-masking augmentation for robustness.
- Triplet loss: contrastive learning on latent representations.
"""

import tensorflow as tf


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    ]
)
def geometric_masking(sequence, mask_ratio):
    """
    Apply random zero-masking to the input sequence (batch-wise).

    A fraction of elements (determined by ``mask_ratio``) is randomly set to
    zero, acting as data augmentation to improve model robustness.

    Args:
        sequence: Tensor of shape (batch, seq_len, n_features).
        mask_ratio: Scalar float32 tensor — fraction of elements to mask.

    Returns:
        Masked sequence with the same shape as input.
    """
    shape = tf.shape(sequence)
    batch_size, seq_len, n_features = shape[0], shape[1], shape[2]
    total_elements = seq_len * n_features

    mask_elements_float = tf.cast(total_elements, tf.float32) * mask_ratio
    mask_elements = tf.cast(mask_elements_float, tf.int32)
    total_elements_int = tf.cast(total_elements, tf.int32)

    flat_mask = tf.ones([batch_size, total_elements_int], dtype=tf.float32)

    def mask_single_sequence(single_flat_mask):
        indices = tf.range(total_elements_int)
        shuffled_indices = tf.random.shuffle(indices)
        mask_indices = shuffled_indices[:mask_elements]
        updates = tf.zeros_like(mask_indices, dtype=tf.float32)
        mask_indices_expanded = tf.expand_dims(mask_indices, axis=-1)
        return tf.tensor_scatter_nd_update(
            single_flat_mask, mask_indices_expanded, updates
        )

    flat_mask_updated = tf.map_fn(mask_single_sequence, flat_mask)
    final_mask = tf.reshape(flat_mask_updated, shape)
    return sequence * final_mask


@tf.function
def triplet_loss(anchor_z, positive_z, negative_z, margin=1.0):
    """
    Triplet loss on latent representations.

    Pulls the anchor closer to its positive (augmented) counterpart while
    pushing it away from a negative sample.

    Args:
        anchor_z:   Latent vector for origin sequences.
        positive_z: Latent vector for augmented sequences.
        negative_z: Latent vector for negative (rolled) sequences.
        margin:     Minimum desired gap between pos/neg distances.

    Returns:
        Scalar mean triplet loss.
    """
    pos_dist = tf.reduce_sum(tf.square(anchor_z - positive_z), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor_z - negative_z), axis=-1)
    loss = tf.maximum(0.0, pos_dist - neg_dist + margin)
    return tf.reduce_mean(loss)
