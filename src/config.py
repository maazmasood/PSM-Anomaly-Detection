"""
Configuration and hyperparameters for GCT-GAN anomaly detection.

All tunable parameters are centralized here for easy experimentation.
"""

import os

# =============================================================================
# Data Paths
# =============================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
LABEL_FILE = os.path.join(DATA_DIR, "test_label.csv")

# =============================================================================
# Output
# =============================================================================
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")

# =============================================================================
# Windowing
# =============================================================================
WINDOW_SIZE = 128   # Sequence length for Transformer input
STRIDE = 16         # Overlap between windows

# =============================================================================
# Training
# =============================================================================
BATCH_SIZE = 128
EPOCHS = 3          # Reduce to 1 for quick testing, increase to 50+ for optimal results
LEARNING_RATE = 0.0002
RANDOM_SEED = 42

# =============================================================================
# Model Architecture
# =============================================================================
LATENT_DIM = 64     # Dimension of the encoded latent vector (Z)
NUM_HEADS = 4       # Number of attention heads in Transformer
FF_DIM = 128        # Hidden layer size in the Transformer Feed-Forward Network
N_LAYERS = 2        # Number of Transformer encoder blocks
DROPOUT_RATE = 0.1  # Dropout rate in Transformer blocks
MASK_RATIO = 0.1    # Ratio of values to mask during geometric masking

# =============================================================================
# Loss Weighting (Critical for GAN stability)
# =============================================================================
LAMBDA_REC = 1.0    # Weight for reconstruction loss
LAMBDA_ADV = 0.01   # Weight for adversarial loss
LAMBDA_CON = 0.5    # Weight for contrastive (triplet) loss

# =============================================================================
# Anomaly Scoring
# =============================================================================
BETA = 0.6          # Anomaly Score weight: S = (1 - BETA) * REC + BETA * (1 - D_score)
THRESHOLD_PERCENTILE = 99.0  # Percentile of normal scores used as threshold
