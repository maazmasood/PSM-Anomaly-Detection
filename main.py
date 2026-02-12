"""
GCT-GAN: Geometric Contrastive Transformer-GAN for Anomaly Detection.

CLI entry point that orchestrates the full pipeline:
    data loading → model building → training → evaluation → visualization.
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

from src import config
from src.data import load_and_prepare_data
from src.evaluate import evaluate
from src.models import build_discriminator, build_generator
from src.train import train
from src.visualize import plot_anomaly_detection, plot_loss_history


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GCT-GAN — Anomaly Detection on PSM Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Training batch size.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.OUTPUT_DIR,
        help="Directory for output plots.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively (requires GUI).",
    )
    return parser.parse_args()


def main():
    """Run the full GCT-GAN anomaly detection pipeline."""
    args = parse_args()

    # Override config with CLI args
    config.OUTPUT_DIR = args.output_dir
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # --- Reproducibility ---
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print("=" * 60)
    print("  GCT-GAN: Anomaly Detection on PSM Dataset")
    print("=" * 60)

    # --- 1. Data Loading & Preprocessing ---
    print("\n[1/4] Loading and preprocessing data...")
    data = load_and_prepare_data()

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    n_features = data["n_features"]

    input_shape = (config.WINDOW_SIZE, n_features)

    # --- 2. Build Models ---
    print("\n[2/4] Building models...")
    generator = build_generator(
        input_shape,
        config.LATENT_DIM,
        config.NUM_HEADS,
        config.FF_DIM,
        config.N_LAYERS,
        config.DROPOUT_RATE,
    )
    discriminator = build_discriminator(input_shape)
    discriminator.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="binary_crossentropy",
    )

    generator.summary()
    discriminator.summary()

    # --- 3. Training ---
    print("\n[3/4] Training...")
    history = train(
        generator,
        discriminator,
        X_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # --- 4. Evaluation & Visualization ---
    print("\n[4/4] Evaluating and generating plots...")
    results = evaluate(generator, discriminator, X_train, X_test, y_test)

    plot_anomaly_detection(
        results["anomaly_scores"],
        results["y_pred"],
        results["threshold"],
        results["f1_score"],
        output_dir=args.output_dir,
        show=args.show_plots,
    )
    plot_loss_history(
        history,
        output_dir=args.output_dir,
        show=args.show_plots,
    )

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  F1-Score: {results['f1_score']:.4f}")
    print(f"  Outputs saved to: {os.path.abspath(args.output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
