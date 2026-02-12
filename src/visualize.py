"""
Visualization utilities for GCT-GAN anomaly detection results.

Saves plots to disk (and optionally displays them).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


def plot_anomaly_detection(
    anomaly_scores,
    y_pred,
    threshold,
    f1_score,
    output_dir=None,
    show=False,
):
    """
    Plot anomaly scores with threshold line and predicted anomaly markers.

    Args:
        anomaly_scores: 1-D array of test anomaly scores.
        y_pred:         1-D int array of predictions.
        threshold:      Detection threshold value.
        f1_score:       F1 metric to display in the title.
        output_dir:     Directory to save the figure (default: ``config.OUTPUT_DIR``).
        show:           Whether to call ``plt.show()`` (useful in notebooks).
    """
    output_dir = output_dir or config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(18, 6))
    plt.plot(anomaly_scores, label="Anomaly Score", color="blue", alpha=0.7)
    plt.axhline(
        threshold,
        color="red",
        linestyle="--",
        label=f"Threshold ({threshold:.4f})",
    )

    predicted_idx = np.where(y_pred == 1)[0]
    plt.scatter(
        predicted_idx,
        anomaly_scores[predicted_idx],
        color="green",
        marker="x",
        label="Predicted Anomaly",
        s=50,
    )

    plt.title(
        f"GCT-GAN Anomaly Detection on PSM Dataset (F1-Score: {f1_score:.4f})"
    )
    plt.xlabel("Sequence Window Index")
    plt.ylabel("Combined Anomaly Score")
    plt.legend()
    plt.grid(True, linestyle="--")
    plt.tight_layout()

    path = os.path.join(output_dir, "anomaly_detection_plot.png")
    plt.savefig(path, dpi=150)
    print(f"Saved anomaly detection plot → {path}")

    if show:
        plt.show()
    plt.close()


def plot_loss_history(history, output_dir=None, show=False):
    """
    Plot Generator loss components over training epochs.

    Args:
        history:    List of dicts with keys ``rec_loss``, ``adv_loss``, ``con_loss``.
        output_dir: Directory to save the figure (default: ``config.OUTPUT_DIR``).
        show:       Whether to call ``plt.show()``.
    """
    output_dir = output_dir or config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    loss_df = pd.DataFrame(history)

    plt.figure(figsize=(12, 5))
    plt.plot(loss_df["rec_loss"], label="Reconstruction Loss (REC)")
    plt.plot(loss_df["adv_loss"], label="Adversarial Loss (ADV)")
    plt.plot(loss_df["con_loss"], label="Contrastive Loss (CON)")
    plt.title("Generator Component Losses Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(output_dir, "loss_history.png")
    plt.savefig(path, dpi=150)
    print(f"Saved loss history plot → {path}")

    if show:
        plt.show()
    plt.close()
