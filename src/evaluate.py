"""
Evaluation and anomaly scoring for GCT-GAN.

Computes combined anomaly scores from reconstruction error and discriminator
output, determines a threshold from normal training data, and reports
classification metrics.
"""

import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support

from . import config


def compute_anomaly_scores(generator, discriminator, X, beta=None):
    """
    Compute per-window anomaly scores.

    The score combines reconstruction error and discriminator confidence:

        S(x) = (1 - beta) * MSE(x, x_hat) + beta * (1 - D(x_hat))

    Args:
        generator:     Trained generator model.
        discriminator: Trained discriminator model.
        X:             Input sequences, shape (N, W, F).
        beta:          Weighting factor (default: ``config.BETA``).

    Returns:
        1-D array of anomaly scores.
    """
    beta = beta if beta is not None else config.BETA

    X_reco, _ = generator.predict(X, verbose=0)
    rec_errors = np.mean(np.square(X - X_reco), axis=(1, 2))
    d_scores = discriminator.predict(X_reco, verbose=0).flatten()

    return (1 - beta) * rec_errors + beta * (1 - d_scores)


def evaluate(generator, discriminator, X_train, X_test, y_test,
             beta=None, threshold_pct=None):
    """
    Full evaluation pipeline.

    1. Compute anomaly scores on training data to derive threshold.
    2. Compute anomaly scores on test data.
    3. Classify and report.

    Args:
        generator:     Trained generator.
        discriminator: Trained discriminator.
        X_train:       Normal training sequences.
        X_test:        Test sequences.
        y_test:        Per-window ground-truth labels.
        beta:          Scoring weight (default: ``config.BETA``).
        threshold_pct: Percentile for threshold (default: ``config.THRESHOLD_PERCENTILE``).

    Returns:
        dict with keys:
            - anomaly_scores: 1-D array of test scores
            - y_pred: 1-D int array of predictions
            - threshold: float
            - precision, recall, f1_score: floats
    """
    beta = beta if beta is not None else config.BETA
    threshold_pct = threshold_pct if threshold_pct is not None else config.THRESHOLD_PERCENTILE

    print("\n--- Starting Evaluation on Test Data ---")

    # Threshold from normal training data
    train_scores = compute_anomaly_scores(generator, discriminator, X_train, beta)
    threshold = np.percentile(train_scores, threshold_pct)
    print(f"\nAnomaly threshold ({threshold_pct}th percentile of normal scores): "
          f"{threshold:.6f}")

    # Test scores
    test_scores = compute_anomaly_scores(generator, discriminator, X_test, beta)

    # Classify
    y_pred = (test_scores > threshold).astype(int)

    # Align lengths
    num_pred = len(y_pred)
    y_true = y_test[:num_pred]

    # Report
    print("\n--- Classification Report ---")
    print(
        classification_report(
            y_true, y_pred, target_names=["Normal", "Anomaly"], digits=4
        )
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1
    )
    print(f"Final F1-Score: {f1:.4f}")

    return {
        "anomaly_scores": test_scores,
        "y_pred": y_pred,
        "y_true": y_true,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
