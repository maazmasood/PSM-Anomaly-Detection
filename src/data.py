"""
Data loading, preprocessing, and windowing for PSM anomaly detection.

Handles CSV ingestion, cleaning, normalization, and sliding-window
segmentation for both training and test data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from . import config


def preprocess_data(df, label_df=None):
    """
    Clean feature data and optionally align with anomaly labels.

    Args:
        df: Feature DataFrame (raw CSV).
        label_df: Optional label DataFrame for test data.

    Returns:
        Tuple of (feature_array, labels_array_or_None).
    """
    # Drop timestamp and any non-numeric columns
    df = df.drop(columns=["timestamp"], errors="ignore")

    # Fill NaN values with forward/backward fill
    df = df.ffill().bfill()

    labels = None
    if label_df is not None:
        # Check and drop header row ('timestamp_(min)')
        if label_df.iloc[0, 0] == "timestamp_(min)":
            label_df = label_df.iloc[1:].copy()

        # Convert labels to numeric, drop NaNs
        labels_series = pd.to_numeric(label_df.iloc[:, 0], errors="coerce")
        labels_clean = labels_series.dropna().values.flatten()

        # Align feature length and label length
        min_len = min(len(df), len(labels_clean))
        df = df.iloc[:min_len]
        labels = labels_clean[:min_len]

        # Cast to integer and enforce binary (0 or 1)
        labels = np.clip(labels.astype(np.int32), 0, 1)

    return df.values, labels


def create_sequences(data, window_size, stride):
    """
    Segment a time series into overlapping windows.

    Args:
        data: 2-D array of shape (timesteps, features).
        window_size: Length of each window.
        stride: Step between consecutive windows.

    Returns:
        3-D NumPy array of shape (num_windows, window_size, features).
    """
    sequences = []
    for i in range(0, len(data) - window_size + 1, stride):
        sequences.append(data[i : i + window_size])
    return np.array(sequences)


def align_labels(labels, window_size, stride):
    """
    Map per-timestep labels to per-window labels.

    A window is considered anomalous if *any* timestep within it is anomalous.

    Args:
        labels: 1-D array of per-timestep labels.
        window_size: Length of each window.
        stride: Step between consecutive windows.

    Returns:
        1-D integer array of per-window labels.
    """
    window_labels = []
    for i in range(0, len(labels) - window_size + 1, stride):
        window_labels.append(np.max(labels[i : i + window_size]))
    return np.array(window_labels, dtype=np.int32)


def load_and_prepare_data(
    train_file=None,
    test_file=None,
    label_file=None,
    window_size=None,
    stride=None,
):
    """
    Full data pipeline: load CSVs → preprocess → scale → window → align labels.

    All arguments default to their config values if not provided.

    Returns:
        dict with keys:
            - X_train: float32 array (num_train_windows, window_size, n_features)
            - X_test:  float32 array (num_test_windows, window_size, n_features)
            - y_test:  int32 array   (num_test_windows,)
            - n_features: int
            - scaler: fitted MinMaxScaler
    """
    train_file = train_file or config.TRAIN_FILE
    test_file = test_file or config.TEST_FILE
    label_file = label_file or config.LABEL_FILE
    window_size = window_size or config.WINDOW_SIZE
    stride = stride or config.STRIDE

    print("Loading PSM data...")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    test_label_df = pd.read_csv(label_file, header=None)

    # Preprocess
    train_data, _ = preprocess_data(train_df)
    test_data, test_labels = preprocess_data(test_df, test_label_df)

    n_features = train_data.shape[1]
    print(f"Number of features: {n_features}")

    # Normalize
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # Window
    X_train = create_sequences(train_scaled, window_size, stride).astype(np.float32)
    X_test = create_sequences(test_scaled, window_size, stride).astype(np.float32)
    y_test = align_labels(test_labels, window_size, stride)

    print(f"Training sequences shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"Test sequences shape:     {X_test.shape}, dtype: {X_test.dtype}")
    print(f"Test labels dtype: {y_test.dtype}, unique classes: {np.unique(y_test)}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "n_features": n_features,
        "scaler": scaler,
    }
