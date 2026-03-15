# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Ignacio Garbayo Fernandez

"""
Sliding-window feature engineering and temporal dataset splitting.

This module implements the core transformation from one or more time series
into a supervised ML dataset using a sliding window formulation.

Mathematical formulation
------------------------
Given a dataset with F metrics of length N, and parameters W (window) and H (horizon):

    For each timestep t in range(W, N - H):
        X[t] = [metric_1[t-W:t], metric_2[t-W:t], ...]  — W * F raw past values
        y[t] = any(incident[t : t+H])                    — 1 if any incident in next H steps

Output shapes:
    X : (N - W - H, W * F)  — feature matrix (one row per sliding window)
    y : (N - W - H,)        — binary labels

Data leakage guarantee
----------------------
The window [t-W : t] is Python slice notation — it is EXCLUSIVE of index t.
The label window [t : t+H] is EXCLUSIVE of index t-1.
These two ranges are disjoint with no overlap, so features and labels never
share timesteps. This guarantee holds for any number of metrics F.

Why raw values instead of pre-computed statistics?
--------------------------------------------------
Each window uses the raw metric values directly as features. This means
X has shape (n_samples, W * F) rather than (n_samples, 7_summary_stats).

Rationale:
  - RandomForest learns its own aggregations (decision trees handle raw
    tabular features naturally via splitting on individual columns).
  - Simpler pipeline = fewer transformation steps that could introduce bugs.
  - Easier to explain: "each column is the metric value k steps ago."
  - No rolling-statistics pre-computation means zero risk of future data
    accidentally leaking into the training set's rolling windows.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Names of the 6 statistical features appended per metric column when
# statistical_features=True is passed to create_sliding_windows.
STAT_FEATURE_NAMES: List[str] = ["mean", "std", "slope", "min", "max", "z_last"]
N_STAT_FEATURES: int = len(STAT_FEATURE_NAMES)


def _window_stats(window_1d: np.ndarray) -> np.ndarray:
    """
    Compute 6 summary statistics for a 1-D window of raw metric values.

    Args:
        window_1d: 1-D array of length W.

    Returns:
        1-D array of length 6: [mean, std, slope, min, max, z_last].

    Notes:
        - slope: linear trend coefficient (units: value per timestep).
        - z_last: z-score of the most recent value; measures how extreme
          the last observation is relative to the window distribution.
          A large positive z_last signals a spike; negative signals a dip.
        - std denominator uses 1e-8 guard to avoid division by zero on
          flat windows (constant metric value).
    """
    mean  = np.mean(window_1d)
    std   = np.std(window_1d)
    slope = np.polyfit(np.arange(len(window_1d)), window_1d, 1)[0]
    min_  = np.min(window_1d)
    max_  = np.max(window_1d)
    z_last = (window_1d[-1] - mean) / (std + 1e-8)
    return np.array([mean, std, slope, min_, max_, z_last], dtype=float)


def create_sliding_windows(
    df: pd.DataFrame,
    W: int = 15,
    H: int = 5,
    feature_cols: list = None,
    statistical_features: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform a time-series DataFrame into (X, y) using a sliding window.

    Supports one or more metric columns (multivariate). Each window flattens
    all metrics into a single feature vector, so the model receives the full
    temporal context of every monitored signal.

    Args:
        df:                   DataFrame with at minimum "is_incident" and the columns
                              listed in feature_cols, sorted in chronological order.
        W:                    Lookback window length. Number of past timesteps used as features.
                              Larger W gives the model more context but requires more data.
        H:                    Prediction horizon. Number of future timesteps to check for incidents.
                              Larger H gives earlier warnings but makes the problem harder.
        feature_cols:         List of metric column names to include as features.
                              Defaults to ['metric'] for single-metric backward compatibility.
                              Pass ['metric', 'metric_2'] for CPU + Memory multivariate mode.
        statistical_features: When True, append 6 summary statistics per metric column
                              (mean, std, slope, min, max, z_last) after the raw window values.
                              Output shape becomes (N-W-H, W*F + 6*F) instead of (N-W-H, W*F).
                              Use STAT_FEATURE_NAMES for human-readable column labels.

    Returns:
        X: np.ndarray of shape (N-W-H, W*F [+ 6*F if statistical_features])
        y: np.ndarray of shape (N-W-H,) — binary incident labels

    Raises:
        ValueError: if df has fewer than W+H+1 rows (no valid windows possible).

    Index arithmetic (critical for understanding the leakage invariant):
        Output row index i corresponds to timestep t = i + W.
        X[i] = [metric_1[i:i+W], metric_2[i:i+W], ...].flatten()
        y[i] = any(incident[i+W : i+W+H])

    Example with W=3, H=2, F=1 on a 7-step series:
        t:       0    1    2    3    4    5    6
        metric: [0.1, 0.2, 0.5, 0.8, 5.2, 5.0, 0.3]
        incident:[0,   0,   0,   0,   1,   1,   0  ]

        i=0 (t=3): X[0]=[0.1,0.2,0.5], y[0]=any([0,1])=1  ← future incident in H
        i=1 (t=4): X[1]=[0.2,0.5,0.8], y[1]=any([1,1])=1
        i=2 (t=5): X[2]=[0.5,0.8,5.2], y[2]=any([1,0])=1
    """
    if feature_cols is None:
        feature_cols = ["metric"]

    n = len(df)
    if n < W + H + 1:
        raise ValueError(
            f"DataFrame too short: need at least {W + H + 1} rows for W={W}, H={H}, "
            f"got {n}."
        )

    features = df[feature_cols].values   # shape (n, F)
    incidents = df["is_incident"].values

    X_list, y_list = [], []

    for t in range(W, n - H):
        # Feature window: W past timesteps for each metric, flattened to 1D.
        # Shape: (W, F) → flatten → (W * F,)
        raw_window = features[t - W : t]          # shape (W, F)
        window = raw_window.flatten()

        if statistical_features:
            # Append 6 stats per metric column: [mean, std, slope, min, max, z_last]
            stats = np.concatenate([
                _window_stats(raw_window[:, f])
                for f in range(raw_window.shape[1])
            ])
            window = np.concatenate([window, stats])

        # Label: 1 if ANY of the next H timesteps contains an incident.
        label = 1 if np.any(incidents[t : t + H] == 1) else 0

        X_list.append(window)
        y_list.append(label)

    return np.array(X_list, dtype=float), np.array(y_list, dtype=int)


def build_feature_names(
    W: int,
    feature_cols: Optional[List[str]] = None,
    statistical_features: bool = False,
) -> List[str]:
    """
    Return human-readable feature names matching the columns of X produced by
    create_sliding_windows with the same parameters.

    Useful for labelling feature-importance plots when statistical_features=True.

    Args:
        W:                    Lookback window size.
        feature_cols:         Metric column names (defaults to ['metric']).
        statistical_features: Whether statistical features were appended.

    Returns:
        List of strings, one per feature column in X.

    Example (W=3, feature_cols=['cpu'], statistical_features=True):
        ['cpu_t-3', 'cpu_t-2', 'cpu_t-1', 'cpu_mean', 'cpu_std',
         'cpu_slope', 'cpu_min', 'cpu_max', 'cpu_z_last']
    """
    if feature_cols is None:
        feature_cols = ["metric"]

    names: List[str] = []
    for col in feature_cols:
        names += [f"{col}_t-{W - i}" for i in range(W)]
    if statistical_features:
        for col in feature_cols:
            names += [f"{col}_{s}" for s in STAT_FEATURE_NAMES]
    return names


def temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split (X, y) into train and test sets using strict temporal ordering.

    Args:
        X:           Feature matrix, rows in chronological order.
        y:           Label array, same ordering as X.
        train_ratio: Fraction of rows assigned to training (default 0.70).

    Returns:
        X_train, X_test, y_train, y_test

    Design decisions:
        - Uses int(n * train_ratio) NOT round() for a deterministic split point.
          This makes the test verifiable: expected = int(n * 0.70).
        - No random_state parameter — there is deliberately no way to shuffle,
          making leakage structurally impossible rather than just unlikely.
        - The 70/30 split preserves at least 30% of the timeline for evaluation,
          which typically spans enough incident windows to compute stable metrics.

    Example:
        If n=9975, split=int(9975*0.70)=6982:
            X_train = X[:6982]  → timesteps 0 to ~t=7000
            X_test  = X[6982:]  → timesteps ~t=7000 to end
        The model is trained on older data and evaluated on newer data,
        exactly as it would be deployed in production.
    """
    n = len(X)
    split = int(n * train_ratio)
    return X[:split], X[split:], y[:split], y[split:]
