# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Ignacio Garbayo Fernandez

"""
Sliding-window feature engineering and temporal dataset splitting.

This module implements the core transformation from a 1D time series into
a supervised ML dataset using a sliding window formulation.

Mathematical formulation
------------------------
Given a time series of length N with parameters W (window) and H (horizon):

    For each timestep t in range(W, N - H):
        X[t] = metric[t-W : t]       — W raw past values (excludes t)
        y[t] = any(incident[t : t+H]) — 1 if any incident in next H steps

Output shapes:
    X : (N - W - H, W)   — feature matrix (one row per sliding window)
    y : (N - W - H,)     — binary labels

Data leakage guarantee
----------------------
The window [t-W : t] is Python slice notation — it is EXCLUSIVE of index t.
The label window [t : t+H] is EXCLUSIVE of index t-1.
These two ranges are disjoint with no overlap, so features and labels never
share timesteps.

Why raw values instead of pre-computed statistics?
--------------------------------------------------
Each window uses the raw metric values directly as features. This means
X has shape (n_samples, W) rather than (n_samples, 7_summary_stats).

Rationale:
  - RandomForest learns its own aggregations (decision trees handle raw
    tabular features naturally via splitting on individual columns).
  - Simpler pipeline = fewer transformation steps that could introduce bugs.
  - Easier to explain: "each column is the metric value k steps ago."
  - No rolling-statistics pre-computation means zero risk of future data
    accidentally leaking into the training set's rolling windows.
"""

from typing import Tuple

import numpy as np
import pandas as pd


def create_sliding_windows(
    df: pd.DataFrame,
    W: int = 15,
    H: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform a time-series DataFrame into (X, y) using a sliding window.

    Args:
        df: DataFrame with at minimum columns "metric" and "is_incident",
            rows sorted in chronological order (ascending t).
        W:  Lookback window length. Number of past timesteps used as features.
            Larger W gives the model more context but requires more data.
        H:  Prediction horizon. Number of future timesteps to check for incidents.
            Larger H gives earlier warnings but makes the problem harder.

    Returns:
        X: np.ndarray of shape (N-W-H, W) — feature matrix (raw window values)
        y: np.ndarray of shape (N-W-H,)   — binary incident labels

    Raises:
        ValueError: if df has fewer than W+H+1 rows (no valid windows possible).

    Index arithmetic (critical for understanding the leakage invariant):
        Output row index i corresponds to timestep t = i + W.
        X[i] = metric[i : i+W]      → past W values ending at t-1
        y[i] = any(incident[i+W : i+W+H])  → future H values starting at t

    Example with W=3, H=2 on a 7-step series:
        t:       0    1    2    3    4    5    6
        metric: [0.1, 0.2, 0.5, 0.8, 5.2, 5.0, 0.3]
        incident:[0,   0,   0,   0,   1,   1,   0  ]

        i=0 (t=3): X[0]=[0.1,0.2,0.5], y[0]=any([0,1])=1  ← future incident in H
        i=1 (t=4): X[1]=[0.2,0.5,0.8], y[1]=any([1,1])=1
        i=2 (t=5): X[2]=[0.5,0.8,5.2], y[2]=any([1,0])=1  ← NOTE: this row sees
                   the spike in its features (t=4 is in [2:5]), showing the model
                   learns to recognise pre-incident patterns.
    """
    n = len(df)
    if n < W + H + 1:
        raise ValueError(
            f"DataFrame too short: need at least {W + H + 1} rows for W={W}, H={H}, "
            f"got {n}."
        )

    metric = df["metric"].values
    incidents = df["is_incident"].values

    X_list, y_list = [], []

    for t in range(W, n - H):
        # Feature window: W past values ending at t-1 (excludes t itself)
        window = metric[t - W : t]

        # Label: 1 if ANY of the next H timesteps contains an incident
        # Using np.any() implements "aggressive labeling" — the model is asked
        # to warn before any incident within the horizon, not just at exact onset.
        label = 1 if np.any(incidents[t : t + H] == 1) else 0

        X_list.append(window)
        y_list.append(label)

    return np.array(X_list, dtype=float), np.array(y_list, dtype=int)


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
