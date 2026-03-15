# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Ignacio Garbayo Fernandez

"""
StreamPredictor: incremental single-timestep inference for production deployment.

Why this module exists
----------------------
AlertPredictor.predict_proba() accepts a full feature matrix (batch mode).
In production, metrics arrive one timestep at a time. StreamPredictor wraps a
trained AlertPredictor with a rolling buffer of W timesteps, producing a
probability estimate as soon as the buffer is full and updating it on every
new data point.

This addresses the "no streaming" limitation of the batch pipeline: the same
trained model is reused unchanged; only the data ingestion layer differs.

Usage pattern
-------------
    # After training:
    streamer = StreamPredictor(trained_predictor, W=30)

    # At each new 5-minute interval:
    for row in live_feed:
        proba = streamer.step(metric=row['cpu'])
        if proba is not None and proba >= AGGRESSIVE.value:
            fire_alert()

Statistical features
--------------------
If the model was trained with statistical_features=True, pass the same flag
to StreamPredictor so the buffer's feature vector matches the training layout.
"""

from typing import Dict, List, Optional

import numpy as np

from src.model import AlertPredictor
from src.preprocess import _window_stats


class StreamPredictor:
    """
    Wraps a trained AlertPredictor for incremental single-timestep inference.

    Maintains a rolling buffer of the W most recent timesteps. Once the buffer
    is full, every call to step() returns a probability in [0, 1]. Before the
    buffer is full, step() returns None (insufficient history).

    Args:
        predictor:            A trained AlertPredictor instance.
        W:                    Lookback window size — must match what was used
                              during create_sliding_windows at training time.
        feature_cols:         Metric column names in the order used at training.
                              Defaults to ['metric'].
        statistical_features: Set True if the model was trained with
                              statistical_features=True. Appends the same 6
                              per-metric stats to each window vector.

    Example (single metric, no stat features):
        streamer = StreamPredictor(trained_predictor, W=30)
        for cpu_value in live_cpu_stream:
            p = streamer.step(metric=cpu_value)
            if p is not None:
                print(f"Incident probability: {p:.3f}")

    Example (two metrics, with stat features):
        streamer = StreamPredictor(
            trained_predictor, W=30,
            feature_cols=['metric', 'error_rate'],
            statistical_features=True,
        )
        p = streamer.step(metric=0.85, error_rate=0.03)
    """

    def __init__(
        self,
        predictor: AlertPredictor,
        W: int,
        feature_cols: Optional[List[str]] = None,
        statistical_features: bool = False,
    ) -> None:
        if not predictor.is_trained:
            raise RuntimeError(
                "StreamPredictor requires a trained AlertPredictor. "
                "Call predictor.train() first."
            )
        self.predictor = predictor
        self.W = W
        self.feature_cols = feature_cols or ["metric"]
        self.statistical_features = statistical_features
        self._buffer: List[List[float]] = []

    def step(self, **values: float) -> Optional[float]:
        """
        Ingest one new timestep and return an incident probability.

        Args:
            **values: Keyword arguments mapping each feature column name to
                      its current value. Keys must match self.feature_cols.
                      Example: step(metric=0.85) or step(metric=0.85, error_rate=0.02)

        Returns:
            float in [0.0, 1.0] once the buffer contains W timesteps,
            None while the buffer is still filling up (first W-1 calls).

        Raises:
            KeyError: if a required feature column is missing from values.
        """
        row = [float(values[col]) for col in self.feature_cols]
        self._buffer.append(row)

        if len(self._buffer) > self.W:
            self._buffer.pop(0)

        if len(self._buffer) < self.W:
            return None

        raw = np.array(self._buffer)          # shape (W, F)
        window = raw.flatten()

        if self.statistical_features:
            stats = np.concatenate([
                _window_stats(raw[:, f])
                for f in range(raw.shape[1])
            ])
            window = np.concatenate([window, stats])

        return float(self.predictor.predict_proba(window.reshape(1, -1))[0])

    def reset(self) -> None:
        """Clear the rolling buffer. Use when switching between time series."""
        self._buffer.clear()

    @property
    def is_ready(self) -> bool:
        """True once the buffer has accumulated W timesteps."""
        return len(self._buffer) >= self.W

    @property
    def buffer_size(self) -> int:
        """Current number of timesteps in the rolling buffer."""
        return len(self._buffer)
