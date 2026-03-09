# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Ignacio Garbayo Fernandez

"""
TDD contract tests for src/preprocess.py.

Critical invariant this file protects: NO DATA LEAKAGE.

The sliding window must only look backward, never forward into the future.
The temporal split must never shuffle rows.

These tests are the primary correctness guarantee for the entire project —
a model trained on leaked data would produce artificially inflated metrics
and fail completely in production.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocess import create_sliding_windows, temporal_split

# Parameters used throughout this test module
W = 15   # lookback window
H = 5    # prediction horizon
N = 200  # fixture size


@pytest.fixture(scope="module")
def synthetic_df():
    """
    Minimal deterministic DataFrame sufficient to test windowing logic.

    Deliberately small (200 rows) so assertions can be computed by hand
    and failures are immediately interpretable.

    Known incident window: rows 100–129 (30 rows), so we can write
    exact assertions about which labels should be 0 and which should be 1.

    Example of what this looks like around the incident boundary:
        t=99:  metric=-0.544, is_incident=0
        t=100: metric=0.506,  is_incident=1  ← incident starts here
        t=101: metric=0.988,  is_incident=1
        ...
        t=129: metric=-0.132, is_incident=1
        t=130: metric=-0.663, is_incident=0  ← incident ends here
    """
    rng = np.random.default_rng(42)
    t = np.arange(N)
    metric = np.sin(t / 10.0) + rng.normal(0, 0.1, N)   # deterministic
    is_incident = np.zeros(N, dtype=int)
    is_incident[100:130] = 1                              # one known window
    return pd.DataFrame({"t": t, "metric": metric, "is_incident": is_incident})


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestWindowCount:
    def test_output_length(self, synthetic_df):
        """
        For a DataFrame of length N with parameters W and H:
            valid row count = N - W - H = 200 - 15 - 5 = 180

        This is because we need W rows of history and H rows of future
        for every sample, so the first W and last H rows cannot be used.
        """
        X, y = create_sliding_windows(synthetic_df, W=W, H=H)
        expected = N - W - H
        assert len(X) == expected, f"Expected {expected} rows, got {len(X)}"
        assert len(y) == expected, f"Expected {expected} labels, got {len(y)}"

    def test_feature_width_equals_W(self, synthetic_df):
        """
        Each row of X must contain exactly W features (the raw window values).

        This shape — (n_samples, W) — is what RandomForest expects as tabular input.
        The model learns its own aggregations from the raw values.
        """
        X, y = create_sliding_windows(synthetic_df, W=W, H=H)
        assert X.shape[1] == W, (
            f"Expected {W} features per sample (one per window step), "
            f"got {X.shape[1]}"
        )


# ---------------------------------------------------------------------------
# No data leakage — the most critical invariant
# ---------------------------------------------------------------------------

class TestNoDataLeakage:
    def test_future_spike_does_not_appear_in_features(self, synthetic_df):
        """
        A spike injected at timestep t=100 must NOT change the feature
        vector for any window that ends at or before t=99.

        How the index math works:
            - create_sliding_windows iterates t in range(W, N-H)
            - For t=99: window = metric[99-15 : 99] = metric[84:99]
              → indices 84, 85, ..., 98 (15 values, excludes index 99 and beyond)
            - Output row index for t=99 is: t - W = 99 - 15 = 84

        If metric[100] is changed to 9999 and row 84 of X changes, we have leakage.
        """
        df_with_spike = synthetic_df.copy()
        df_with_spike.loc[100, "metric"] = 9999.0

        X_clean, _ = create_sliding_windows(synthetic_df, W=W, H=H)
        X_spike, _ = create_sliding_windows(df_with_spike, W=W, H=H)

        row_for_t99 = 99 - W   # = 84
        np.testing.assert_array_equal(
            X_clean[row_for_t99],
            X_spike[row_for_t99],
            err_msg=(
                f"Row {row_for_t99} (t=99) changed when spike was added at t=100. "
                "Future values must not appear in the feature window."
            ),
        )

    def test_features_only_use_past_values(self, synthetic_df):
        """
        The feature window for row i must be metric[t-W : t], exclusive of t.

        We verify this directly: the last element of X[i] should equal
        metric[t-1], not metric[t].

        Example for t=W (first valid timestep, t=15):
            X[0] = metric[0:15] → last element = metric[14]
            metric[15] must NOT appear in X[0].
        """
        X, _ = create_sliding_windows(synthetic_df, W=W, H=H)
        metric = synthetic_df["metric"].values

        # For the first output row (t=W=15):
        # X[0] should be metric[0:15], so X[0][-1] == metric[14]
        assert X[0][-1] == pytest.approx(metric[W - 1]), (
            f"Last element of X[0] should be metric[{W-1}]={metric[W-1]:.4f}, "
            f"got {X[0][-1]:.4f}. "
            "Window is using metric[t] instead of metric[t-1] as last element."
        )


# ---------------------------------------------------------------------------
# Label creation correctness
# ---------------------------------------------------------------------------

class TestLabelCreation:
    def test_label_positive_when_incident_in_horizon(self, synthetic_df):
        """
        Label at output row (t - W) should be 1 if ANY of the H future
        timesteps [t, t+H) contains an incident.

        Known incident window: is_incident[100:130] = 1.
        H = 5, so the horizon for t=100 is [100, 105).

        For t=100 (row index = 100-15 = 85):
            future = is_incident[100:105] = [1, 1, 1, 1, 1] → label = 1
        """
        X, y = create_sliding_windows(synthetic_df, W=W, H=H)
        row_t100 = 100 - W   # = 85
        assert y[row_t100] == 1, (
            f"Row {row_t100} (t=100): incident starts at t=100, "
            f"horizon [100, 105) contains incidents. Expected label=1, got {y[row_t100]}."
        )

    def test_label_negative_when_no_incident_in_horizon(self, synthetic_df):
        """
        For t=95 (row index = 95-15 = 80):
            future = is_incident[95:100] = [0, 0, 0, 0, 0] → label = 0

        The incident starts at t=100, which is outside the horizon [95, 100).
        """
        X, y = create_sliding_windows(synthetic_df, W=W, H=H)
        row_t95 = 95 - W   # = 80
        assert y[row_t95] == 0, (
            f"Row {row_t95} (t=95): incident at t=100 is outside "
            f"horizon [95, 100). Expected label=0, got {y[row_t95]}."
        )

    def test_label_boundary_at_incident_start(self, synthetic_df):
        """
        Edge case: t=96 with H=5 means horizon is [96, 101).
        is_incident[100] = 1, which IS within [96, 101). → label = 1.

        This tests the boundary condition:
            last timestep in horizon = t + H - 1 = 96 + 5 - 1 = 100
            is_incident[100] = 1 → label must be 1
        """
        X, y = create_sliding_windows(synthetic_df, W=W, H=H)
        row_t96 = 96 - W   # = 81
        assert y[row_t96] == 1, (
            f"Row {row_t96} (t=96): horizon [96, 101) includes t=100 (incident). "
            f"Expected label=1, got {y[row_t96]}."
        )


# ---------------------------------------------------------------------------
# Temporal split — no shuffle allowed
# ---------------------------------------------------------------------------

class TestTemporalSplit:
    @pytest.fixture(scope="class")
    def windows(self, synthetic_df):
        return create_sliding_windows(synthetic_df, W=W, H=H)

    def test_train_size(self, windows):
        """Train set must be exactly int(n * 0.70) rows."""
        X, y = windows
        X_train, _, _, _ = temporal_split(X, y, train_ratio=0.70)
        expected = int(len(X) * 0.70)
        assert len(X_train) == expected, (
            f"Expected {expected} training rows, got {len(X_train)}"
        )

    def test_test_size(self, windows):
        """Test set must be the remainder after the train split."""
        X, y = windows
        _, X_test, _, _ = temporal_split(X, y, train_ratio=0.70)
        expected = len(X) - int(len(X) * 0.70)
        assert len(X_test) == expected, (
            f"Expected {expected} test rows, got {len(X_test)}"
        )

    def test_no_shuffle_train_then_test(self, windows):
        """
        Concatenating train and test must exactly reconstruct the original X.

        This is the core guarantee: data is split by position in time,
        never by random sampling. If this fails, some test data is from
        before some training data, creating a leakage of the future.
        """
        X, y = windows
        X_train, X_test, y_train, y_test = temporal_split(X, y, train_ratio=0.70)
        np.testing.assert_array_equal(
            np.vstack([X_train, X_test]),
            X,
            err_msg="temporal_split must not reorder rows. "
            "Train must be strictly the first 70%, test the last 30%.",
        )

    def test_no_overlap_between_sets(self, windows):
        """
        The last row of X_train must immediately precede the first row of X_test
        in the original time ordering.

        Example: if n=180, split=126.
            X_train = X[0:126]  → last row is X[125]
            X_test  = X[126:]   → first row is X[126]
        """
        X, y = windows
        X_train, X_test, _, _ = temporal_split(X, y, train_ratio=0.70)
        split = int(len(X) * 0.70)
        np.testing.assert_array_equal(
            X_train[-1], X[split - 1],
            err_msg="Last row of X_train must be X[split-1].",
        )
        np.testing.assert_array_equal(
            X_test[0], X[split],
            err_msg="First row of X_test must be X[split].",
        )
