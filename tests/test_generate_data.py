# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Ignacio Garbayo Fernandez

"""
TDD contract tests for src/generate_data.py.

These tests are written BEFORE the implementation exists (red-green cycle).
They define what "correct output" means for the data generation step.

Note: TestCSVExists will fail until you run `python src/generate_data.py` first.
This is intentional — TDD tests fail red until the implementation exists.
"""

import pytest
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/synthetic_metrics.csv")

EXPECTED_ROWS = 10_000
INCIDENT_RATE_LOW = 0.01   # at least 1% of timesteps are incidents
INCIDENT_RATE_HIGH = 0.15  # at most 15% (point anomalies with short duration)
MIN_INCIDENT_WINDOWS = 5   # at least 5 distinct contiguous incident regions


class TestCSVExists:
    def test_file_exists(self):
        """
        Precondition: run `python src/generate_data.py` before running tests.

        Example: after running the script you should see:
            data/synthetic_metrics.csv  (around 500 KB)
        """
        assert DATA_PATH.exists(), (
            f"Expected {DATA_PATH} to exist. "
            "Run `python src/generate_data.py` first."
        )


class TestCSVShape:
    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_csv(DATA_PATH)

    def test_row_count(self, df):
        """Exactly 10,000 rows — one per simulated timestep."""
        assert len(df) == EXPECTED_ROWS, (
            f"Expected {EXPECTED_ROWS} rows, got {len(df)}"
        )

    def test_required_columns(self, df):
        """
        Must have at minimum: t (timestep index), metric (simulated value),
        is_incident (binary label).
        """
        required = {"t", "metric", "is_incident"}
        missing = required - set(df.columns)
        assert not missing, f"Missing columns: {missing}"


class TestDtypes:
    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_csv(DATA_PATH)

    def test_metric_is_numeric(self, df):
        """metric column must be float (sinusoidal signal + noise)."""
        assert pd.api.types.is_float_dtype(df["metric"]), (
            f"Expected float dtype for 'metric', got {df['metric'].dtype}"
        )

    def test_is_incident_is_binary_integer(self, df):
        """
        is_incident must be integer 0/1, not float, not string.

        Example of a correct row:
            t=42, metric=0.831, is_incident=0
            t=137, metric=5.921, is_incident=1   ← anomaly injected here
        """
        assert pd.api.types.is_integer_dtype(df["is_incident"]), (
            f"Expected integer dtype for 'is_incident', got {df['is_incident'].dtype}"
        )
        unique_vals = set(df["is_incident"].unique())
        assert unique_vals.issubset({0, 1}), (
            f"is_incident must only contain 0 and 1, found: {unique_vals}"
        )


class TestIncidentRate:
    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_csv(DATA_PATH)

    def test_incident_rate_in_bounds(self, df):
        """
        With anomaly_fraction=0.02 and duration 1-5 steps, the rate should
        be between 1% and 15% of total timesteps.

        A rate outside this range means the synthetic data does not represent
        a realistic rare-event scenario — either too many or too few incidents
        for the classifier to learn from.
        """
        rate = df["is_incident"].mean()
        assert INCIDENT_RATE_LOW <= rate <= INCIDENT_RATE_HIGH, (
            f"Incident rate {rate:.2%} outside "
            f"[{INCIDENT_RATE_LOW:.0%}, {INCIDENT_RATE_HIGH:.0%}]. "
            "Check anomaly_fraction and duration parameters."
        )

    def test_anomaly_windows_are_contiguous(self, df):
        """
        Anomaly regions must appear as contiguous blocks, not scattered single rows.
        We validate by counting 0→1 transitions in is_incident.

        Example of a valid incident window (t=100 to t=103):
            t=99:  is_incident=0
            t=100: is_incident=1  ← transition here
            t=101: is_incident=1
            t=102: is_incident=1
            t=103: is_incident=1
            t=104: is_incident=0
        """
        incidents = df["is_incident"].values
        transitions = sum(
            1 for i in range(1, len(incidents))
            if incidents[i] == 1 and incidents[i - 1] == 0
        )
        assert transitions >= MIN_INCIDENT_WINDOWS, (
            f"Expected at least {MIN_INCIDENT_WINDOWS} incident windows "
            f"(0→1 transitions), found {transitions}."
        )
