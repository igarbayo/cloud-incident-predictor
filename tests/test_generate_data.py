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

from src.generate_data import INCIDENT_TYPES

DATA_PATH = Path("data/synthetic_metrics.csv")

EXPECTED_ROWS      = 10_000
INCIDENT_RATE_LOW  = 0.01   # at least 1% of timesteps are incidents
INCIDENT_RATE_HIGH = 0.20   # at most 20% (six types with varied durations)
MIN_INCIDENT_WINDOWS = 5    # at least 5 distinct contiguous incident regions


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
        Must have: t (timestep index), metric (simulated value),
        is_incident (binary label), incident_type (anomaly type string).
        """
        required = {"t", "metric", "metric_2", "is_incident", "incident_type"}
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
            t=42, metric=0.831, is_incident=0, incident_type=''
            t=137, metric=5.921, is_incident=1, incident_type='spike'
        """
        assert pd.api.types.is_integer_dtype(df["is_incident"]), (
            f"Expected integer dtype for 'is_incident', got {df['is_incident'].dtype}"
        )
        unique_vals = set(df["is_incident"].unique())
        assert unique_vals.issubset({0, 1}), (
            f"is_incident must only contain 0 and 1, found: {unique_vals}"
        )

    def test_incident_type_is_string(self, df):
        """
        incident_type must be a string column. Normal timesteps have empty
        string ''; incident timesteps have one of the six known type names.
        """
        # pandas <3.x uses object dtype; pandas ≥3.x infers StringDtype — accept both
        assert df["incident_type"].dtype == object or isinstance(
            df["incident_type"].dtype, pd.StringDtype
        ), f"Expected string-like dtype for 'incident_type', got {df['incident_type'].dtype}"


class TestIncidentRate:
    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_csv(DATA_PATH)

    def test_incident_rate_in_bounds(self, df):
        """
        With six incident types and varied durations, the rate should be
        between 1% and 20% of total timesteps.

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


class TestIncidentTypes:
    @pytest.fixture(scope="class")
    def df(self):
        # Empty strings in incident_type are saved as blank cells in CSV,
        # which pandas reads back as NaN — fill them to restore the original value.
        df = pd.read_csv(DATA_PATH)
        df['incident_type'] = df['incident_type'].fillna('')
        return df

    def test_all_incident_types_present(self, df):
        """
        All six incident types must appear at least once in the dataset.
        If any type is missing, the dataset does not cover the full anomaly
        taxonomy and the model will not learn to detect that pattern.
        """
        observed = set(df.loc[df["incident_type"] != "", "incident_type"].unique())
        expected = set(INCIDENT_TYPES)
        missing = expected - observed
        assert not missing, (
            f"Incident types not present in dataset: {missing}. "
            "Increase n_steps or anomaly_fraction."
        )

    def test_incident_type_only_on_incident_rows(self, df):
        """
        Normal timesteps (is_incident=0) must have incident_type == ''.
        Incident timesteps (is_incident=1) must have a non-empty incident_type.

        This validates that the type label is consistent with the binary label.
        """
        normal_with_type = df[(df["is_incident"] == 0) & (df["incident_type"] != "")]
        assert len(normal_with_type) == 0, (
            f"Found {len(normal_with_type)} normal timesteps with non-empty incident_type."
        )

        incident_without_type = df[(df["is_incident"] == 1) & (df["incident_type"] == "")]
        assert len(incident_without_type) == 0, (
            f"Found {len(incident_without_type)} incident timesteps with empty incident_type."
        )

    def test_incident_type_values_are_valid(self, df):
        """
        Every non-empty incident_type value must be one of the known types.
        Unknown strings indicate a bug in the injection logic.
        """
        valid = set(INCIDENT_TYPES) | {""}
        observed = set(df["incident_type"].unique())
        invalid = observed - valid
        assert not invalid, (
            f"Unknown incident_type values found: {invalid}. "
            f"Valid values are: {valid}"
        )
