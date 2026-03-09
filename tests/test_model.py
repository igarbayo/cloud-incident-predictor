# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Ignacio Garbayo Fernandez

"""
TDD contract tests for src/model.py.

These tests are narrow by design: we are not testing scikit-learn internals,
only that our AlertPredictor wrapper behaves correctly and exposes the
right interface.

The goal is to verify the contract:
  1. The model has the correct, reproducible hyperparameters.
  2. predict_proba returns a 1D probability array in [0, 1].
  3. A saved model produces identical predictions after loading.
"""

import numpy as np
import pytest

from src.model import AlertPredictor

N_SAMPLES = 300
N_FEATURES = 15   # matches default W=15
RANDOM_SEED = 42


@pytest.fixture(scope="module")
def dummy_data():
    """
    Minimal random dataset sufficient to train and evaluate the wrapper.

    Uses a fixed seed for reproducibility. The data has no real structure —
    we are only testing the wrapper's interface, not the model's predictive power.

    Example shape:
        X_train: (210, 15)
        X_test:  (90, 15)
        y_train: mix of 0s and 1s
    """
    rng = np.random.default_rng(RANDOM_SEED)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y = rng.integers(0, 2, size=N_SAMPLES)
    split = int(N_SAMPLES * 0.70)
    return X[:split], X[split:], y[:split], y[split:]


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

class TestAlertPredictorHyperparameters:
    def test_n_estimators(self):
        """100 trees — enough for stability without being expensive."""
        predictor = AlertPredictor()
        assert predictor.clf.n_estimators == 100

    def test_class_weight_balanced(self):
        """
        'balanced' is the design choice for imbalanced incident data.

        Without this, a model trained on 98% non-incident data can achieve
        98% accuracy by always predicting 0 — useless for alerting.
        class_weight='balanced' penalises misclassifying the rare class more.
        """
        predictor = AlertPredictor()
        assert predictor.clf.class_weight == "balanced"

    def test_random_state_reproducible(self):
        """
        Fixed random_state=42 ensures results are reproducible across runs.

        Example: two analysts running the same notebook should see the same
        PR-AUC and the same confusion matrix values.
        """
        predictor = AlertPredictor()
        assert predictor.clf.random_state == 42


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestAlertPredictorTraining:
    def test_train_completes_without_error(self, dummy_data):
        X_train, _, y_train, _ = dummy_data
        predictor = AlertPredictor()
        predictor.train(X_train, y_train)   # must not raise

    def test_is_trained_flag_set_after_train(self, dummy_data):
        X_train, _, y_train, _ = dummy_data
        predictor = AlertPredictor()
        assert not predictor.is_trained
        predictor.train(X_train, y_train)
        assert predictor.is_trained

    def test_predict_proba_raises_if_not_trained(self, dummy_data):
        """
        Calling predict_proba before train must raise RuntimeError, not silently
        return garbage values. This prevents accidental use of an untrained model.
        """
        _, X_test, _, _ = dummy_data
        predictor = AlertPredictor()
        with pytest.raises(RuntimeError, match="train"):
            predictor.predict_proba(X_test)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class TestAlertPredictorPrediction:
    @pytest.fixture(scope="class")
    def trained_predictor(self, dummy_data):
        X_train, _, y_train, _ = dummy_data
        p = AlertPredictor()
        p.train(X_train, y_train)
        return p

    def test_predict_proba_shape(self, trained_predictor, dummy_data):
        """
        predict_proba must return a 1D array — one probability per sample.

        The underlying sklearn method returns a (n, 2) matrix (one column per class).
        AlertPredictor extracts column 1 (positive class = incident) to simplify
        downstream threshold comparisons: `proba >= threshold.value`.
        """
        _, X_test, _, _ = dummy_data
        proba = trained_predictor.predict_proba(X_test)
        assert proba.ndim == 1, (
            f"predict_proba must return 1D array, got shape {proba.shape}"
        )
        assert len(proba) == len(X_test)

    def test_predict_proba_in_unit_interval(self, trained_predictor, dummy_data):
        """
        All probabilities must be in [0.0, 1.0].

        Example output for 5 samples:
            [0.12, 0.87, 0.45, 0.03, 0.91]
            → alert fires where value >= threshold (e.g., 0.5)
            → alerts at positions 1 and 4
        """
        _, X_test, _, _ = dummy_data
        proba = trained_predictor.predict_proba(X_test)
        assert np.all(proba >= 0.0), f"Found probabilities below 0: {proba.min()}"
        assert np.all(proba <= 1.0), f"Found probabilities above 1: {proba.max()}"

    def test_predict_binary_output(self, trained_predictor, dummy_data):
        """predict() must return only 0s and 1s."""
        _, X_test, _, _ = dummy_data
        preds = trained_predictor.predict(X_test, threshold=0.5)
        assert set(np.unique(preds)).issubset({0, 1})


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestAlertPredictorPersistence:
    def test_save_and_load_roundtrip(self, tmp_path, dummy_data):
        """
        A saved and reloaded model must produce byte-identical predictions.

        This is the deployment guarantee: the model trained offline can be
        loaded into a production alerting service and behave identically.

        Uses pytest's tmp_path fixture to avoid polluting the filesystem.
        """
        X_train, X_test, y_train, _ = dummy_data

        predictor = AlertPredictor()
        predictor.train(X_train, y_train)
        proba_before = predictor.predict_proba(X_test)

        model_path = tmp_path / "alert_predictor.pkl"
        predictor.save(str(model_path))

        loaded = AlertPredictor()
        loaded.load(str(model_path))
        proba_after = loaded.predict_proba(X_test)

        np.testing.assert_array_equal(
            proba_before,
            proba_after,
            err_msg="Loaded model predictions differ from original. "
            "Serialisation is not deterministic.",
        )

    def test_loaded_model_is_trained(self, tmp_path, dummy_data):
        """is_trained flag must be True after loading a saved model."""
        X_train, _, y_train, _ = dummy_data
        predictor = AlertPredictor()
        predictor.train(X_train, y_train)
        predictor.save(str(tmp_path / "model.pkl"))

        loaded = AlertPredictor()
        assert not loaded.is_trained
        loaded.load(str(tmp_path / "model.pkl"))
        assert loaded.is_trained
