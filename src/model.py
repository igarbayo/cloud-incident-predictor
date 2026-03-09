# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Ignacio Garbayo Fernandez

"""
AlertPredictor: domain-aware wrapper around RandomForestClassifier.

Naming philosophy (MDD)
-----------------------
The class is called AlertPredictor, not BinaryClassifier or RFWrapper.
This naming ensures that when engineers read the notebook or production code,
they see business concepts: "the alert predictor fired on this window" rather
than "the classifier predicted class 1."

Why RandomForest?
-----------------
The KISS principle applies here. A Random Forest:
  - Handles tabular data (our W-column feature matrix) natively and well.
  - Requires minimal hyperparameter tuning to achieve good results.
  - Is interpretable via feature_importances (useful for the H2 hypothesis).
  - Is robust to the moderate class imbalance we have (combined with class_weight).
  - Can be explained to a non-ML audience: "many decision trees vote."

Alternatives considered and rejected:
  - LSTM/CNN: requires more data, longer training, harder to explain.
  - Logistic Regression: linear decision boundary may miss non-linear patterns.
  - XGBoost: slightly better performance but more hyperparameters; RF is sufficient
    to demonstrate the problem formulation, which is the evaluation focus.

Why class_weight='balanced'?
----------------------------
With anomaly_fraction=0.02, incidents make up roughly 5-8% of all labels.
Without balancing, a model that always predicts 0 achieves ~93% accuracy —
which is useless for alerting. class_weight='balanced' tells sklearn to give
each sample a weight inversely proportional to its class frequency:
    weight_for_class_c = n_samples / (n_classes * n_samples_in_class_c)
This effectively equalises the contribution of each class to the loss.
"""

import pickle
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class AlertPredictor:
    """
    Binary incident predictor for the sliding-window alerting system.

    Wraps RandomForestClassifier with:
      - A domain-aware interface (predict_proba returns 1D positive-class array).
      - A guard against accidental use before training.
      - Simple pickle-based persistence for deployment scenarios.

    Typical usage:
        predictor = AlertPredictor()
        predictor.train(X_train, y_train)

        # Get probability scores for threshold analysis
        y_proba = predictor.predict_proba(X_test)

        # Apply a business threshold (from AlertThreshold domain object)
        from src import AGGRESSIVE
        alerts = predictor.predict(X_test, threshold=AGGRESSIVE.value)
    """

    def __init__(self) -> None:
        self.clf = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
        )
        self.is_trained: bool = False

    def train(self, X: np.ndarray, y: np.ndarray) -> "AlertPredictor":
        """
        Fit the classifier on training data.

        Args:
            X: Feature matrix of shape (n_samples, W).
               Each row is one sliding window of raw metric values.
            y: Binary label array of shape (n_samples,).
               1 = an incident will occur in the next H timesteps, 0 = no incident.

        Returns:
            self (for optional method chaining).

        Example:
            predictor = AlertPredictor()
            predictor.train(X_train, y_train)
            print(f"Trained on {len(X_train)} windows, {y_train.sum()} positive labels.")
        """
        self.clf.fit(X, y)
        self.is_trained = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return the probability of an incident in the next H timesteps.

        The underlying sklearn method returns a (n, 2) matrix — column 0 is
        the probability of no incident, column 1 is the probability of an incident.
        We return only column 1 (the positive class) as a 1D array.

        This simplifies threshold application downstream:
            alerts = proba >= threshold.value   (no indexing needed)

        Args:
            X: Feature matrix of shape (n_samples, W).

        Returns:
            1D array of shape (n_samples,) with values in [0.0, 1.0].

        Raises:
            RuntimeError: if called before train().

        Example:
            y_proba = predictor.predict_proba(X_test)
            # y_proba might look like: [0.04, 0.87, 0.12, 0.91, 0.03]
            # Values near 1.0 indicate high confidence of an upcoming incident.
        """
        if not self.is_trained:
            raise RuntimeError(
                "AlertPredictor.train() must be called before predict_proba(). "
                "Did you forget to train the model?"
            )
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Apply a probability threshold to produce binary alert decisions.

        Args:
            X:         Feature matrix of shape (n_samples, W).
            threshold: Probability cutoff. Use AlertThreshold.value from src/__init__.py
                       to ensure the threshold has business meaning attached to it.
                       Default 0.5 (balanced precision/recall).

        Returns:
            Binary array of shape (n_samples,): 1 = fire alert, 0 = no alert.

        Example:
            from src import AGGRESSIVE, CONSERVATIVE

            # High-recall mode: alert often, miss fewer incidents
            alerts_aggressive = predictor.predict(X_test, threshold=AGGRESSIVE.value)

            # High-precision mode: alert rarely, fewer false positives
            alerts_conservative = predictor.predict(X_test, threshold=CONSERVATIVE.value)
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str) -> None:
        """
        Persist the trained classifier to disk using pickle.

        Args:
            path: File path for the serialised model (e.g. "models/predictor.pkl").

        Note: sklearn models are safe to pickle for same-version deployment.
        For cross-version compatibility, consider joblib or ONNX.

        Example:
            predictor.save("models/alert_predictor_v1.pkl")
        """
        with open(path, "wb") as f:
            pickle.dump(self.clf, f)

    def load(self, path: str) -> "AlertPredictor":
        """
        Load a previously saved classifier from disk.

        Args:
            path: File path to the serialised model.

        Returns:
            self (for optional method chaining).

        Example:
            predictor = AlertPredictor()
            predictor.load("models/alert_predictor_v1.pkl")
            alerts = predictor.predict(X_live)
        """
        with open(path, "rb") as f:
            self.clf = pickle.load(f)
        self.is_trained = True
        return self

    @property
    def feature_importances(self) -> np.ndarray:
        """
        Mean impurity decrease for each of the W features across all trees.

        Returns:
            1D array of shape (W,). Values sum to 1.0.
            Higher value = this time step's raw value is more useful for prediction.

        Used in the notebook's H2 hypothesis test:
            "Do earlier timesteps in the window contribute more than recent ones?"

        Example interpretation:
            importances = predictor.feature_importances
            # importances[0] = contribution of metric[t-W] (oldest step)
            # importances[-1] = contribution of metric[t-1] (most recent step)
            # If importances[-1] >> importances[0], recent values dominate.

        Raises:
            RuntimeError: if called before train().
        """
        if not self.is_trained:
            raise RuntimeError(
                "feature_importances requires a trained model. "
                "Call train() first."
            )
        return self.clf.feature_importances_
