# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Ignacio Garbayo Fernandez

"""
Domain models for the cloud incident alerting system.

These dataclasses define the shared business vocabulary used across all modules,
tests, and the notebook. Importing from this package means code reads in
business terms, not ML framework terms.

MDD (Model-Driven Development) foundation: the domain model is defined once
here and consumed everywhere else. If the business meaning of a threshold
changes, this is the single place to update it.
"""

from dataclasses import dataclass


@dataclass
class AlertThreshold:
    """
    Business concept: the probability cutoff above which the model fires an alert.

    Encodes the precision/recall trade-off explicitly in business language.
    A lower value means more alerts (higher recall, lower precision).
    A higher value means fewer alerts (lower recall, higher precision).

    Example usage:
        # A critical payment system where every incident must be caught
        threshold = AGGRESSIVE
        alerts = predictor.predict(X_test, threshold=threshold.value)

        # A low-priority monitoring system where alert fatigue is a concern
        threshold = CONSERVATIVE
        alerts = predictor.predict(X_test, threshold=threshold.value)
    """

    value: float
    label: str
    description: str


@dataclass
class AlertPolicy:
    """
    Business concept: an operational policy defined by a threshold and its cost model.

    The ratio missed_incident_cost / false_alert_cost determines which threshold
    is operationally preferred. For example:
      - ratio > 5  → prefer AGGRESSIVE  (catching incidents outweighs alert fatigue)
      - ratio ~ 1  → prefer BALANCED    (equal weight to both error types)
      - ratio < 1  → prefer CONSERVATIVE (avoiding false alerts is the priority)

    Example usage:
        # Hospital monitoring system — missing an incident is catastrophic
        policy = AlertPolicy(
            threshold=AGGRESSIVE,
            missed_incident_cost=100.0,  # patient harm
            false_alert_cost=1.0,        # nurse checks a false alarm
        )

        # Internal dev tool — incidents are annoying but not critical
        policy = AlertPolicy(
            threshold=CONSERVATIVE,
            missed_incident_cost=2.0,
            false_alert_cost=1.0,
        )
    """

    threshold: AlertThreshold
    missed_incident_cost: float
    false_alert_cost: float

    @property
    def cost_ratio(self) -> float:
        """How many false alerts is one missed incident worth tolerating."""
        return self.missed_incident_cost / self.false_alert_cost


# ---------------------------------------------------------------------------
# Canonical threshold instances — single source of truth for business policies
# ---------------------------------------------------------------------------

AGGRESSIVE = AlertThreshold(
    value=0.3,
    label="aggressive",
    description="High recall — catches most incidents, accepts more false alerts. "
    "Use for critical systems where missing an incident is unacceptable.",
)

BALANCED = AlertThreshold(
    value=0.5,
    label="balanced",
    description="Equal weight to precision and recall. "
    "A reasonable default when incident and false-alert costs are similar.",
)

CONSERVATIVE = AlertThreshold(
    value=0.8,
    label="conservative",
    description="High precision — only alerts when highly confident. "
    "Use when alert fatigue is a concern and incidents are recoverable.",
)

CANONICAL_THRESHOLDS = [AGGRESSIVE, BALANCED, CONSERVATIVE]
