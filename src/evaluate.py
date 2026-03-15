# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Ignacio Garbayo Fernandez

"""
Evaluation functions with business-layer interpretation.

Design philosophy
-----------------
All public functions accept AlertThreshold domain objects (from src/__init__.py),
not raw floats. This forces callers to think in business terms:
  - "Which AlertPolicy am I evaluating?" rather than "What is the threshold float?"

All plot functions return the matplotlib Figure object so the notebook can
call plt.show() or fig.savefig() without re-running the computation.

Why PR-AUC and not ROC-AUC?
----------------------------
In imbalanced datasets (like ours, with ~5-8% incident rate), ROC-AUC is
misleading. A model that predicts "no incident" for everything achieves:
  - Accuracy: ~93% (useless)
  - ROC-AUC:  ~0.5 (appears mediocre but it's the random baseline)
  - PR-AUC:   ~0.05 (correctly identified as terrible — matches incident rate)

PR-AUC's random baseline equals the positive class rate, making it easy to
tell if the model adds any value at all.

Why n_alerts in the threshold sweep?
-------------------------------------
Precision and recall measure quality, but operational cost is partly
determined by volume. A threshold that fires 500 alerts per day may be
technically correct (high recall) but operationally untenable (alert fatigue).
Including n_alerts makes the cost model explicit in the evaluation output.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)

from src import AlertThreshold, CANONICAL_THRESHOLDS


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Precision-Recall Curve",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot the full precision-recall curve with PR-AUC annotation.

    The curve traces precision and recall at every possible threshold
    from 0 to 1. A perfect classifier reaches the top-right corner (1,1).
    The baseline (random classifier) is a horizontal line at the incident rate.

    Args:
        y_true:  Binary ground-truth labels.
        y_proba: Predicted probabilities for the positive class (1D array).
        title:   Plot title.
        ax:      Optional existing Axes to plot on (useful for subplots).

    Returns:
        matplotlib Figure containing the plot.

    Example output you should expect:
        PR-AUC = 0.72 (if 0.72 > incident_rate ≈ 0.06, the model is useful)
    """
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
    auc = average_precision_score(y_true, y_proba)
    baseline = y_true.mean()   # random classifier baseline = incident rate

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    ax.plot(recall_vals, precision_vals, lw=2, label=f"AlertPredictor (PR-AUC = {auc:.3f})")
    ax.axhline(y=baseline, color="grey", linestyle="--", lw=1,
               label=f"Random baseline (PR-AUC = {baseline:.3f})")
    ax.set_xlabel("Recall  (fraction of real incidents that triggered an alert)")
    ax.set_ylabel("Precision  (fraction of alerts that were real incidents)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)

    return fig


def threshold_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[List[AlertThreshold]] = None,
) -> Dict[str, dict]:
    """
    Evaluate model performance at each AlertThreshold.

    Args:
        y_true:     Binary ground-truth labels.
        y_proba:    Predicted probabilities for the positive class.
        thresholds: List of AlertThreshold domain objects to evaluate.
                    Defaults to CANONICAL_THRESHOLDS [AGGRESSIVE, BALANCED, CONSERVATIVE].

    Returns:
        Dict keyed by threshold label, each value containing:
            threshold:   float  — the probability cutoff
            description: str    — business meaning
            precision:   float  — fraction of alerts that were real incidents
            recall:      float  — fraction of real incidents that triggered an alert
            f1:          float  — harmonic mean of precision and recall
            n_alerts:    int    — total number of alerts fired on y_true
            n_incidents: int    — total number of real incidents in y_true

    Example interpretation of results:
        "aggressive": precision=0.42, recall=0.91, n_alerts=1842
            → Catches 91% of incidents but 58% of alerts are false alarms.
            → 1842 alerts over the test period might be operationally acceptable
              for a critical system (e.g. payment processing), but not for a
              low-priority internal tool.

        "conservative": precision=0.88, recall=0.31, n_alerts=298
            → Only 298 alerts, 88% of them real, but misses 69% of incidents.
            → Suitable only if incidents are recoverable and alert fatigue is severe.
    """
    if thresholds is None:
        thresholds = CANONICAL_THRESHOLDS

    results: Dict[str, dict] = {}
    n_incidents = int(y_true.sum())

    for t in thresholds:
        y_pred = (y_proba >= t.value).astype(int)
        n_alerts = int(y_pred.sum())

        # Handle the edge case where no alerts are fired (precision undefined)
        prec = precision_score(y_true, y_pred, zero_division=0.0)
        rec  = recall_score(y_true, y_pred, zero_division=0.0)
        f1   = f1_score(y_true, y_pred, zero_division=0.0)

        results[t.label] = {
            "threshold":   t.value,
            "description": t.description,
            "precision":   round(prec, 4),
            "recall":      round(rec, 4),
            "f1":          round(f1, 4),
            "n_alerts":    n_alerts,
            "n_incidents": n_incidents,
        }

    return results


def plot_threshold_comparison(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[List[AlertThreshold]] = None,
) -> plt.Figure:
    """
    Bar chart comparing precision and recall across multiple AlertThresholds.

    Shows precision (blue) and recall (orange) side by side for each threshold,
    annotated with the number of alerts fired. This makes the precision/recall
    trade-off and operational cost visible at a glance.

    Args:
        y_true:     Binary ground-truth labels.
        y_proba:    Predicted probabilities for the positive class.
        thresholds: AlertThreshold instances to compare.

    Returns:
        matplotlib Figure.

    How to read this chart:
        - Blue bar high + orange bar low = conservative (few alerts, misses incidents)
        - Orange bar high + blue bar low = aggressive (catches incidents, noisy)
        - Equal bars = balanced
        - n_alerts annotation shows the operational volume at each policy.
    """
    if thresholds is None:
        thresholds = CANONICAL_THRESHOLDS

    sweep = threshold_sweep(y_true, y_proba, thresholds)
    labels = [t.label for t in thresholds]
    precisions = [sweep[l]["precision"] for l in labels]
    recalls    = [sweep[l]["recall"]    for l in labels]
    n_alerts   = [sweep[l]["n_alerts"]  for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_p = ax.bar(x - width / 2, precisions, width, label="Precision", color="steelblue")
    bars_r = ax.bar(x + width / 2, recalls,    width, label="Recall",    color="darkorange")

    # Annotate with n_alerts above each group
    for i, (p_bar, r_bar, n) in enumerate(zip(bars_p, bars_r, n_alerts)):
        mid_x = (p_bar.get_x() + r_bar.get_x() + r_bar.get_width()) / 2
        ax.text(mid_x, max(p_bar.get_height(), r_bar.get_height()) + 0.03,
                f"{n} alerts", ha="center", va="bottom", fontsize=9, color="dimgray")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(threshold={sweep[l]['threshold']})" for l in labels])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Precision vs Recall by Alert Policy")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    return fig


def plot_feature_importances(
    importances: np.ndarray,
    W: int,
    title: str = "Feature Importances by Window Position",
    feature_names: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Bar chart of feature importances by position in the sliding window.

    Each bar represents one feature. By default (feature_names=None), features
    are labelled as raw timestep positions: t-W (oldest) to t-1 (most recent).
    When statistical_features=True was used in create_sliding_windows, pass the
    extended feature name list so the stat features are labelled correctly.

    Args:
        importances:   Array from AlertPredictor.feature_importances.
                       Shape (W,) for raw features, (W + 6,) with stat features.
        W:             Lookback window size (used only when feature_names is None).
        title:         Plot title.
        feature_names: Optional list of strings, one per feature. When provided,
                       len(feature_names) must equal len(importances).
                       Build with src.preprocess.build_feature_names() for the
                       standard raw+stats layout.

    Returns:
        matplotlib Figure.
    """
    n = len(importances)
    positions = np.arange(n)

    if feature_names is not None:
        labels = feature_names
    else:
        labels = [f"t-{W - i}" for i in range(W)]   # t-W (oldest) to t-1 (newest)

    fig, ax = plt.subplots(figsize=(max(10, n // 2), 4))
    ax.bar(positions, importances, color="steelblue", edgecolor="white")
    ax.set_xticks(positions[::max(1, n // 15)])
    ax.set_xticklabels(labels[::max(1, n // 15)], rotation=45, ha="right")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean impurity decrease (importance)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    return fig


def print_classification_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: AlertThreshold,
) -> None:
    """
    Print sklearn's classification report for a specific AlertThreshold.

    Prefixes the report with the threshold's business description so the
    output is self-documenting when saved to a notebook cell.

    Args:
        y_true:    Binary ground-truth labels.
        y_proba:   Predicted probabilities for the positive class.
        threshold: AlertThreshold domain object.

    Example output:
        ════════════════════════════════════════════════════════════
        Policy: aggressive (threshold=0.3)
        High recall — catches most incidents, accepts more false alerts.
        Use for critical systems where missing an incident is unacceptable.
        ════════════════════════════════════════════════════════════
                          precision  recall  f1-score   support
             no_incident       0.99    0.73      0.84      2800
                incident       0.16    0.91      0.27       180
               ...
    """
    y_pred = (y_proba >= threshold.value).astype(int)
    separator = "═" * 64
    print(separator)
    print(f"Policy: {threshold.label}  (threshold={threshold.value})")
    print(threshold.description)
    print(separator)
    print(classification_report(
        y_true, y_pred,
        target_names=["no_incident", "incident"],
        zero_division=0,
    ))
