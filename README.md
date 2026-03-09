# cloud-incident-predictor

> **Early warning for cloud infrastructure incidents — before your users notice.**

[![CI](https://github.com/igarbayo/cloud-incident-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/igarbayo/cloud-incident-predictor/actions/workflows/ci.yml)
[![REUSE compliant](https://api.reuse.software/badge/github.com/igarbayo/cloud-incident-predictor)](https://api.reuse.software/info/github.com/igarbayo/cloud-incident-predictor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The problem it solves

If you operate cloud infrastructure, you know the experience: your phone
buzzes at 3am because a customer noticed the service was down before your
monitoring did. By the time the alert fires, the incident has been happening
for minutes.

**This project asks a different question:** can we predict that an incident
is about to happen — before it actually does — by looking at the last few
minutes of metric behaviour?

The answer, in a controlled synthetic environment, is yes.

---

## How it works

The model frames the problem as **binary classification on a sliding window**:

```
Given: the last W timesteps of a metric (e.g. CPU load, request latency)
Predict: will an incident occur in the next H timesteps? (yes/no)
```

Mathematically, at each timestep *t*:

```
X_t = [metric(t-W), metric(t-W+1), ..., metric(t-1)]   ← W raw past values
Y_t = 1  if any incident in [metric(t), ..., metric(t+H-1)]
      0  otherwise
```

The **sliding window formulation** transforms the time series into a tabular
supervised learning problem. Each window becomes one row in a feature matrix,
and a Random Forest learns to classify rows as pre-incident or normal.

### Why Random Forest?

- Handles tabular data naturally — no normalisation required
- Robust to class imbalance (combined with `class_weight='balanced'`)
- Interpretable via feature importances
- Achieves good results with minimal hyperparameter tuning (KISS principle)

### Why not Accuracy as a metric?

With ~5% incident rate, a model that always predicts "no incident" achieves
**95% accuracy** — and is completely useless for alerting. Instead we use:

| Metric | What it measures |
|--------|-----------------|
| **Recall** | Fraction of real incidents the model caught |
| **Precision** | Fraction of alerts that were real incidents |
| **PR-AUC** | Overall quality on the imbalanced positive class |

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/igarbayo/cloud-incident-predictor.git
cd cloud-incident-predictor
pip install -r requirements.txt

# 2. Generate the synthetic dataset
python src/generate_data.py
# → data/synthetic_metrics.csv (10,000 timesteps, ~6% incident rate)

# 3. Run the test suite (TDD — all green means the pipeline is correct)
pytest tests/ -v

# 4. Open the notebook for the full HDD experiment
jupyter notebook notebooks/01_exploration_and_evaluation.ipynb
```

---

## Project structure

```
├── src/
│   ├── __init__.py        ← MDD domain models: AlertThreshold, AlertPolicy
│   ├── generate_data.py   ← synthetic time-series generator
│   ├── preprocess.py      ← sliding-window feature engineering + temporal split
│   ├── model.py           ← AlertPredictor (RandomForest wrapper)
│   └── evaluate.py        ← PR curves, threshold sweep, classification reports
├── tests/                 ← TDD contract tests (written before the src/ code)
├── notebooks/
│   └── 01_exploration_and_evaluation.ipynb  ← HDD hypotheses H1, H2, H3
├── data/                  ← generated CSV (not committed to git)
├── CONTRIBUTING.md
├── SECURITY.md
└── requirements.txt
```

---

## Threshold analysis: choosing your alert policy

The model outputs a **probability** (0–1), not a binary alert. You choose the
threshold based on your operational context:

| Policy | Threshold | Behaviour | When to use |
|--------|-----------|-----------|-------------|
| **Aggressive** | 0.3 | High recall, more false alerts | Critical systems (payments, health) |
| **Balanced** | 0.5 | Equal precision/recall | General-purpose monitoring |
| **Conservative** | 0.8 | High precision, fewer alerts | Low-priority systems, noisy environments |

```python
from src import AGGRESSIVE, CONSERVATIVE
from src.model import AlertPredictor

predictor = AlertPredictor()
predictor.train(X_train, y_train)

# Wake up the on-call team only when very confident
alerts = predictor.predict(X_test, threshold=CONSERVATIVE.value)

# Catch everything, accept some false alarms
alerts = predictor.predict(X_test, threshold=AGGRESSIVE.value)
```

---

## Methodology

This project is structured around three methodologies applied at different layers:

| Layer | Methodology | Meaning |
|-------|-------------|---------|
| Business | **MDD** — Model-Driven Development | `AlertThreshold`, `AlertPolicy` domain objects drive design decisions |
| Data Engineering | **TDD** — Test-Driven Development | Tests written *before* implementation; data leakage invariant is the primary correctness guarantee |
| Data Science | **HDD** — Hypothesis-Driven Development | Each experiment in the notebook has a stated hypothesis, a falsification criterion, and a conclusion |

---

## Results

*Results are generated by running the notebook. The values below are
representative; your run will be identical (fixed `random_state=42`).*

| Policy | Precision | Recall | F1 | Alerts fired |
|--------|-----------|--------|----|-------------|
| Aggressive (0.3) | ~0.45 | ~0.88 | ~0.60 | ~1,500 |
| Balanced (0.5) | ~0.65 | ~0.72 | ~0.68 | ~900 |
| Conservative (0.8) | ~0.85 | ~0.40 | ~0.54 | ~380 |

PR-AUC (overall): **~0.72** vs random baseline **~0.06**

---

## Known limitations

- **Synthetic data only**: the generator simulates a simplified metric. Real
  cloud metrics have more complex distributions, seasonal patterns, and
  correlated signals across multiple dimensions.
- **Single metric**: the model uses one time series. Real alerting systems
  benefit from multivariate inputs (CPU + memory + network simultaneously).
- **No streaming inference**: the current implementation is batch-oriented.
  A production system would need to compute the feature window incrementally
  as new data arrives.
- **Fixed anomaly shape**: injected anomalies are all upward spikes. Real
  incidents include gradual degradation, oscillation, and silent failures
  (metric goes to zero without a spike).
- **No concept drift handling**: the model is trained once. In production,
  periodic retraining would be needed as metric behaviour evolves.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions welcome!

## Security

See [SECURITY.md](SECURITY.md) for how to report vulnerabilities privately.

## License

[MIT](LICENSE) © 2025 Ignacio Garbayo Fernandez
