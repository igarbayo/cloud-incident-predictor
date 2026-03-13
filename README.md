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
Given: the last W timesteps of F metrics (CPU usage, Memory usage, ...)
Predict: will an incident occur in the next H timesteps? (yes/no)
```

Mathematically, at each timestep *t*:

```
X_t = [cpu(t-W), ..., cpu(t-1), mem(t-W), ..., mem(t-1)]   ← W × F raw past values
Y_t = 1  if any incident in [t, ..., t+H-1]
      0  otherwise
```

The **sliding window formulation** transforms the time series into a tabular
supervised learning problem. Each window becomes one row in a feature matrix
of shape `(N-W-H, W×F)`, and a Random Forest learns to classify rows as
pre-incident or normal.

---

## Synthetic dataset

The generator (`src/generate_data.py`) produces **two correlated cloud metrics**
over 10,000 timesteps:

| Column | Signal | Real-world meaning |
|--------|--------|--------------------|
| `metric` | Sinusoidal baseline + noise + incidents | **CPU usage (%)** |
| `metric_2` | mixing coeff. 0.6 with CPU + independent cosine + noise | **Memory usage (%)** |

The generator mixes Memory as `0.6 * CPU + 0.4 * memory_base`. The measured
Pearson correlation is **r = 0.887** (not 0.6) because incident spikes dominate
the variance of CPU, amplifying the shared component. Memory still carries ~21%
independent variance (r² = 0.79), which is what Hypothesis H5 tests.

### Six incident types

| Type | Duration | Real-world cause |
|------|----------|-----------------|
| `spike` | 1–5 steps | DDoS, flash sale, traffic burst |
| `threshold_breach` | 8–15 steps | Runaway process, CPU pinned at 100% |
| `gradual_degradation` | 15–25 steps | Memory leak, disk fill, GC pressure |
| `level_shift` | 10–20 steps | Bad deploy, partial failure |
| `drop` | 3–8 steps | Service crash, network blackout |
| `oscillation` | 5–12 steps | Autoscaler thrashing, feedback loop |

Incident rate: ~6–13% of timesteps depending on random seed.

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
# → columns: t, metric (CPU %), metric_2 (Memory %), is_incident, incident_type

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
from src.preprocess import create_sliding_windows

# Single metric (CPU only)
X, y = create_sliding_windows(df, W=30, H=5)

# Multivariate (CPU + Memory — improved PR-AUC)
X, y = create_sliding_windows(df, W=30, H=5, feature_cols=["metric", "metric_2"])

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

### Hypotheses investigated

| # | Hypothesis | Result |
|---|-----------|--------|
| H1 | Longer lookback window (W) improves PR-AUC | CONFIRMED — W=30 optimal |
| H2 | PR-AUC substantially outperforms the random baseline | CONFIRMED — 4× lift |
| H3 | `class_weight='balanced'` improves recall over unweighted | CONFIRMED for precision/PR-AUC |
| H4 | PR-AUC degrades as prediction horizon H increases | investigated in notebook |
| H5 | CPU + Memory (multivariate) improves over CPU alone | investigated in notebook |

---

## Results

*Results are generated by running the notebook. Your run will be identical (fixed `random_state=42`).*

Test set: 2,990 windows — 15.79% incident rate — W=30, H=5, `any()` labeling, `class_weight='balanced'`

| Policy | Threshold | Precision | Recall | F1 | Alerts fired |
|--------|-----------|-----------|--------|----|-------------|
| Aggressive | 0.3 | 0.81 | 0.50 | 0.62 | 293 |
| Balanced | 0.5 | 0.89 | 0.39 | 0.55 | 210 |
| Conservative | 0.8 | 0.91 | 0.24 | 0.38 | 124 |

PR-AUC: **0.6339** vs random baseline **0.1579** — lift **4.0×**

---

## Adapting to a real alerting system

The current implementation is a proof-of-concept on synthetic data. Moving to production would require the following changes:

| Gap | Production approach |
|-----|---------------------|
| **Batch inference** | Replace sliding-window precomputation with an incremental buffer that appends each new datapoint and predicts in real time |
| **Two metrics** | Extend to further signals beyond CPU + Memory (network I/O, disk, error rate, request latency) |
| **Fixed anomaly shape** | Train on historical incidents from real systems, which include gradual degradation, oscillations, and silent failures |
| **No concept drift** | Schedule periodic retraining (e.g. weekly) with a rolling window of recent data; monitor PR-AUC in production to detect drift |
| **No feedback loop** | Log every alert and its outcome (true/false positive); use that signal to retrain and tune the threshold over time |
| **Threshold selection** | Run the cost model (`AlertPolicy`) with real on-call cost estimates rather than the illustrative 50×/2× examples |

The core abstractions (`AlertPredictor`, `AlertThreshold`, `AlertPolicy`) are designed to be drop-in replaceable — the domain model does not change, only the data pipeline and retraining schedule around it.

---

## Known limitations

- **Synthetic data only**: the generator simulates a simplified metric. Real
  cloud metrics have more complex distributions, seasonal patterns, and
  correlated signals across multiple dimensions.
- **Two synthetic metrics**: the model uses CPU and Memory usage. Real alerting
  systems add further signals (network I/O, disk, error rate, request latency).
- **No streaming inference**: the current implementation is batch-oriented.
  A production system would need to compute the feature window incrementally
  as new data arrives.
- **Six synthetic anomaly types**: the generator covers spike, threshold_breach,
  gradual_degradation, level_shift, drop, and oscillation. Real incidents
  include additional patterns such as correlated cascading failures across services.
- **No concept drift handling**: the model is trained once. In production,
  periodic retraining would be needed as metric behaviour evolves.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions welcome!

## Security

See [SECURITY.md](SECURITY.md) for how to report vulnerabilities privately.

## Third-party data

Section 9 of the notebook downloads data from the
[Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB) at runtime:

| File | Source path in NAB | Licence |
|------|-------------------|---------|
| `data/nab_cpu_asg.csv` | `data/realKnownCause/cpu_utilization_asg_misconfiguration.csv` | MIT |
| `data/nab_windows.json` | `labels/combined_windows.json` | MIT |

Full attribution: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

---

## License

[MIT](LICENSE) © 2025 Ignacio Garbayo Fernandez
