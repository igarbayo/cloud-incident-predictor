# Decision Log — Predictive Alerting for Cloud Metrics

<!-- SPDX-License-Identifier: MIT -->
<!-- SPDX-FileCopyrightText: 2025 Ignacio Garbayo Fernandez -->

This document records every significant decision made during the project — accepted and rejected — in the order they happened. The goal is simple: anyone picking up this codebase should be able to understand not just *what* the code does, but *why* it ended up this way. Rejected decisions are treated as findings, not failures.

---

## How This Project Makes Decisions

The project applies three different disciplines at three different layers. Each layer has its own way of reasoning about decisions.

### HDD — Hypothesis-Driven Development (data science layer)

Every experiment in the notebook starts with a written hypothesis and a falsification criterion before any code runs. This forces us to define upfront what "this idea worked" actually means, so we can't move the goalposts after seeing the results.

In practice this looks like: *"We predict PR-AUC will be higher for W=15 than W=5. If the difference is less than 2 pp, window size doesn't matter."* Then we run the experiment. If the data says the hypothesis is wrong, we update our understanding — we don't adjust the criterion to make the result look like a success.

Examples in this project where HDD changed the direction:
- H3: we expected `class_weight='balanced'` to improve recall. It didn't (−1.3 pp). We kept it anyway because PR-AUC and precision were better — a legitimate trade-off, but the hypothesis was still called REJECTED.
- H5: we expected adding a second metric (Memory) to help. It hurt (−4.0 pp). That rejection led directly to H5b, which confirmed that the right second metric (error_rate) does help (+9.4 pp). The rejection was the useful result.

### TDD — Test-Driven Development (data engineering layer)

The most dangerous bug in time-series ML is data leakage — using future data to predict the past. In `preprocess.py`, the temporal split is enforced structurally: `temporal_split()` has no `random_state` parameter and no shuffle option. It's not that we promise not to shuffle; it's that the function cannot shuffle.

The test suite in `tests/test_preprocess.py` verifies this invariant directly. It checks that train indices always precede test indices, that feature windows don't overlap with label windows, and that the split ratio is deterministic. If someone modifies `temporal_split()` in a way that introduces leakage, the tests fail.

Same principle applies to `StreamPredictor`: the verification in Section 11 isn't just a sanity check, it's a correctness proof — batch and streaming produce bit-for-bit identical results (max diff = 0.00).

### MDD — Model-Driven Design (business layer)

The business concepts in this project are first-class code objects, not magic numbers. The three alert policies are defined once in `src/__init__.py` as `AlertThreshold` instances with names, threshold values, and human-readable descriptions:

```python
AGGRESSIVE   = AlertThreshold("aggressive",   0.3, "High recall — catches most incidents...")
BALANCED     = AlertThreshold("balanced",     0.5, "Balanced trade-off...")
CONSERVATIVE = AlertThreshold("conservative", 0.8, "High precision — fewer alerts...")
```

Every function that evaluates thresholds — `threshold_sweep()`, `print_classification_report()`, `plot_threshold_comparison()` — accepts `AlertThreshold` objects, not floats. This means callers must think in business terms ("which policy am I evaluating?") rather than numerical terms ("what float should I pass?"). It also means the threshold values have a single source of truth: if `AGGRESSIVE` changes from 0.3 to 0.25, it changes in one place.

---

## Decision Log

### 1 — Frame the problem as binary classification

**What we decided:** predict a binary label (incident / no incident) for each timestep, not a continuous anomaly score.

**Why:** alerting systems need a clear signal. "Fire an alert or don't" is the action. A continuous score requires someone to decide a threshold in production, which is a harder operational problem. Binary classification with named policies (`AGGRESSIVE`, `BALANCED`, `CONSERVATIVE`) makes the trade-off explicit at design time.

**Alternatives considered:** anomaly detection (unsupervised, no labeled data needed) and regression (predict incident severity). Anomaly detection doesn't produce probability estimates that map cleanly to alert policies. Regression adds complexity without a clearer operational benefit.

---

### 2 — Use PR-AUC as primary metric, not ROC-AUC or accuracy

**What we decided:** PR-AUC is the single number that summarises model quality throughout the project.

**Why:** the incident rate is around 16%. A model that always predicts "no incident" achieves 84% accuracy — completely useless for alerting. ROC-AUC for such a model is around 0.5, which looks mediocre but is actually the random baseline. PR-AUC for that model equals the positive class rate (~0.16), so the gap between baseline and our model is immediately readable. Lift = model PR-AUC / baseline PR-AUC is the clearest way to say "how much better than chance are we?"

---

### 3 — Temporal split 70/30, never shuffled

**What we decided:** training data is the first 70% of the timeline; test data is the last 30%. No randomisation, ever.

**Why:** time-series data has a direction. Shuffling the split would mean training on data from the future and evaluating on data from the past — the model would learn patterns that don't exist in real deployment. The temporal split is enforced structurally in `temporal_split()`: the function has no `random_state` parameter and makes shuffling architecturally impossible. `tests/test_preprocess.py` verifies the invariant on every run.

---

### 4 — Use RandomForest as the model

**What we decided:** RandomForest with `class_weight='balanced'` is the core model.

**Why:** the feature matrix is a flat array of raw timestep values — a tabular problem. RandomForest handles this natively and well. It's interpretable via `feature_importances_` (useful for understanding which part of the window matters most), robust to class imbalance, and explainable to a non-ML audience as "many decision trees vote." It requires minimal preprocessing and minimal hyperparameter tuning to perform well.

**Alternatives considered:**
- LSTM / CNN: better at capturing long-range sequential dependencies, but requires significantly more data, longer training, and is much harder to explain. Not justified for a 10,000-step dataset.
- GradientBoosting: tested in the model comparison (Section 7). PR-AUC = 0.597 vs RF's 0.634. Precision drops from 0.886 to 0.562 — too noisy for an alerting system.
- Logistic Regression: tested in Section 7. PR-AUC = 0.486. The decision boundary for this problem is non-linear; a linear model can't capture it.

**Confirmed in Section 7:** RF wins on both PR-AUC and precision across all three candidates.

---

### 5 — `class_weight='balanced'` retained despite mixed H3 result

**What we decided:** keep `class_weight='balanced'` as a fixed setting.

**Context:** this was set upfront based on the assumption that balancing class weights would improve recall on the minority class (incidents). Then H3 tested this assumption directly.

**H3 result — REJECTED on recall:**

| | Recall | Precision | PR-AUC |
|---|---|---|---|
| balanced | 0.394 | 0.886 | 0.634 |
| unbalanced | 0.407 | 0.807 | 0.616 |

Balanced does NOT improve recall (−1.3 pp). The hypothesis was wrong.

**Why we kept it anyway:** balanced achieves higher PR-AUC (+0.018) and substantially better precision (+0.079). In an alerting system, precision directly measures alert fatigue — every false alert is an engineer waking up at 3am for nothing. The −1.3 pp recall cost is negligible; the +7.9 pp precision gain is operationally significant.

The hypothesis was rejected. The decision to retain the setting was made explicitly, not by ignoring the result.

---

### 6 — AlertThreshold as a domain object (not a raw float)

**What we decided:** define the three alert policies as named constants in `src/__init__.py`, make them the only way to call evaluation functions.

**Why:** if threshold values are raw floats scattered across notebooks and scripts, changing `AGGRESSIVE` from 0.3 to 0.25 requires a search-and-replace across the whole codebase. A single wrong 0.3 that doesn't get updated would silently produce different results in different places. Named objects with a single source of truth eliminate this class of bug.

It also forces callers to reason about the business decision: `print_classification_report(y_true, y_proba, AGGRESSIVE)` is self-documenting. `print_classification_report(y_true, y_proba, 0.3)` requires a comment to explain what 0.3 means.

---

### 7 — Window size W=30 (H1 — CONFIRMED)

**Hypothesis:** a lookback window of W=15 captures sufficient context; W=5 loses too much history.

**Falsification criterion:** if Δ PR-AUC (W=15 − W=5) < 2 pp, window size doesn't matter.

**Results:**

| W | PR-AUC |
|---|---|
| 5 | 0.592 |
| 15 | 0.630 |
| **30** | **0.634** ← optimum |
| 60 | 0.621 |
| 90 | 0.598 |

**Decision:** use W=30. Context matters: the difference W=15 − W=5 is +3.8 pp (exceeds threshold), confirming the hypothesis. But the empirical optimum is W=30, not W=15. Beyond W=30, older history adds noise rather than signal — the model is given data from so far in the past that it's no longer informative about the next 5 steps.

---

### 8 — `any()` labeling over the horizon (H2 — CONFIRMED)

**Hypothesis:** labeling a window as positive if *any* of the next H timesteps is an incident produces better early-warning than labeling only if the timestep exactly at t+H is an incident.

**Falsification criterion:** if Δ recall < 5 pp, labeling strategy doesn't matter.

**Results:**

| Strategy | PR-AUC | Recall |
|---|---|---|
| **any()** | **0.634** | **0.394** |
| last-step | 0.393 | 0.219 |

Δ recall = +17.5 pp. Not even close to the 5 pp threshold.

**Why it matters so much:** with six incident types, many anomalies build over multiple steps (gradual_degradation, level_shift, threshold_breach). `any()` marks a window as positive as soon as the incident has started but before the peak — exactly the warning we want. Last-step labeling only fires at the end of the incident window, missing the early warning entirely.

---

### 9 — Expanding the incident taxonomy from 1 to 6 types

**What we decided:** replace the single "spike" anomaly type with six distinct types, each mapping to a real cloud failure mode.

| Type | Real-world analogue |
|---|---|
| spike | DDoS, flash sale, traffic burst |
| threshold_breach | Runaway process, CPU pinned at 100% |
| gradual_degradation | Memory leak, disk fill, GC pressure |
| level_shift | Bad deploy, partial failure |
| drop | Service crash, network blackout |
| oscillation | Autoscaler thrashing, feedback loop |

**Why:** a dataset with only spikes teaches the model to detect spikes. Real systems fail in many ways. The expanded taxonomy makes the model generalise across failure modes and makes the benchmark more honest. It's also why H2's `any()` result is so strong — gradual_degradation and level_shift span multiple timesteps, so `any()` labeling catches them early while last-step labeling misses most of the positive windows.

---

### 10 — Horizon H=5 (H4 — CONFIRMED)

**Hypothesis:** prediction quality degrades as horizon H increases. Predicting 1 step ahead is easier than 20 steps ahead.

**Falsification criterion:** if PR-AUC at H=20 is within 5 pp of PR-AUC at H=1, horizon has negligible impact.

**Results:**

| H | PR-AUC | Positive rate |
|---|---|---|
| 1 | 0.745 | 11.8% |
| 3 | 0.675 | 13.8% |
| **5** | **0.634** | **15.8%** |
| 10 | 0.572 | 20.6% |
| 20 | 0.530 | 29.6% |

Δ (H=1 → H=20) = −21.5 pp. Confirmed.

**Decision:** use H=5. H=1 gives better metrics (PR-AUC = 0.745) but almost no reaction time — one timestep ahead is not useful for an on-call engineer. H=5 is the minimum operationally useful lead time. H=10 or H=20 degrade quality too much for the extra lead time to compensate.

---

### 11 — Second metric: Memory rejected (H5 — REJECTED)

**Hypothesis:** using two correlated metrics (CPU + Memory) improves performance over CPU alone because Memory carries additional signal.

**Falsification criterion:** if Δ PR-AUC < 1 pp, second metric adds no meaningful signal.

**Results:**

| Features | PR-AUC |
|---|---|
| CPU only (30 features) | 0.634 |
| CPU + Memory (60 features) | 0.594 |

Δ = −4.0 pp. The second metric actively hurts.

**Why:** Memory and CPU are correlated at r=0.887. High correlation means Memory carries only ~21% independent variance relative to CPU. That 21% independent variance doesn't add signal — it adds noise. Doubling the feature count from 30 to 60 gives the RF more dimensions to overfit, diluting the useful CPU signal.

**What this tells us:** adding a second metric only helps if it's genuinely independent. The threshold question isn't "is the correlation non-zero?" but "is there enough independent variance to contribute signal?"

---

### 12 — Second metric: error_rate accepted (H5b — CONFIRMED)

**Context:** H5's rejection was the prompt. The issue wasn't "multivariate models don't work," it was "Memory is the wrong second metric."

**New hypothesis:** error_rate (application error rate) is sufficiently independent of CPU (r=0.448) to contribute useful signal.

**Results:**

| Features | PR-AUC |
|---|---|
| CPU only | 0.634 |
| CPU + error_rate | **0.728** |

Δ = +9.4 pp. Confirmed strongly.

**Why it works:** error_rate has its own noise profile during normal operation (~2% base, Gaussian) and spikes independently during incidents (~20%). It's not measuring the same thing as CPU — it's measuring the application layer's response to whatever is causing the incident. r=0.448 means the two metrics share about 20% of variance, leaving 80% independent. That independent 80% carries genuinely new predictive information.

**Final model uses CPU + error_rate.** The H5 rejection was the best thing that happened in this project — it led directly to a +9.4 pp improvement.

---

### 13 — Model comparison: RandomForest confirmed (Section 7)

All three models tested on the final W=30, H=5, `any()`, balanced configuration (before adding error_rate):

| Model | PR-AUC | Recall | Precision |
|---|---|---|---|
| **RandomForest** | **0.634** | 0.394 | **0.886** |
| GradientBoosting | 0.597 | 0.549 | 0.562 |
| LogisticRegression | 0.486 | 0.540 | 0.345 |

RandomForest wins on PR-AUC (+3.7 pp over GB, +14.8 pp over LR) and by a wide margin on precision (+0.324 over GB). GB and LR fire more alerts overall — most of them false positives. Low precision means alert fatigue. RF was the right choice.

---

### 14 — Real-data validation: NAB dataset selection (Section 9)

**First candidate chosen and rejected: `cpu_utilization_asg_misconfiguration.csv`**

After downloading and analysing the data: this file has a single 5-day anomaly block right at the end of the series. With a 70/30 temporal split, all 1,499 incident timesteps fall in the test set — the training set contains zero positive examples. The model has never seen an incident and can't learn to predict one. No meaningful evaluation is possible.

**Selected instead: `machine_temperature_system_failure.csv`**

22,695 rows spanning 79 days, with 4 distinct anomaly windows distributed across the timeline. The 70/30 split gives training examples in the first 56 days and test examples in the last 23 days. The model can actually learn from the training anomalies.

**Result: PR-AUC = 0.617, lift = 3.7×.** The synthetic pipeline generalises to real sensor data with only an 11 pp drop in PR-AUC despite a full distribution shift. The temporal structure of pre-incident degradation is learnable across domains.

---

### 15 — Boundary condition: ELB dataset (Section 9.2)

**What happened:** tested the pipeline on `elb_request_count_8c0756.csv` (AWS load balancer traffic). Result: PR-AUC = 0.1575, lift = 0.9× — slightly below random baseline.

**Why it failed:**

Traffic spikes on a load balancer are externally triggered: a viral event, a DDoS attack, a flash sale. The 30 timesteps before the spike look completely normal because nothing in the system caused the spike — something outside the system did. The sliding window contains no signal about what is about to happen.

This is a fundamental mismatch between the problem formulation and the anomaly type. The pipeline assumes: *past metric values carry predictive signal about future incidents.* That assumption holds when incidents are system-internal (a CPU that's been climbing for 15 steps will keep climbing). It fails when incidents are externally triggered step-changes with no precursor.

**This is a boundary condition, not a bug.** The model isn't broken — it's being asked to do something it structurally cannot do. Documented honestly in the notebook and summary.

---

### 16 — Statistical features: rejected (Section 10)

**What we tested:** adding 6 summary statistics per metric per window (mean, std, slope, min, max, z_last) as explicit features alongside the raw timestep values.

**Results:**

| Configuration | PR-AUC | vs baseline |
|---|---|---|
| Raw features, default | 0.728 | — |
| Stat features, default | 0.723 | −0.5 pp |
| Stat features + tuning | 0.727 | −0.1 pp |

Stat features make things slightly worse, not better.

**Why:** RandomForest learns its own aggregations. A split like "if metric_t-3 > 1.2" combined with "if metric_t-4 > 1.0" implicitly computes a local maximum. Providing explicit mean, std, slope adds dimensions that partially duplicate what the RF already computes internally — and adds 12 extra features that the RF has to evaluate at every split, slightly diluting the raw temporal signal.

**The feature is kept in `preprocess.py` as an opt-in parameter** (`statistical_features=True`) because it might be useful for other model types (logistic regression, SVM) that don't learn their own aggregations. But it's not used in the final RF model.

---

### 17 — Hyperparameter tuning: marginal gain, retained (Section 10)

**What we tested:** GridSearchCV over max_depth [None, 10, 20], min_samples_leaf [1, 5, 10], max_features ['sqrt', 'log2'], with TimeSeriesSplit (5 folds, no leakage).

**Result:** best params are max_depth=20, max_features='sqrt'. Gain: +0.68 pp (0.728 → 0.735).

**Decision:** retain tuning in the recommended configuration. +0.68 pp is small but consistent. Using TimeSeriesSplit instead of standard k-fold is non-negotiable — standard k-fold would leak future folds into training during the grid search.

The default RF (max_depth=None, max_features='sqrt') is already close to optimal for this problem, which reflects well on the original design choices. Tuning is not a substitute for good feature engineering and correct problem framing — it's a small improvement on top of an already well-calibrated model.

---

### 18 — Streaming inference: StreamPredictor (Section 11)

**The limitation:** the original pipeline was batch-only. In production, metrics arrive one timestep at a time, not as a pre-computed matrix.

**What we built:** `StreamPredictor` in `src/stream.py` — a rolling buffer that wraps a trained `AlertPredictor`. Each call to `step(metric=x, error_rate=y)` adds one new row, drops the oldest if the buffer exceeds W, and returns a probability once the buffer is full.

**Key design constraint:** the trained model is reused completely unchanged. The only new code is the ingestion layer. This guarantees that streaming and batch produce identical results — because they use the same model on the same feature vectors.

**Verified:** max diff between streaming and batch outputs = 0.00 (bit-for-bit identical, no floating-point rounding). The window alignment proof: `probas[i]` uses buffer `df[6975+i : 7005+i]` = `X_test_f[i]`.

---

## Where Things Stand

The final model (CPU + error_rate, W=30, H=5, `any()`, RandomForest, `class_weight='balanced'`) achieves:

- PR-AUC = **0.728** on the synthetic test set (lift **4.6×** over random baseline)
- PR-AUC = **0.617** on real sensor data (NAB machine temperature, lift **3.7×**)
- All three alert policies achieve **>90% precision**
- Streaming inference verified at **max diff = 0.00**

Two honest limitations remain:

1. **Recall ceiling:** even AGGRESSIVE (threshold=0.3) catches only ~50% of incidents. Early prediction 5 steps ahead is inherently hard. A sequence model (LSTM, CNN) or a shorter horizon would improve recall.
2. **No concept drift handling:** the model assumes the metric distribution stays stable over time. A deployed system would need periodic retraining as workload patterns shift.
