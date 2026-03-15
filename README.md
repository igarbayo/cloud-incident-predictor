# cloud-incident-predictor

Binary classification system that predicts cloud infrastructure incidents before they happen, using a sliding window over metric time series.

[![CI](https://github.com/igarbayo/cloud-incident-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/igarbayo/cloud-incident-predictor/actions/workflows/ci.yml)
[![REUSE compliant](https://api.reuse.software/badge/github.com/igarbayo/cloud-incident-predictor)](https://api.reuse.software/info/github.com/igarbayo/cloud-incident-predictor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This is an internship project developed for JetBrains by Ignacio Garbayo Fernandez.

---

## Read the decision log first

**[docs/decision_log.md](docs/decision_log.md)** is the right place to understand why the code looks the way it does. It records every significant design choice in chronological order — accepted and rejected — with the actual numbers and reasoning behind each one. It also explains the three methodologies applied in the project (HDD, TDD, MDD) with concrete examples from this codebase.

Reading just this README gives you the what. The decision log gives you the why.

---

## What this is

Given the last W timesteps of cloud metrics, the model predicts whether an incident will occur in the next H timesteps. This is a binary classification problem on a sliding window feature matrix of shape `(N-W-H, W×F)`.

Three methodologies are applied at different layers: **HDD** for data science (the notebook runs hypotheses H1–H5b with stated falsification criteria), **TDD** for data engineering (the no-leakage invariant is tested in `tests/`, not just documented), and **MDD** for business (`AlertThreshold` and `AlertPolicy` are domain objects, not raw floats).

The final model is a RandomForest trained on CPU usage and application error rate (W=30, H=5). It achieves PR-AUC = 0.728 on the synthetic test set (lift 4.6× over the random baseline) and PR-AUC = 0.617 on real sensor data from the Numenta Anomaly Benchmark (lift 3.7×).

---

## How to run

```bash
pip install -r requirements.txt

# Generate the synthetic dataset (required before tests or notebook)
python src/generate_data.py

# Run the test suite
pytest tests/ -v

# Open the notebook
jupyter notebook notebooks/01_exploration_and_evaluation.ipynb
```

---

## Project structure

```
src/
├── __init__.py        — AlertThreshold, AlertPolicy domain objects
├── generate_data.py   — synthetic 10,000-step time-series generator (6 incident types)
├── preprocess.py      — sliding-window feature engineering + temporal split
├── model.py           — AlertPredictor (RandomForest wrapper + tune())
├── stream.py          — StreamPredictor (rolling buffer for single-timestep inference)
└── evaluate.py        — PR curves, threshold sweep, feature importances

tests/                 — TDD contract tests; test_preprocess.py enforces no data leakage
notebooks/
└── 01_exploration_and_evaluation.ipynb  — HDD experiment (Sections 1–11)
docs/
└── decision_log.md    — full decision history
data/                  — generated CSVs, not committed to git
```

---

## Third-party data

Section 9 of the notebook downloads data from the [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB) at runtime. Full attribution: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

See [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE) © 2025 Ignacio Garbayo Fernandez
