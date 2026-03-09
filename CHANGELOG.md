# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- `src/__init__.py`: MDD domain models — `AlertThreshold`, `AlertPolicy`, and canonical threshold instances (`AGGRESSIVE`, `BALANCED`, `CONSERVATIVE`)
- `src/generate_data.py`: synthetic time-series generator with sinusoidal base signal and injected spike anomalies
- `src/preprocess.py`: sliding-window feature engineering (`create_sliding_windows`) and temporal train/test split (`temporal_split`)
- `src/model.py`: `AlertPredictor` class wrapping `RandomForestClassifier` with domain-aware interface
- `src/evaluate.py`: evaluation suite — PR curve, threshold sweep, feature importance chart, classification report
- `tests/test_generate_data.py`: TDD contracts for the data generator (shape, dtypes, incident rate)
- `tests/test_preprocess.py`: TDD contracts for the sliding window (data leakage invariant, label off-by-one, split ordering)
- `tests/test_model.py`: TDD contracts for the model wrapper (hyperparameters, predict_proba shape, save/load roundtrip)
- `notebooks/01_exploration_and_evaluation.ipynb`: HDD notebook with H1/H2/H3 hypothesis testing and threshold analysis
- `.github/workflows/ci.yml`: GitHub Actions CI — pytest + REUSE compliance check
- REUSE v3.3 compliance: SPDX headers in all source files, `LICENSES/MIT.txt`, `.reuse/dep5`
- Community health files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `GOVERNANCE.md`

---

<!-- Versioning template for future releases:

## [1.0.0] - YYYY-MM-DD

### Added
- ...

### Changed
- ...

### Fixed
- ...

### Removed
- ...

[Unreleased]: https://github.com/igarbayo/cloud-incident-predictor/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/igarbayo/cloud-incident-predictor/releases/tag/v1.0.0

-->
