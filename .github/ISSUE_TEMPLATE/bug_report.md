---
name: Bug report
about: Something isn't working as expected
title: "[BUG] "
labels: bug
assignees: igarbayo
---

## What happened?

Describe what went wrong. Include the actual output or error message you received.

**Example:**
> Running `pytest tests/test_preprocess.py::TestNoDataLeakage` fails with:
> `AssertionError: Arrays are not equal — future spike at t=100 leaked into features`

## What did you expect?

Describe what you expected to happen instead.

**Example:**
> The test should pass, because metric[100] is outside the window metric[84:99].

## Steps to reproduce

Please provide a numbered list so anyone can reproduce the issue from scratch.

1. Clone the repository: `git clone https://github.com/igarbayo/cloud-incident-predictor.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Generate data: `python src/generate_data.py`
4. Run the failing test: `pytest tests/test_preprocess.py -v`

## Environment

- **OS**: (e.g. Ubuntu 22.04, Windows 11, macOS 14)
- **Python version**: (run `python --version`)
- **scikit-learn version**: (run `python -c "import sklearn; print(sklearn.__version__)"`)
- **Branch / commit**: (run `git rev-parse --short HEAD`)

## Additional context

Any other context, screenshots, or log output that might help diagnose the issue.
