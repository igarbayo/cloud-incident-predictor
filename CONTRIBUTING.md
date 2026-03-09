# Contributing

Thank you for making it this far and wanting to support this project — it
means a lot. Every contribution, whether it's a bug report, a fix, or a
question in the issues, helps make this project better.

## A note on response time

I'm a student maintaining this in my spare time. Issues and pull requests
**will be reviewed**, but resolution time is indefinite — please be patient.
I appreciate your understanding.

---

## How to contribute

### 1. Set up the project

```bash
git clone https://github.com/igarbayo/cloud-incident-predictor.git
cd cloud-incident-predictor
pip install -r requirements.txt
```

Generate the synthetic dataset before running tests:

```bash
python src/generate_data.py
```

Run the test suite to make sure everything is green:

```bash
pytest tests/ -v
```

---

### 2. Find or create an issue

- Browse [open issues](https://github.com/igarbayo/cloud-incident-predictor/issues) first.
- If you found a bug or have a feature idea, open a new issue using the
  appropriate template and describe it clearly.
- For significant changes, please open an issue before submitting a PR so
  we can discuss the approach.

---

### 3. Branch from `develop` using Git Flow

This project uses [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/).

The two main long-lived branches are:

| Branch    | Purpose                                    |
|-----------|--------------------------------------------|
| `main`    | Stable, released code only                 |
| `develop` | Active development — all PRs target this   |

**For a new feature or fix:**

```bash
# Start from develop
git checkout develop
git pull origin develop

# Create your feature branch
git checkout -b feature/add-xgboost-support
# or for a bug fix:
git checkout -b feature/fix-leakage-in-preprocess
```

Work on your branch, then open a PR targeting `develop` (not `main`).

---

### 4. Follow Conventional Commits

All commits must follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): short description in imperative mood
```

**Types**: `feat`, `fix`, `docs`, `test`, `chore`, `refactor`

**Scopes**: `generate-data`, `preprocess`, `model`, `evaluate`, `notebook`, `ci`, `docs`

**Examples:**

```bash
# Adding a new feature
git commit -m "feat(model): add XGBoost as alternative classifier"

# Fixing a bug
git commit -m "fix(preprocess): correct off-by-one in label horizon boundary"

# Adding or updating tests
git commit -m "test(preprocess): add leakage test for boundary at incident start"

# Updating documentation
git commit -m "docs(readme): add results table for PR-AUC comparison"

# CI/build changes
git commit -m "chore(ci): pin reuse-action to v4"
```

---

### 5. Add SPDX headers to new files

This project follows [REUSE Spec v3.3](https://reuse.software/spec/).
Every new Python file must start with:

```python
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Ignacio Garbayo Fernandez
```

For files that cannot have headers (notebooks, data files, markdown), they are
covered by `.reuse/dep5` automatically — no action needed.

To verify compliance before opening a PR:

```bash
pip install reuse
reuse lint
```

---

### 6. Run tests before opening a PR

```bash
pytest tests/ -v --tb=short
```

All tests must pass. If a test fails, either fix the code or update the test
with a clear explanation of why the behaviour changed.

---

### 7. Open a pull request

- Target the `develop` branch.
- Fill in the PR template completely.
- Link to the issue it resolves (e.g., `Closes #12`).
- Keep PRs focused — one logical change per PR.

---

## What makes a good contribution?

- A clear problem statement (what was wrong / what was missing)
- A minimal, focused implementation (avoid scope creep)
- Tests for new behaviour
- Human-friendly examples in docstrings (see existing code for style)
- Conventional commit messages throughout

Thank you again for your time and interest!
