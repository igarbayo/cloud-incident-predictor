## Description

What does this PR do? Summarise the change and why it is needed.
Link to the related issue: `Closes #<issue-number>`

---

## Type of change

- [ ] Bug fix (`fix(scope): ...`)
- [ ] New feature (`feat(scope): ...`)
- [ ] Documentation update (`docs(scope): ...`)
- [ ] Test addition or update (`test(scope): ...`)
- [ ] Refactor (`refactor(scope): ...`)
- [ ] CI / build (`chore(scope): ...`)

---

## How to test this

Provide a short, human-readable walkthrough so the reviewer can verify the change.

**Example:**
1. `git checkout feature/add-xgboost-support`
2. `pip install -r requirements.txt`
3. `python src/generate_data.py`
4. `pytest tests/ -v` — all tests should pass
5. In the notebook, run the H1 section — you should now see XGBoost in the comparison table

---

## Checklist

- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] New behaviour has tests (if applicable)
- [ ] SPDX header added to any new Python files
- [ ] `reuse lint` passes
- [ ] Commit messages follow Conventional Commits (`type(scope): description`)
- [ ] CHANGELOG.md updated under `[Unreleased]`
