# Governance

## Overview

**cloud-incident-predictor** is maintained by a single developer using a
**benevolent dictator** model. All decisions about direction, scope, and
technical approach rest with the sole maintainer.

---

## Maintainer

**Ignacio Garbayo Fernandez** ([@igarbayo](https://github.com/igarbayo))

- Reviews and merges pull requests
- Makes release decisions
- Manages branch protection rules
- Enforces the Code of Conduct

---

## Branching model (Git Flow)

```
main ──────────────────────────────── stable releases only
  │
develop ──────────────────────────── active integration
  │
  ├── feature/<description>          one per feature/fix
  ├── release/x.y.z                  stabilisation before release
  └── hotfix/<description>           emergency fix from main
```

### Branch rules

| Branch | Protection | Who can merge |
|--------|------------|---------------|
| `main` | Protected; no direct pushes | Maintainer only, via `release/*` or `hotfix/*` |
| `develop` | Open to PRs | Maintainer approval required |
| `feature/*` | No protection | Author, reviewed before merging to `develop` |

---

## Release process

1. Cut a `release/x.y.z` branch from `develop`
2. Update `CHANGELOG.md` — move `[Unreleased]` items under the new version
3. Bump version if applicable
4. Merge `release/x.y.z` → `main` and → `develop`
5. Tag `main` with `vx.y.z`
6. Push the tag; GitHub Actions CI must pass before the tag is considered stable

```bash
# Example release flow for v1.1.0
git checkout develop
git checkout -b release/1.1.0
# ... update CHANGELOG, bump version ...
git commit -m "chore(release): prepare v1.1.0"
git checkout main && git merge --no-ff release/1.1.0
git tag -a v1.1.0 -m "Release v1.1.0"
git checkout develop && git merge --no-ff release/1.1.0
git branch -d release/1.1.0
```

---

## Contributor recognition

Contributors whose pull requests are accepted are listed in the CHANGELOG
under the relevant version's attribution section. If you would prefer not
to be listed, state this in your PR.

---

## Decision-making

There is no formal voting process. The maintainer makes final decisions,
weighing:

1. Alignment with the project's educational purpose
2. Correctness and simplicity (KISS principle)
3. Compatibility with existing methodology choices (TDD, HDD, MDD)
4. Maintainability for a single developer

Disagreements are resolved through respectful discussion in issues or PRs.
The maintainer's decision is final.
