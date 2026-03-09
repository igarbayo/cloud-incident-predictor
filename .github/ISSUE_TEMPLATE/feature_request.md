---
name: Feature request
about: Suggest an improvement or new capability
title: "[FEAT] "
labels: enhancement
assignees: igarbayo
---

## What problem does this solve for you?

Explain the situation or limitation you ran into. Be specific about the
context — what were you trying to do when you hit this limitation?

**Example:**
> When I tried to evaluate the model on a real dataset (not synthetic), I had no
> way to pass a custom CSV path to `generate_data.py`. I had to edit the source
> file directly, which felt fragile.

## Describe the solution you'd like

A clear description of what you would like to happen.

**Example:**
> Add a `--input-csv` command-line argument to `src/generate_data.py` so users
> can pass a pre-existing CSV instead of generating synthetic data.
> ```
> python src/generate_data.py --input-csv my_real_metrics.csv
> ```

## Alternatives you've considered

What other approaches did you think about, and why did you prefer the one above?

**Example:**
> I considered editing `OUTPUT_PATH` in the source file, but that would break
> the default behaviour for other users. A CLI argument is non-breaking.

## Additional context

Any other context, mockups, or references that would help evaluate this request.
