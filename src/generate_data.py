# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Ignacio Garbayo Fernandez

"""
Synthetic cloud metrics generator.

Generates a time series that simulates a cloud metric (e.g. CPU usage,
request latency) with realistic structure and injected anomaly windows.

Signal structure:
  - Base: sinusoidal wave simulating diurnal patterns (day/night cycles)
  - Noise: Gaussian noise simulating measurement uncertainty
  - Anomalies: random spike windows injected at random positions

The anomalies simulate real incident scenarios:
  - A sudden traffic spike (DDoS, viral content, flash sale)
  - Duration 1-5 timesteps (short burst incidents)

Output schema (saved to data/synthetic_metrics.csv):
  t           : int   — timestep index 0..N-1
  metric      : float — simulated metric value (e.g. normalised CPU %)
  is_incident : int   — 1 during injected anomaly windows, 0 otherwise
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants — single source of truth, referenced by tests
# ---------------------------------------------------------------------------

N_STEPS = 10_000
ANOMALY_FRACTION = 0.02   # fraction of timesteps that are anomaly *start* points
RANDOM_SEED = 42
OUTPUT_PATH = Path("data/synthetic_metrics.csv")


def generate_synthetic_data(
    n_steps: int = N_STEPS,
    anomaly_fraction: float = ANOMALY_FRACTION,
    seed: int = RANDOM_SEED,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Generate a synthetic time series with labelled incident windows.

    The signal is a sinusoidal wave (simulating diurnal load patterns) plus
    Gaussian noise (measurement uncertainty). Anomalies are injected as
    sudden spikes drawn from Normal(5, 1), lasting 1–5 timesteps each.

    Args:
        n_steps:          Total number of timesteps to generate.
        anomaly_fraction: Fraction of timesteps that are anomaly start points.
                          With duration 1-5, the actual incident rate will be
                          roughly anomaly_fraction * 3 ≈ 6% of all timesteps.
        seed:             Random seed for full reproducibility.
        output_path:      Where to save the CSV. Relative to working directory.

    Returns:
        DataFrame with columns: t, metric, is_incident.

    Example output (first few rows around an anomaly):
        t=137: metric=0.712,  is_incident=0
        t=138: metric=5.834,  is_incident=1  ← spike injected here
        t=139: metric=6.102,  is_incident=1
        t=140: metric=0.534,  is_incident=0
    """
    np.random.seed(seed)
    time = np.arange(n_steps)

    # Base signal: sinusoidal + Gaussian noise
    # sin(t * 0.1) has a period of ~63 steps — roughly simulating hourly cycles
    # in a dataset where each step is a minute
    metric = np.sin(time * 0.1) + np.random.normal(0, 0.2, n_steps)
    is_incident = np.zeros(n_steps, dtype=int)

    # Inject anomalies
    n_anomalies = int(n_steps * anomaly_fraction)
    anomaly_starts = np.random.choice(n_steps, n_anomalies, replace=False)

    for idx in anomaly_starts:
        # Random duration: 1 to 5 timesteps
        duration = np.random.randint(1, 6)
        end = min(idx + duration, n_steps)  # clamp to array bounds

        # Spike: add a value drawn from Normal(5, 1)
        # Normal: mean=5 puts the spike ~25 standard deviations above the base,
        # making it clearly distinguishable. std=1 adds natural-looking variation.
        spike_magnitude = np.random.normal(5, 1)
        metric[idx:end] += spike_magnitude
        is_incident[idx:end] = 1

    df = pd.DataFrame({
        "t": time,
        "metric": metric.astype(float),
        "is_incident": is_incident,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    df = generate_synthetic_data()
    incident_rate = df["is_incident"].mean()
    n_windows = (
        (df["is_incident"].diff() == 1).sum()
    )
    print(f"Generated {len(df):,} timesteps")
    print(f"Incident rate:    {incident_rate:.2%}")
    print(f"Incident windows: {n_windows}")
    print(f"Saved to:         {OUTPUT_PATH}")
