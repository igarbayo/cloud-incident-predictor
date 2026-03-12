# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Ignacio Garbayo Fernandez

"""
Synthetic cloud metrics generator.

Generates a time series that simulates a cloud metric (e.g. CPU usage,
request latency) with realistic structure and six distinct incident types.

Signal structure:
  - Base: sinusoidal wave simulating diurnal patterns (day/night cycles)
  - Noise: Gaussian noise simulating measurement uncertainty
  - Incidents: six anomaly types injected at random positions

Incident types:
  - spike:               sudden upward burst (DDoS, flash sale, traffic spike)
  - threshold_breach:    metric sustained above a fixed upper limit (CPU saturation)
  - gradual_degradation: slow ramp-up over time (memory leak, disk fill)
  - level_shift:         abrupt jump to a new elevated baseline (config change, partial failure)
  - drop:                metric collapses to near-zero (service crash, network blackout)
  - oscillation:         increased variance / instability (thrashing, feedback loop)

Output schema (saved to data/synthetic_metrics.csv):
  t             : int    — timestep index 0..N-1
  metric        : float  — simulated metric value (e.g. normalised CPU %)
  is_incident   : int    — 1 during any incident window, 0 otherwise
  incident_type : str    — one of the six types above, or '' for normal timesteps
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants — single source of truth, referenced by tests
# ---------------------------------------------------------------------------

N_STEPS = 10_000
ANOMALY_FRACTION = 0.01   # fraction of timesteps that are anomaly start points
RANDOM_SEED = 42
OUTPUT_PATH = Path("data/synthetic_metrics.csv")

INCIDENT_TYPES = [
    'spike',
    'threshold_breach',
    'gradual_degradation',
    'level_shift',
    'drop',
    'oscillation',
]


# ---------------------------------------------------------------------------
# Per-type injection functions
# Each function modifies metric[idx:end] in place and returns end index.
# ---------------------------------------------------------------------------

def _inject_spike(metric, idx, n_steps, rng):
    """Sudden upward burst. Duration 1–5 steps. Magnitude ~5 units above normal."""
    duration = int(rng.integers(1, 6))
    end = min(idx + duration, n_steps)
    magnitude = float(rng.normal(5, 1))
    metric[idx:end] += magnitude
    return end


def _inject_threshold_breach(metric, idx, n_steps, rng):
    """
    Metric held above a fixed saturation level for 8–15 steps.
    Simulates CPU pinned at 100% or latency exceeding SLA threshold.
    """
    duration = int(rng.integers(8, 16))
    end = min(idx + duration, n_steps)
    # Set metric to an elevated plateau (well above the normal [-1.2, 1.2] range)
    plateau = 2.0 + rng.normal(0, 0.1, end - idx)
    metric[idx:end] = plateau
    return end


def _inject_gradual_degradation(metric, idx, n_steps, rng):
    """
    Slow linear ramp up over 15–25 steps.
    Simulates memory leak, disk fill, or connection pool exhaustion.
    """
    duration = int(rng.integers(15, 26))
    end = min(idx + duration, n_steps)
    actual = end - idx
    peak = float(rng.normal(3, 0.5))
    ramp = np.linspace(0, peak, actual)
    metric[idx:end] += ramp
    return end


def _inject_level_shift(metric, idx, n_steps, rng):
    """
    Abrupt jump to a new elevated baseline for 10–20 steps.
    Simulates a configuration change, canary gone wrong, or partial failure.
    """
    duration = int(rng.integers(10, 21))
    end = min(idx + duration, n_steps)
    offset = float(rng.normal(2, 0.3))
    metric[idx:end] += offset
    return end


def _inject_drop(metric, idx, n_steps, rng):
    """
    Metric collapses to near-zero for 3–8 steps.
    Simulates service crash, network blackout, or silent failure.
    """
    duration = int(rng.integers(3, 9))
    end = min(idx + duration, n_steps)
    metric[idx:end] *= 0.05
    return end


def _inject_oscillation(metric, idx, n_steps, rng):
    """
    High-frequency noise injected for 5–12 steps (std ~2×, vs normal ~0.2×).
    Simulates thrashing, feedback loop, or oscillating auto-scaler.
    """
    duration = int(rng.integers(5, 13))
    end = min(idx + duration, n_steps)
    noise = rng.normal(0, 2.0, end - idx)
    metric[idx:end] += noise
    return end


_INJECTORS = {
    'spike':               _inject_spike,
    'threshold_breach':    _inject_threshold_breach,
    'gradual_degradation': _inject_gradual_degradation,
    'level_shift':         _inject_level_shift,
    'drop':                _inject_drop,
    'oscillation':         _inject_oscillation,
}


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    n_steps: int = N_STEPS,
    anomaly_fraction: float = ANOMALY_FRACTION,
    seed: int = RANDOM_SEED,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Generate a synthetic time series with six distinct labelled incident types.

    The base signal is a sinusoidal wave (simulating diurnal load patterns) plus
    Gaussian noise (measurement uncertainty). Anomaly starts are drawn uniformly
    at random; each start is assigned one of the six incident types at random.

    Args:
        n_steps:          Total number of timesteps to generate.
        anomaly_fraction: Fraction of timesteps chosen as anomaly start points.
                          With six types and varied durations, the resulting
                          incident rate will be roughly 8–13% of all timesteps.
        seed:             Random seed for full reproducibility.
        output_path:      Where to save the CSV. Relative to working directory.

    Returns:
        DataFrame with columns: t, metric, is_incident, incident_type.

    Example output around a spike incident:
        t=137: metric=0.712,  is_incident=0, incident_type=''
        t=138: metric=5.834,  is_incident=1, incident_type='spike'
        t=139: metric=6.102,  is_incident=1, incident_type='spike'
        t=140: metric=0.534,  is_incident=0, incident_type=''
    """
    rng = np.random.default_rng(seed)
    time = np.arange(n_steps)

    # Base signal: sinusoidal + Gaussian noise
    metric = np.sin(time * 0.1) + rng.normal(0, 0.2, n_steps)
    is_incident = np.zeros(n_steps, dtype=int)
    incident_type = np.full(n_steps, '', dtype=object)

    # Select anomaly start positions and inject each type
    n_anomalies = int(n_steps * anomaly_fraction)
    anomaly_starts = rng.choice(n_steps, n_anomalies, replace=False)

    for idx in anomaly_starts:
        itype = str(rng.choice(INCIDENT_TYPES))
        end = _INJECTORS[itype](metric, int(idx), n_steps, rng)
        is_incident[idx:end] = 1
        incident_type[idx:end] = itype

    df = pd.DataFrame({
        "t":             time,
        "metric":        metric.astype(float),
        "is_incident":   is_incident,
        "incident_type": incident_type,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    df = generate_synthetic_data()
    incident_rate = df["is_incident"].mean()
    n_windows = (df["is_incident"].diff() == 1).sum()
    print(f"Generated {len(df):,} timesteps")
    print(f"Incident rate:    {incident_rate:.2%}")
    print(f"Incident windows: {n_windows}")
    print()
    print("Breakdown by incident type:")
    for itype in INCIDENT_TYPES:
        count = (df["incident_type"] == itype).sum()
        print(f"  {itype:<22} {count:>5} timesteps  ({count/len(df):.2%})")
    print()
    print(f"Saved to: {OUTPUT_PATH}")
