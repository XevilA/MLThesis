"""
Fault Injection Protocol (Section 3.4)
Injects controlled faults into clean telemetry and generates ground-truth log.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

from config import FaultParams


@dataclass
class FaultEvent:
    """Ground-truth annotation for a single injected fault."""
    fault_type: str       # F1, F2, F3a, F3b, F3c
    start_idx: int
    end_idx: int
    affected_channels: List[str]
    magnitude: float      # NaN for F1; drift factor for F2; delta for F3
    description: str


def inject_faults(
    df_clean: pd.DataFrame,
    params: FaultParams,
    rng: np.random.Generator,
) -> tuple:
    """
    Inject faults into clean telemetry.

    Args:
        df_clean: DataFrame with columns [E_grid, E_DG, Q_prod, T_amb, GHI]
        params: FaultParams configuration
        rng: numpy random generator

    Returns:
        df_faulted: DataFrame with faults applied
        fault_log: List[FaultEvent] ground-truth annotations
    """
    df = df_clean.copy()
    n = len(df)
    fault_log = []

    # Track occupied timesteps to avoid overlapping faults
    occupied = np.zeros(n, dtype=bool)

    # ── F1: Sensor Dropout ──────────────────────────────────────────────────
    n_f1 = rng.integers(params.f1_events_min, params.f1_events_max + 1)

    for _ in range(n_f1):
        gap_len = rng.integers(params.f1_gap_min, params.f1_gap_max + 1)
        channel = rng.choice(["E_grid", "E_DG"])

        # Find a valid start position
        start = _find_free_window(occupied, gap_len, n, rng, max_attempts=100)
        if start is None:
            continue

        end = min(start + gap_len, n)
        occupied[start:end] = True

        # Inject NaN
        df.loc[df.index[start:end], channel] = np.nan

        fault_log.append(FaultEvent(
            fault_type="F1",
            start_idx=start,
            end_idx=end,
            affected_channels=[channel],
            magnitude=np.nan,
            description=f"Sensor dropout: {channel}, {gap_len}h gap"
        ))

    # ── F2: Sensor Drift ────────────────────────────────────────────────────
    n_f2 = rng.integers(params.f2_events_min, params.f2_events_max + 1)

    for _ in range(n_f2):
        duration = rng.integers(params.f2_duration_min, params.f2_duration_max + 1)
        onset = rng.integers(params.f2_onset_min, params.f2_onset_max + 1)
        total_len = onset + duration
        drift_factor = rng.uniform(params.f2_drift_min, params.f2_drift_max)
        channel = rng.choice(["E_grid", "E_DG"])

        start = _find_free_window(occupied, total_len, n, rng, max_attempts=100)
        if start is None:
            continue

        end = min(start + total_len, n)
        occupied[start:end] = True

        # Apply gradual onset then sustained drift
        for i in range(start, end):
            local_t = i - start
            if local_t < onset:
                # Gradual onset: linear ramp from 1.0 to drift_factor
                progress = local_t / onset
                current_factor = 1.0 + (drift_factor - 1.0) * progress
            else:
                current_factor = drift_factor

            df.iloc[i, df.columns.get_loc(channel)] *= current_factor

        fault_log.append(FaultEvent(
            fault_type="F2",
            start_idx=start,
            end_idx=end,
            affected_channels=[channel],
            magnitude=drift_factor,
            description=f"Sensor drift: {channel}, factor={drift_factor:.3f}, "
                        f"onset={onset}h, duration={duration}h"
        ))

    # ── F3: Deliberate Manipulation ─────────────────────────────────────────
    n_f3 = rng.integers(params.f3_events_min, params.f3_events_max + 1)

    # Distribute across sub-types (roughly equal, with randomness)
    subtypes = rng.choice(["F3a", "F3b", "F3c"], size=n_f3)

    for subtype in subtypes:
        if subtype == "F3a":
            _inject_f3a(df, params, rng, occupied, fault_log, n)
        elif subtype == "F3b":
            _inject_f3b(df, params, rng, occupied, fault_log, n)
        elif subtype == "F3c":
            _inject_f3c(df, params, rng, occupied, fault_log, n)

    return df, fault_log


def _inject_f3a(df, params, rng, occupied, fault_log, n):
    """F3a: Energy deflation — reduce E_grid while holding Q_prod constant."""
    delta = rng.uniform(params.f3a_delta_min, params.f3a_delta_max)
    duration = rng.integers(params.f3a_duration_min, params.f3a_duration_max + 1)

    start = _find_free_window(occupied, duration, n, rng, max_attempts=100)
    if start is None:
        return

    end = min(start + duration, n)
    occupied[start:end] = True

    suppression = 1.0 - delta
    df.iloc[start:end, df.columns.get_loc("E_grid")] *= suppression

    fault_log.append(FaultEvent(
        fault_type="F3a",
        start_idx=start,
        end_idx=end,
        affected_channels=["E_grid"],
        magnitude=delta,
        description=f"Energy deflation: E_grid reduced by {delta*100:.1f}% "
                    f"for {duration}h"
    ))


def _inject_f3b(df, params, rng, occupied, fault_log, n):
    """F3b: Production inflation — increase Q_prod while holding energy constant."""
    delta = rng.uniform(params.f3b_delta_min, params.f3b_delta_max)
    duration = rng.integers(params.f3b_duration_min, params.f3b_duration_max + 1)

    start = _find_free_window(occupied, duration, n, rng, max_attempts=100)
    if start is None:
        return

    end = min(start + duration, n)
    occupied[start:end] = True

    inflation = 1.0 + delta
    df.iloc[start:end, df.columns.get_loc("Q_prod")] *= inflation

    fault_log.append(FaultEvent(
        fault_type="F3b",
        start_idx=start,
        end_idx=end,
        affected_channels=["Q_prod"],
        magnitude=delta,
        description=f"Production inflation: Q_prod increased by {delta*100:.1f}% "
                    f"for {duration}h"
    ))


def _inject_f3c(df, params, rng, occupied, fault_log, n):
    """F3c: Solar inflation — increase E_DG during daylight hours."""
    delta = rng.uniform(params.f3c_delta_min, params.f3c_delta_max)
    duration = rng.integers(params.f3c_duration_min, params.f3c_duration_max + 1)

    start = _find_free_window(occupied, duration, n, rng, max_attempts=100)
    if start is None:
        return

    end = min(start + duration, n)
    occupied[start:end] = True

    # Only inflate during daylight hours (when E_DG > 0)
    inflation = 1.0 + delta
    for i in range(start, end):
        if df.iloc[i, df.columns.get_loc("E_DG")] > 0:
            df.iloc[i, df.columns.get_loc("E_DG")] *= inflation

    # Also adjust E_grid downward (since inflated solar reduces grid import)
    for i in range(start, end):
        solar_increase = df.iloc[i, df.columns.get_loc("E_DG")] * (1 - 1/inflation)
        current_grid = df.iloc[i, df.columns.get_loc("E_grid")]
        df.iloc[i, df.columns.get_loc("E_grid")] = max(0, current_grid - solar_increase)

    fault_log.append(FaultEvent(
        fault_type="F3c",
        start_idx=start,
        end_idx=end,
        affected_channels=["E_DG", "E_grid"],
        magnitude=delta,
        description=f"Solar inflation: E_DG increased by {delta*100:.1f}%, "
                    f"E_grid adjusted downward, for {duration}h"
    ))


def _find_free_window(
    occupied: np.ndarray, length: int, n: int,
    rng: np.random.Generator, max_attempts: int = 100,
) -> int:
    """Find a contiguous window of `length` that doesn't overlap existing faults."""
    for _ in range(max_attempts):
        start = rng.integers(0, max(1, n - length))
        end = min(start + length, n)
        if not occupied[start:end].any():
            return start
    return None


def fault_log_to_dataframe(fault_log: List[FaultEvent], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Convert fault log to a DataFrame for export."""
    records = []
    for f in fault_log:
        records.append({
            "fault_type": f.fault_type,
            "start_time": index[f.start_idx],
            "end_time": index[min(f.end_idx - 1, len(index) - 1)],
            "duration_hours": f.end_idx - f.start_idx,
            "affected_channels": ", ".join(f.affected_channels),
            "magnitude": f.magnitude,
            "description": f.description,
        })
    return pd.DataFrame(records)


def generate_anomaly_labels(
    fault_log: List[FaultEvent], n: int
) -> np.ndarray:
    """
    Generate per-timestep anomaly labels for Stage 2 evaluation.
    0 = normal, 1 = anomalous (F2 or F3 only; F1 is handled by Stage 1).
    """
    labels = np.zeros(n, dtype=int)
    for f in fault_log:
        if f.fault_type in ("F2", "F3a", "F3b", "F3c"):
            labels[f.start_idx:f.end_idx] = 1
    return labels
