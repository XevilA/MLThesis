"""
Production Schedule Model (Section 3.3)
Generates hourly production throughput Q_prod based on shift structure,
seasonal demand, and stochastic unplanned stoppages.
"""

import numpy as np
import pandas as pd
from config import ProductionParams, THAI_HOLIDAYS_2024, YEAR


def _is_holiday(dt) -> bool:
    """Check if a date is a Thai public holiday."""
    return (dt.month, dt.day) in THAI_HOLIDAYS_2024


def _get_seasonal_multiplier(month: int, params: ProductionParams) -> float:
    """Map month to quarterly seasonal demand multiplier."""
    quarter = (month - 1) // 3  # 0-indexed quarter
    return params.seasonal_multipliers[quarter]


def generate_production_schedule(
    index: pd.DatetimeIndex,
    params: ProductionParams,
    rng: np.random.Generator,
) -> pd.Series:
    """
    Generate hourly production throughput Q_prod (parts/hour).

    Implements:
    - Two-shift weekday operation (06:00-22:00)
    - Stochastic Saturday overtime
    - Ramp-up / ramp-down within each shift
    - Poisson-distributed unplanned stoppages
    - Seasonal demand modulation
    - Thai public holidays (zero production)
    """
    n = len(index)
    Q_prod = np.zeros(n)

    # Pre-compute daily properties
    dates = index.normalize().unique()

    # Decide Saturday overtime for each Saturday
    saturdays = [d for d in dates if d.dayofweek == 5]
    saturday_overtime = {
        d: rng.random() < params.saturday_overtime_prob for d in saturdays
    }

    for i, ts in enumerate(index):
        hour = ts.hour
        dow = ts.dayofweek  # 0=Mon, 6=Sun
        date_key = ts.normalize()

        # Check holiday
        if _is_holiday(ts):
            Q_prod[i] = 0.0
            continue

        # Check if production is scheduled this hour
        is_weekday = dow < 5
        is_saturday_ot = dow == 5 and saturday_overtime.get(date_key, False)
        is_production_day = is_weekday or is_saturday_ot

        if not is_production_day:
            Q_prod[i] = 0.0
            continue

        # Check if within shift hours
        s1_start, s1_end = params.shift_1
        s2_start, s2_end = params.shift_2

        in_shift1 = s1_start <= hour < s1_end
        in_shift2 = s2_start <= hour < s2_end

        if not (in_shift1 or in_shift2):
            Q_prod[i] = 0.0
            continue

        # Determine position within shift
        if in_shift1:
            shift_start, shift_end = s1_start, s1_end
        else:
            shift_start, shift_end = s2_start, s2_end

        hours_into_shift = hour - shift_start
        shift_duration = shift_end - shift_start

        # Ramp-up / steady-state / ramp-down
        # (ramp times are in minutes but we operate at hourly resolution,
        #  so first hour ≈ ramp-up, last hour ≈ ramp-down)
        if hours_into_shift == 0:
            # First hour: ramp-up
            throughput_factor = params.ramp_up_factor
        elif hours_into_shift == shift_duration - 1:
            # Last hour: ramp-down
            throughput_factor = params.ramp_down_factor
        else:
            throughput_factor = 1.0

        # Base throughput with stochastic variation
        Q_base = rng.normal(params.Q_nominal, params.Q_std)
        Q_base = max(Q_base, 0)

        # Apply seasonal multiplier
        seasonal = _get_seasonal_multiplier(ts.month, params)

        # Apply ramp factor
        Q_hour = Q_base * throughput_factor * seasonal

        # Saturday overtime runs at reduced capacity (single shift, 06:00-14:00 only)
        if is_saturday_ot:
            if not in_shift1:
                Q_hour = 0.0
            else:
                Q_hour *= 0.85  # reduced Saturday efficiency

        Q_prod[i] = max(Q_hour, 0)

    # Inject unplanned stoppages (Poisson-distributed per shift)
    Q_prod = _inject_stoppages(Q_prod, index, params, rng)

    return pd.Series(Q_prod, index=index, name="Q_prod")


def _inject_stoppages(
    Q_prod: np.ndarray,
    index: pd.DatetimeIndex,
    params: ProductionParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Inject Poisson-distributed unplanned stoppages into production."""
    Q_out = Q_prod.copy()
    n = len(index)

    # Identify production hours
    production_mask = Q_prod > 0
    production_indices = np.where(production_mask)[0]

    if len(production_indices) == 0:
        return Q_out

    # Count approximate number of shifts
    # Each shift is ~8 hours of production
    total_production_hours = production_mask.sum()
    n_shifts = total_production_hours / 8

    # Number of stoppage events
    n_stoppages = rng.poisson(params.stoppage_lambda * n_shifts)

    for _ in range(n_stoppages):
        # Random start point during production
        start_idx = rng.choice(production_indices)

        # Duration in minutes → convert to hours (round up)
        duration_min = rng.integers(
            params.stoppage_min_minutes, params.stoppage_max_minutes + 1
        )
        duration_hours = max(1, int(np.ceil(duration_min / 60)))

        # Zero out production for stoppage duration
        end_idx = min(start_idx + duration_hours, n)
        Q_out[start_idx:end_idx] = 0.0

    return Q_out


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    index = pd.date_range(f"{YEAR}-01-01", periods=8760, freq="1h", tz="Asia/Bangkok")
    params = ProductionParams()
    Q = generate_production_schedule(index, params, rng)
    total_parts = Q.sum()
    prod_hours = (Q > 0).sum()
    print(f"Total annual production: {total_parts:,.0f} parts")
    print(f"Production hours: {prod_hours:,} / {len(index):,}")
    print(f"Mean throughput (when active): {Q[Q > 0].mean():.1f} parts/hr")
    print(f"Max throughput: {Q.max():.1f} parts/hr")
