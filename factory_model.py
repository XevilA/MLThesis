"""
Factory Energy Balance Model (Section 3.1)
Generates E_total, E_base, E_prod, E_aux from production schedule and weather.
"""

import numpy as np
import pandas as pd

from config import FacilityParams, ProductionParams, THAI_HOLIDAYS_2024


def _is_holiday(ts) -> bool:
    return (ts.month, ts.day) in THAI_HOLIDAYS_2024


def generate_factory_energy(
    index: pd.DatetimeIndex,
    Q_prod: pd.Series,
    T_amb: pd.Series,
    params: FacilityParams,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate hourly factory energy consumption.

    E_total(t) = E_base(t) + E_prod(t) + E_aux(t)

    Returns DataFrame with columns: E_base, E_prod, E_aux, E_total
    """
    n = len(index)

    # ── 1. Baseline load (Section 3.1.1) ────────────────────────────────────
    E_base = np.zeros(n)

    for i, ts in enumerate(index):
        hour = ts.hour
        dow = ts.dayofweek

        # Determine operational regime
        is_holiday = _is_holiday(ts)
        is_weekend = dow >= 5
        is_night = hour < 6 or hour >= 22

        if is_holiday or (is_weekend and dow == 6):
            # Sunday / holiday: minimal load
            regime_factor = params.weekend_factor
        elif is_weekend:
            # Saturday: may have overtime, but base load is reduced
            regime_factor = params.night_factor if is_night else 0.65
        elif is_night:
            regime_factor = params.night_factor
        else:
            regime_factor = 1.0

        # HVAC temperature modulation
        T_excess = max(0, T_amb.iloc[i] - params.T_set)
        hvac_load = params.alpha_HVAC * T_excess

        # Stochastic variation
        noise = rng.normal(0, params.sigma_base)

        E_base[i] = max(0, params.P_base * regime_factor * (1 + hvac_load / params.P_base) + noise)

    # ── 2. Production-dependent load (Section 3.1.2) ────────────────────────
    E_prod = np.zeros(n)
    Q_values = Q_prod.values

    # Track cumulative run time per day for warm-up calculation
    run_hours = 0.0
    prev_producing = False

    for i in range(n):
        if Q_values[i] > 0:
            if not prev_producing:
                run_hours = 0  # Reset on production start

            # SEC with warm-up and degradation
            warmup_factor = 1 + params.beta_warmup * np.exp(-run_hours / params.tau_warmup)
            annual_progress = i / n  # fraction of year elapsed
            degrade_factor = 1 + params.gamma_degrade * annual_progress
            SEC_t = params.SEC_nom * warmup_factor * degrade_factor

            # Active machines (proportional to throughput relative to max)
            utilisation = min(Q_values[i] / (params.Q_nominal * 1.2), 1.0)
            n_active = max(1, int(np.ceil(utilisation * params.n_machines)))

            # Production energy + idle machine power
            E_prod[i] = SEC_t * Q_values[i] + params.P_idle * n_active

            # Add stochastic variation
            E_prod[i] += rng.normal(0, params.sigma_prod)
            E_prod[i] = max(0, E_prod[i])

            run_hours += 1
            prev_producing = True
        else:
            # No production — but idle machines may still draw power
            # if within shift hours (standby)
            hour = index[i].hour
            in_shift = 6 <= hour < 22
            dow = index[i].dayofweek
            is_workday = dow < 5

            if in_shift and is_workday:
                # Some machines on standby
                n_standby = rng.integers(2, 5)
                E_prod[i] = params.P_idle * n_standby
            else:
                E_prod[i] = 0.0

            prev_producing = False

    # ── 3. Auxiliary load ────────────────────────────────────────────────────
    # Coolant pumps, dust extraction, intermittent maintenance
    E_aux = np.maximum(
        0, rng.normal(params.P_aux_mean, params.P_aux_std, size=n)
    )

    # Auxiliary load correlates partially with production
    production_active = (Q_values > 0).astype(float)
    E_aux *= (0.4 + 0.6 * production_active)  # 40% always-on, 60% production-linked

    # ── 4. Total energy ─────────────────────────────────────────────────────
    E_total = E_base + E_prod + E_aux

    return pd.DataFrame({
        "E_base": E_base,
        "E_prod": E_prod,
        "E_aux": E_aux,
        "E_total": E_total,
    }, index=index)
