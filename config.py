"""
Digital Twin Configuration
All parameters from Section 3 (Tables 2–5) centralised here.
Modify this file to recalibrate the twin for different facility archetypes.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

# ── Random seed for reproducibility ──────────────────────────────────────────
SEED = 42

# ── Temporal parameters ──────────────────────────────────────────────────────
YEAR = 2024  # Simulation year
FREQ = "1h"  # Temporal resolution
TIMESTEPS = 8760  # Hours in non-leap year

# ── Location (Rayong, Thailand — Eastern Seaboard) ──────────────────────────
LATITUDE = 12.68
LONGITUDE = 101.27
ALTITUDE = 10  # metres above sea level
TIMEZONE = "Asia/Bangkok"

# ── Grid emission factor (TGO 2023) ─────────────────────────────────────────
GRID_EF = 0.4999  # tCO2e/MWh


@dataclass
class FacilityParams:
    """Table 2: Factory energy balance parameters."""
    # Baseline load
    P_base: float = 85.0          # kW, nominal baseline
    alpha_HVAC: float = 0.025     # kW/°C, HVAC temperature sensitivity
    T_set: float = 25.0           # °C, cooling setpoint
    sigma_base: float = 3.5       # kW, stochastic noise std
    night_factor: float = 0.35    # fraction of P_base during 22:00-06:00
    weekend_factor: float = 0.15  # fraction of P_base during weekends/holidays

    # Production-dependent load
    SEC_nom: float = 1.35         # kWh/part, nominal specific energy consumption
    P_idle: float = 8.5           # kW per machine, idle power draw
    n_machines: int = 12          # total CNC machines
    sigma_prod: float = 5.2       # kW, production load noise std
    Q_nominal: float = 150.0      # parts/hour (mirrors ProductionParams for utilisation calc)

    # SEC degradation model
    beta_warmup: float = 0.12     # warm-up energy overhead fraction
    tau_warmup: float = 2.0       # hours, warm-up time constant
    gamma_degrade: float = 0.03   # annual efficiency drift

    # Auxiliary load
    P_aux_mean: float = 12.0      # kW, mean auxiliary load
    P_aux_std: float = 4.0        # kW, auxiliary load std


@dataclass
class SolarParams:
    """Table 3: Solar PV system specification."""
    capacity_kWp: float = 150.0
    tilt: float = 12.0            # degrees (latitude-optimised)
    azimuth: float = 180.0        # south-facing
    module_type: str = "polycrystalline"
    soiling_loss: float = 0.03
    mismatch_loss: float = 0.02
    wiring_dc_loss: float = 0.015
    inverter_standby_loss: float = 0.02
    transformer_loss: float = 0.01
    inverter_efficiency: float = 0.965
    annual_degradation: float = 0.007  # 0.7%/year


@dataclass
class ProductionParams:
    """Section 3.3: Production schedule parameters."""
    # Shift structure
    shift_1: Tuple[int, int] = (6, 14)    # 06:00–14:00
    shift_2: Tuple[int, int] = (14, 22)   # 14:00–22:00
    saturday_overtime_prob: float = 0.3

    # Within-shift dynamics
    ramp_up_minutes: int = 30
    ramp_down_minutes: int = 30
    ramp_up_factor: float = 0.40
    ramp_down_factor: float = 0.60
    Q_nominal: float = 150.0       # parts/hour, steady-state mean
    Q_std: float = 15.0            # parts/hour, steady-state std

    # Unplanned stoppages
    stoppage_lambda: float = 0.15  # Poisson rate per shift
    stoppage_min_minutes: int = 20
    stoppage_max_minutes: int = 90

    # Seasonal demand multipliers (Q1–Q4)
    seasonal_multipliers: List[float] = field(
        default_factory=lambda: [0.90, 1.05, 1.10, 0.95]
    )

    # Thai public holidays 2024 (18 days — approximate)
    n_holidays: int = 18


@dataclass
class FaultParams:
    """Table 4: Fault injection protocol."""
    # F1: Sensor dropout
    f1_gap_min: int = 1            # hours
    f1_gap_max: int = 168          # hours
    f1_events_min: int = 20        # total events across both channels
    f1_events_max: int = 30

    # F2: Sensor drift
    f2_drift_min: float = 0.92
    f2_drift_max: float = 1.08
    f2_onset_min: int = 24         # hours
    f2_onset_max: int = 120
    f2_duration_min: int = 72      # hours
    f2_duration_max: int = 336
    f2_events_min: int = 4
    f2_events_max: int = 6

    # F3: Deliberate manipulation
    f3_events_min: int = 4
    f3_events_max: int = 8

    # F3a: Energy deflation
    f3a_delta_min: float = 0.10
    f3a_delta_max: float = 0.25
    f3a_duration_min: int = 48     # hours
    f3a_duration_max: int = 168

    # F3b: Production inflation
    f3b_delta_min: float = 0.15
    f3b_delta_max: float = 0.30
    f3b_duration_min: int = 48
    f3b_duration_max: int = 168

    # F3c: Solar inflation
    f3c_delta_min: float = 0.20
    f3c_delta_max: float = 0.40
    f3c_duration_min: int = 48
    f3c_duration_max: int = 168


# ── Thai public holidays 2024 (month, day) ───────────────────────────────────
THAI_HOLIDAYS_2024 = [
    (1, 1), (1, 2),       # New Year
    (2, 24),              # Makha Bucha (approx)
    (4, 6),               # Chakri Memorial
    (4, 13), (4, 14), (4, 15), (4, 16),  # Songkran
    (5, 1),               # Labour Day
    (5, 4),               # Coronation Day
    (5, 22),              # Visakha Bucha (approx)
    (6, 3),               # Queen Suthida Birthday
    (7, 20),              # Asalha Puja (approx)
    (7, 22),              # Buddhist Lent (approx)
    (7, 28),              # King's Birthday
    (8, 12),              # Queen Mother Birthday
    (10, 13),             # King Bhumibol Memorial
    (10, 23),             # Chulalongkorn Day
    (12, 5),              # King Bhumibol Birthday
    (12, 10),             # Constitution Day
    (12, 31),             # New Year's Eve
]
