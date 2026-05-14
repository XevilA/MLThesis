"""
Digital Twin — Main Orchestrator
Generates the complete dataset for the ML pipeline (Section 3.6).

Usage:
    python main.py

Outputs:
    - clean_telemetry.csv      : Gold-standard pre-injection data
    - faulted_telemetry.csv    : Data with injected faults (ML pipeline input)
    - fault_log.csv            : Ground-truth fault annotations
    - anomaly_labels.csv       : Per-timestep binary labels for Stage 2
    - validation_report.txt    : Calibration validation results
"""

import sys
import os
import numpy as np
import pandas as pd

# Ensure package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SEED, YEAR, FREQ, TIMESTEPS, GRID_EF, TIMEZONE,
    FacilityParams, SolarParams, ProductionParams, FaultParams,
)
from solar_model import simulate_solar_pv
from production_model import generate_production_schedule
from factory_model import generate_factory_energy
from fault_injection import inject_faults, fault_log_to_dataframe, generate_anomaly_labels
from validation import validate_twin


def main(output_dir: str = None):
    """Generate the complete digital twin dataset."""

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("DIGITAL TWIN — Tier-2 Thai Manufacturing SME")
    print("Eastern Seaboard (Rayong-Chonburi), Thailand")
    print("=" * 65)

    # Initialise
    rng = np.random.default_rng(SEED)
    facility = FacilityParams()
    solar = SolarParams()
    production = ProductionParams()
    faults = FaultParams()

    # Time index
    index = pd.date_range(
        start=f"{YEAR}-01-01",
        periods=TIMESTEPS,
        freq=FREQ,
        tz=TIMEZONE,
    )
    print(f"\nSimulation period: {index[0]} to {index[-1]}")
    print(f"Timesteps: {TIMESTEPS:,} ({FREQ} resolution)")

    # ── Step 1: Solar PV Generation ─────────────────────────────────────────
    print("\n[1/5] Generating solar PV output...")
    solar_df = simulate_solar_pv(index, solar, rng)
    print(f"  Annual solar yield: {solar_df['E_DG'].sum()/1000:.0f} MWh")

    # ── Step 2: Production Schedule ─────────────────────────────────────────
    print("[2/5] Generating production schedule...")
    Q_prod = generate_production_schedule(index, production, rng)
    prod_hours = (Q_prod > 0).sum()
    print(f"  Annual production: {Q_prod.sum():,.0f} parts")
    print(f"  Production hours: {prod_hours:,} ({prod_hours/TIMESTEPS*100:.1f}%)")

    # ── Step 3: Factory Energy Balance ──────────────────────────────────────
    print("[3/5] Computing factory energy balance...")
    energy_df = generate_factory_energy(
        index, Q_prod, solar_df["T_amb"], facility, rng
    )

    # Compute grid consumption (residual after solar)
    E_grid = np.maximum(0, energy_df["E_total"].values - solar_df["E_DG"].values)

    print(f"  Annual energy consumption: {energy_df['E_total'].sum()/1000:.0f} MWh")
    print(f"  Annual grid import: {E_grid.sum()/1000:.0f} MWh")
    print(f"  Solar self-consumption share: "
          f"{(1 - E_grid.sum()/energy_df['E_total'].sum())*100:.1f}%")

    # ── Assemble clean telemetry ────────────────────────────────────────────
    df_clean = pd.DataFrame({
        "E_grid": E_grid,
        "E_DG": solar_df["E_DG"].values,
        "Q_prod": Q_prod.values,
        "E_total": energy_df["E_total"].values,
        "T_amb": solar_df["T_amb"].values,
        "GHI": solar_df["GHI"].values,
    }, index=index)

    # Emissions
    annual_emissions = (df_clean["E_grid"].sum() / 1000) * GRID_EF
    print(f"  Annual Scope 2 emissions: {annual_emissions:.0f} tCO2e "
          f"(EF={GRID_EF} tCO2e/MWh)")

    # ── Step 4: Validation ──────────────────────────────────────────────────
    print("\n[4/5] Running calibration validation...")
    val_results = validate_twin(df_clean, verbose=True)

    # ── Step 5: Fault Injection ─────────────────────────────────────────────
    print(f"\n[5/5] Injecting faults...")

    # Prepare DataFrame for fault injection (only modifiable columns)
    df_for_injection = df_clean[["E_grid", "E_DG", "Q_prod", "T_amb", "GHI"]].copy()
    df_faulted, fault_log = inject_faults(df_for_injection, faults, rng)

    # Recalculate E_total for faulted data (may be affected by E_grid changes)
    # Note: E_total in the faulted set reflects the *reported* total, not physical
    df_faulted["E_total"] = df_faulted["E_grid"] + df_faulted["E_DG"]

    # Generate anomaly labels
    anomaly_labels = generate_anomaly_labels(fault_log, TIMESTEPS)
    fault_df = fault_log_to_dataframe(fault_log, index)

    # Fault summary
    n_f1 = sum(1 for f in fault_log if f.fault_type == "F1")
    n_f2 = sum(1 for f in fault_log if f.fault_type == "F2")
    n_f3 = sum(1 for f in fault_log if f.fault_type.startswith("F3"))
    n_nan = df_faulted["E_grid"].isna().sum() + df_faulted["E_DG"].isna().sum()
    n_anomaly = anomaly_labels.sum()
    clean_pct = (1 - (n_nan + n_anomaly) / (TIMESTEPS * 2)) * 100

    print(f"  F1 (dropout) events: {n_f1}")
    print(f"  F2 (drift) events: {n_f2}")
    print(f"  F3 (manipulation) events: {n_f3}")
    print(f"  Total NaN timesteps: {n_nan:,}")
    print(f"  Anomalous timesteps (Stage 2): {n_anomaly:,} "
          f"({n_anomaly/TIMESTEPS*100:.1f}%)")
    print(f"  Approximate clean data: {clean_pct:.0f}%")

    # ── Export ──────────────────────────────────────────────────────────────
    print(f"\nExporting to {output_dir}...")

    # Remove timezone for CSV compatibility
    df_clean_out = df_clean.copy()
    df_clean_out.index = df_clean_out.index.tz_localize(None)
    df_faulted_out = df_faulted.copy()
    df_faulted_out.index = df_faulted_out.index.tz_localize(None)

    df_clean_out.to_csv(os.path.join(output_dir, "clean_telemetry.csv"))
    df_faulted_out.to_csv(os.path.join(output_dir, "faulted_telemetry.csv"))
    fault_df.to_csv(os.path.join(output_dir, "fault_log.csv"), index=False)

    labels_df = pd.DataFrame({
        "timestamp": df_clean_out.index,
        "anomaly_label": anomaly_labels,
    })
    labels_df.to_csv(os.path.join(output_dir, "anomaly_labels.csv"), index=False)

    # Validation report
    report_path = os.path.join(output_dir, "validation_report.txt")
    with open(report_path, "w") as f:
        f.write("DIGITAL TWIN VALIDATION REPORT\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Seed: {SEED}\n\n")
        for k, v in val_results.items():
            if isinstance(v, dict):
                f.write(f"{k}:\n")
                for kk, vv in v.items():
                    f.write(f"  {kk}: {vv}\n")
            else:
                f.write(f"{k}: {v}\n")

    print(f"  clean_telemetry.csv    ({os.path.getsize(os.path.join(output_dir, 'clean_telemetry.csv'))/1024:.0f} KB)")
    print(f"  faulted_telemetry.csv  ({os.path.getsize(os.path.join(output_dir, 'faulted_telemetry.csv'))/1024:.0f} KB)")
    print(f"  fault_log.csv          ({len(fault_df)} events)")
    print(f"  anomaly_labels.csv     ({n_anomaly:,} positive labels)")
    print(f"  validation_report.txt")

    print(f"\n{'=' * 65}")
    print("DIGITAL TWIN GENERATION COMPLETE")
    print(f"{'=' * 65}")

    return df_clean, df_faulted, fault_log, anomaly_labels, val_results


if __name__ == "__main__":
    main()
