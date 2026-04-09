from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KPIResult:
    comfort_violation_hours: float
    rmse: float
    mae: float
    energy_proxy: float
    degree_minutes_outside_band: float
    underheating_degree_minutes: float
    overheating_degree_minutes: float


def compute_kpi(df: pd.DataFrame, sampling_minutes: int, skip_steps: int = 0) -> KPIResult:
    if skip_steps > 0:
        df = df.iloc[skip_steps:].copy()

    y = df["y"].to_numpy(dtype=float)
    sp = df["setpoint"].to_numpy(dtype=float)
    u = df["u"].to_numpy(dtype=float)

    lower = sp - 0.5
    upper = sp + 0.5
    violations = ((y < lower) | (y > upper)).astype(float)

    under = np.maximum(0.0, lower - y)
    over = np.maximum(0.0, y - upper)
    outside = under + over

    underheating_degree_minutes = np.sum(under) * sampling_minutes
    overheating_degree_minutes = np.sum(over) * sampling_minutes
    degree_minutes_outside_band = np.sum(outside) * sampling_minutes

    err = y - sp

    return KPIResult(
        comfort_violation_hours=float(violations.sum() * sampling_minutes / 60.0),
        rmse=float(np.sqrt(np.mean(err**2))),
        mae=float(np.mean(np.abs(err))),
        energy_proxy=float(np.sum(u) * sampling_minutes / 60.0),
        degree_minutes_outside_band=float(degree_minutes_outside_band),
        underheating_degree_minutes=float(underheating_degree_minutes),
        overheating_degree_minutes=float(overheating_degree_minutes),
    )
