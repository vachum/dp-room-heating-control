"""Fail-fast validation tests for production scenario validation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.scenario_utils import validate_scenario_df


def _make_valid_df(n=10, sampling_minutes=10):
    start = pd.Timestamp("2025-01-01")
    timestamps = pd.date_range(start, periods=n, freq=f"{sampling_minutes}min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "T_out": np.full(n, 5.0),
            "solar": np.full(n, 200.0),
            "occupancy": np.full(n, 2.0),
            "setpoint": np.full(n, 22.0),
        }
    )


def test_valid_standard_scenario_passes():
    df = _make_valid_df()
    validate_scenario_df(df, mode="standard")


def test_missing_required_column_raises():
    df = _make_valid_df().drop(columns=["T_out"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_scenario_df(df, mode="standard")


def test_missing_setpoint_raises():
    df = _make_valid_df().drop(columns=["setpoint"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_scenario_df(df, mode="standard")


def test_hidden_occupancy_mode_requires_extra_columns():
    df = _make_valid_df()
    with pytest.raises(ValueError, match="occupancy_actual"):
        validate_scenario_df(df, mode="hidden_occupancy")


def test_hidden_occupancy_mode_passes_with_extra_columns():
    df = _make_valid_df()
    df["occupancy_actual"] = df["occupancy"]
    df["unexpected_occupancy"] = 0.0
    validate_scenario_df(df, mode="hidden_occupancy")


def test_nan_in_critical_column_raises():
    df = _make_valid_df()
    df.loc[3, "T_out"] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        validate_scenario_df(df, mode="standard")


def test_irregular_timestamps_raises():
    df = _make_valid_df()
    timestamps = list(df["timestamp"])
    timestamps[5] = timestamps[4] + pd.Timedelta(minutes=25)
    for i in range(6, len(timestamps)):
        timestamps[i] = timestamps[i - 1] + pd.Timedelta(minutes=10)
    df["timestamp"] = timestamps
    with pytest.raises(ValueError, match="irregular timestamp sampling"):
        validate_scenario_df(df, mode="standard")


def test_explicit_sampling_minutes_mismatch_raises():
    df = _make_valid_df(sampling_minutes=10)
    with pytest.raises(ValueError, match="expected 15 min"):
        validate_scenario_df(
            df,
            mode="standard",
            expected_sampling_minutes=15,
        )
