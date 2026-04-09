"""Tests for scenario reproducibility — same seed → same output."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from app.simulation import ScenarioSpec, make_scenario


def test_same_seed_same_scenario():
    spec = ScenarioSpec(name="cold_day", steps=100, sampling_minutes=10)
    s1 = make_scenario(spec, day_setpoint=22.0, night_setpoint=20.5, seed=42)
    s2 = make_scenario(spec, day_setpoint=22.0, night_setpoint=20.5, seed=42)
    assert (s1["T_out"].values == s2["T_out"].values).all()
    assert (s1["solar"].values == s2["solar"].values).all()
    assert (s1["occupancy"].values == s2["occupancy"].values).all()


def test_different_seed_different_t_out():
    spec = ScenarioSpec(name="cold_day", steps=100, sampling_minutes=10)
    s1 = make_scenario(spec, day_setpoint=22.0, night_setpoint=20.5, seed=42)
    s2 = make_scenario(spec, day_setpoint=22.0, night_setpoint=20.5, seed=99)
    assert not (s1["T_out"].values == s2["T_out"].values).all()


def test_different_scenario_type_different_t_out():
    spec_cold = ScenarioSpec(name="cold_day", steps=100, sampling_minutes=10)
    spec_hot = ScenarioSpec(name="summer_heatwave", steps=100, sampling_minutes=10)
    cold = make_scenario(spec_cold, day_setpoint=22.0, night_setpoint=20.5, seed=42)
    hot = make_scenario(spec_hot, day_setpoint=22.0, night_setpoint=20.5, seed=42)
    assert cold["T_out"].mean() < hot["T_out"].mean()


def test_scenario_has_required_columns():
    spec = ScenarioSpec(name="spring_day", steps=50, sampling_minutes=10)
    df = make_scenario(spec, day_setpoint=22.0, night_setpoint=20.5, seed=0)
    for col in ["timestamp", "T_out", "solar", "occupancy", "setpoint",
                "occupancy_actual", "unexpected_occupancy"]:
        assert col in df.columns, f"Missing column: {col}"


def test_scenario_no_nan():
    spec = ScenarioSpec(name="mixed_day", steps=144, sampling_minutes=10)
    df = make_scenario(spec, day_setpoint=22.0, night_setpoint=20.5, seed=7)
    assert not df.isnull().any().any(), "Scenario contains NaN values"


def test_occupancy_nonnegative():
    spec = ScenarioSpec(name="sunny_day", steps=144, sampling_minutes=10)
    df = make_scenario(spec, day_setpoint=22.0, night_setpoint=20.5, seed=3)
    assert (df["occupancy"] >= 0).all()
    assert (df["occupancy_actual"] >= 0).all()
