"""Tests for KPI computation (simple_metrics.py)."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.metrics import compute_kpi


def _make_df(y, sp, u=None, sampling_minutes=10):
    n = len(y)
    if u is None:
        u = [0.5] * n
    return pd.DataFrame({"y": y, "setpoint": sp, "u": u})


def test_comfort_violation_hours_exact():
    # 10 steps below comfort band (sp=22, band=±0.5 → lower=21.5)
    y = [22.0] * 10 + [21.4] * 10 + [22.0] * 10
    sp = [22.0] * 30
    df = _make_df(y, sp)
    kpi = compute_kpi(df, sampling_minutes=10)
    expected = 10 * 10 / 60  # 10 steps × 10 min / 60
    assert abs(kpi.comfort_violation_hours - expected) < 1e-9


def test_no_violations_gives_zero():
    y = [22.0] * 20
    sp = [22.0] * 20
    df = _make_df(y, sp)
    kpi = compute_kpi(df, sampling_minutes=10)
    assert kpi.comfort_violation_hours == 0.0
    assert kpi.degree_minutes_outside_band == 0.0


def test_overheating_above_band():
    y = [23.0] * 10  # 23 > 22.5 (sp=22 + 0.5)
    sp = [22.0] * 10
    df = _make_df(y, sp)
    kpi = compute_kpi(df, sampling_minutes=10)
    assert kpi.comfort_violation_hours > 0.0
    assert kpi.overheating_degree_minutes > 0.0
    assert kpi.underheating_degree_minutes == 0.0


def test_energy_proxy_nonnegative_with_heating_only():
    # With u_min=0.0, all u values >= 0; energy_proxy must be >= 0
    y = [22.0] * 20
    sp = [22.0] * 20
    u = [0.3, 0.0, 0.8, 0.5] * 5
    df = _make_df(y, sp, u=u)
    kpi = compute_kpi(df, sampling_minutes=10)
    assert kpi.energy_proxy >= 0.0


def test_energy_proxy_zero_when_no_heating():
    y = [22.0] * 10
    sp = [22.0] * 10
    u = [0.0] * 10
    df = _make_df(y, sp, u=u)
    kpi = compute_kpi(df, sampling_minutes=10)
    assert kpi.energy_proxy == 0.0


def test_skip_steps_excludes_warmup():
    # Warmup has violations, eval has none
    y_warmup = [21.0] * 5
    y_eval = [22.0] * 10
    y = y_warmup + y_eval
    sp = [22.0] * 15
    df = _make_df(y, sp)
    kpi_no_skip = compute_kpi(df, sampling_minutes=10, skip_steps=0)
    kpi_skip = compute_kpi(df, sampling_minutes=10, skip_steps=5)
    assert kpi_no_skip.comfort_violation_hours > 0.0
    assert kpi_skip.comfort_violation_hours == 0.0


def test_degree_minutes_proportional_to_deviation():
    # 10 steps at 21.0 °C with sp=22.0 → deviation = 0.5 below band (22-0.5=21.5)
    y = [21.0] * 10
    sp = [22.0] * 10
    df = _make_df(y, sp)
    kpi = compute_kpi(df, sampling_minutes=10)
    # Each step: 21.5 - 21.0 = 0.5 °C, 10 steps × 10 min = 50 deg-min
    expected = 0.5 * 10 * 10
    assert abs(kpi.degree_minutes_outside_band - expected) < 1e-9
