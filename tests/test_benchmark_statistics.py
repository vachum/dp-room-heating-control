"""Tests for run-level and scenario-level benchmark statistics."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from benchmark_statistics import (
    _compute_scope_stats,
    _pair_key_columns,
    _resolve_input_csv,
)


def _make_scenario_level_df() -> pd.DataFrame:
    controllers = {
        "hybrid_mpc": 0.0,
        "pure_mpc": 0.4,
        "pid": 0.8,
        "onoff": 1.2,
    }
    rows = []
    scenarios = ["cold_day", "sunny_day", "spring_day"]
    for seed_idx, seed in enumerate([40, 41]):
        for variant in [0, 1]:
            for scenario_idx, scenario in enumerate(scenarios):
                base = 1.0 + seed_idx * 0.2 + variant * 0.1 + scenario_idx * 0.05
                for controller, penalty in controllers.items():
                    rows.append(
                        {
                            "room_id": "small_office",
                            "seed": seed,
                            "variant": variant,
                            "scenario": scenario,
                            "controller": controller,
                            "comfort_violation_hours": base + penalty,
                            "degree_minutes_outside_band": 10.0 * (base + penalty),
                            "rmse": 0.5 + base + penalty,
                            "mae": 0.3 + base + penalty,
                            "energy_proxy": 5.0 + base + penalty,
                        }
                    )
    return pd.DataFrame(rows)


def _make_run_level_df(scenario_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "comfort_violation_hours",
        "degree_minutes_outside_band",
        "rmse",
        "mae",
        "energy_proxy",
    ]
    return (
        scenario_df.groupby(["room_id", "seed", "variant", "controller"], as_index=False)[
            metric_cols
        ]
        .mean()
        .reset_index(drop=True)
    )


def test_resolve_input_csv_defaults_for_run_and_scenario():
    assert _resolve_input_csv(None, "artifacts", "tag", "run") == (
        Path("artifacts") / "tag" / "benchmark_combined" / "benchmark_detailed_all_rooms.csv"
    )
    assert _resolve_input_csv(None, "artifacts", "tag", "scenario") == (
        Path("artifacts")
        / "tag"
        / "benchmark_combined"
        / "benchmark_detailed_scenarios_all_rooms.csv"
    )


def test_pair_key_columns_differ_between_run_and_scenario():
    assert _pair_key_columns("run") == ["room_id", "seed", "variant"]
    assert _pair_key_columns("scenario") == [
        "room_id",
        "seed",
        "variant",
        "scenario",
    ]


def test_scenario_level_analysis_has_more_pairs_than_run_level():
    scenario_df = _make_scenario_level_df()
    run_df = _make_run_level_df(scenario_df)

    run_stats = _compute_scope_stats(
        run_df,
        scope="combined",
        analysis_unit="run",
        key_cols=_pair_key_columns("run"),
    )
    scenario_stats = _compute_scope_stats(
        scenario_df,
        scope="combined",
        analysis_unit="scenario",
        key_cols=_pair_key_columns("scenario"),
    )

    run_pairs = int(
        run_stats.loc[
            (run_stats["controller_vs"] == "pure_mpc") & (run_stats["kpi"] == "rmse"),
            "n_pairs",
        ].iloc[0]
    )
    scenario_pairs = int(
        scenario_stats.loc[
            (scenario_stats["controller_vs"] == "pure_mpc")
            & (scenario_stats["kpi"] == "rmse"),
            "n_pairs",
        ].iloc[0]
    )

    assert scenario_pairs > run_pairs


def test_scenario_mode_requires_scenario_column():
    run_df = _make_run_level_df(_make_scenario_level_df())
    with pytest.raises(ValueError, match="missing required columns"):
        _compute_scope_stats(
            run_df,
            scope="combined",
            analysis_unit="scenario",
            key_cols=_pair_key_columns("scenario"),
        )


def test_holm_correction_and_wilcoxon_outputs_remain_available():
    run_df = _make_run_level_df(_make_scenario_level_df())
    stats = _compute_scope_stats(
        run_df,
        scope="combined",
        analysis_unit="run",
        key_cols=_pair_key_columns("run"),
    )

    assert not stats.empty
    assert (stats["p_value_corrected"] >= stats["p_value"]).all()
    assert stats["wilcoxon_statistic"].notna().all()
    assert (stats["analysis_unit"] == "run").all()
    assert (stats["pair_key_columns"] == "room_id,seed,variant").all()
