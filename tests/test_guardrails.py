"""Guard-rail tests for critical correctness risks."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from app.config import load_config
from benchmark_report import _aggregate_summary
from app.lstm import FEATURE_COLUMNS, LSTMTrainResult
from run_mvp import (
    _build_eval_scenarios,
    _build_lstm_compatibility_signature,
    _ensure_lstm_residual,
    _read_config_hash,
    _resolve_experiment_settings,
    run_simple,
)
from app.scenario_utils import EXPECTED_SCENARIO_TYPES
from app.controllers import FEATURE_COUNT, LinearRoomModel, MPCController, MPCWeights
from app.simulation import ScenarioSpec, ToyRoomPlant, generate_excitation_data, make_scenario


def _make_detailed_df(
    n_rooms: int = 3,
    n_seeds: int = 2,
    n_variants: int = 2,
) -> pd.DataFrame:
    controllers = ["onoff", "pid", "pure_mpc", "hybrid_mpc"]
    rows = []
    for r in range(n_rooms):
        for s in range(n_seeds):
            for v in range(n_variants):
                for i, ctrl in enumerate(controllers):
                    rows.append(
                        {
                            "room_id": f"room_{r}",
                            "seed": 40 + s,
                            "variant": v,
                            "controller": ctrl,
                            "overall_vs_best_pct": float(i),
                            "comfort_violation_hours": float(i),
                            "rmse": float(i),
                            "mae": float(i),
                            "energy_proxy": float(i),
                            "degree_minutes_outside_band": float(i),
                            "underheating_degree_minutes": float(i),
                            "overheating_degree_minutes": float(i),
                            "solver_optimal_steps": 0,
                            "solver_optimal_inaccurate_steps": 0,
                            "solver_fallbacks": 0,
                        }
                    )
    return pd.DataFrame(rows)


def _make_valid_scenario_csv(path: Path, sampling_minutes: int) -> None:
    timestamps = pd.date_range("2025-01-01", periods=12, freq=f"{sampling_minutes}min")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "T_out": np.full(len(timestamps), 5.0),
            "solar": np.full(len(timestamps), 200.0),
            "occupancy": np.full(len(timestamps), 2.0),
            "occupancy_actual": np.full(len(timestamps), 2.0),
            "unexpected_occupancy": np.zeros(len(timestamps)),
            "setpoint": np.full(len(timestamps), 22.0),
        }
    )
    df.to_csv(path, index=False)


def _dummy_lstm_result(train_out_dir: Path) -> LSTMTrainResult:
    return LSTMTrainResult(
        model_path=str(train_out_dir / "residual_lstm.pt"),
        meta_path=str(train_out_dir / "residual_lstm_meta.json"),
        samples=64,
        train_samples=51,
        val_samples=13,
        train_rmse=0.1,
        val_rmse=0.2,
    )


def _fast_run_monkeypatch(monkeypatch):
    import run_mvp

    def fake_ensure(*, train_out_dir: Path, **kwargs):
        return (
            _dummy_lstm_result(train_out_dir),
            0.25,
            0.25,
            False,
            {
                "reuse_requested": False,
                "reuse_check_passed": False,
                "reuse_mismatch_fields": [],
                "train_signature_hash": "dummy-signature",
                "expected_signature_hash": "dummy-signature",
            },
        )

    def fake_run_closed_loop(plant, controller, scenario_df, warmup_steps):
        n_rows = max(warmup_steps + 2, 4)
        return pd.DataFrame(
            {
                "y": np.full(n_rows, 22.0),
                "setpoint": np.full(n_rows, 22.0),
                "u": np.full(n_rows, 0.3),
                "step_runtime_ms": np.linspace(1.0, 2.0, n_rows),
            }
        )

    def fake_plot(avg_df, out_path, show):
        out_path.write_bytes(b"")

    monkeypatch.setattr(run_mvp, "_ensure_lstm_residual", fake_ensure)
    monkeypatch.setattr(
        run_mvp,
        "_build_controllers",
        lambda cfg, model_dir: {
            name: SimpleNamespace(_solver_status_counts={})
            for name in ["onoff", "pid", "pure_mpc", "hybrid_mpc"]
        },
    )
    monkeypatch.setattr(run_mvp, "run_closed_loop", fake_run_closed_loop)
    monkeypatch.setattr(run_mvp, "_plot", fake_plot)


def test_combined_wins_sum_equals_n_runs():
    n_rooms, n_seeds, n_variants = 3, 2, 2
    df = _make_detailed_df(n_rooms=n_rooms, n_seeds=n_seeds, n_variants=n_variants)
    summary = _aggregate_summary(df)
    total_wins = int(summary["wins"].sum())
    expected = n_rooms * n_seeds * n_variants
    assert total_wins == expected, (
        f"Combined wins sum is {total_wins}, expected {expected}. "
        "Check that room_id is included in the wins groupby."
    )


def test_per_room_wins_sum_equals_n_seeds_times_variants():
    n_seeds, n_variants = 5, 4
    controllers = ["onoff", "pid", "pure_mpc", "hybrid_mpc"]
    rows = []
    for s in range(n_seeds):
        for v in range(n_variants):
            for i, ctrl in enumerate(controllers):
                rows.append(
                    {
                        "seed": 40 + s,
                        "variant": v,
                        "controller": ctrl,
                        "overall_vs_best_pct": float(i),
                        "comfort_violation_hours": float(i),
                        "rmse": float(i),
                        "mae": float(i),
                        "energy_proxy": float(i),
                        "degree_minutes_outside_band": float(i),
                        "underheating_degree_minutes": float(i),
                        "overheating_degree_minutes": float(i),
                        "solver_optimal_steps": 0,
                        "solver_optimal_inaccurate_steps": 0,
                        "solver_fallbacks": 0,
                    }
                )
    df = pd.DataFrame(rows)
    summary = _aggregate_summary(df)
    total_wins = int(summary["wins"].sum())
    expected = n_seeds * n_variants
    assert total_wins == expected


def test_excitation_data_no_negative_u():
    scenario = make_scenario(
        ScenarioSpec(name="cold_day", steps=200, sampling_minutes=10),
        day_setpoint=22.0,
        night_setpoint=20.0,
        seed=0,
    )
    plant = ToyRoomPlant(dt_minutes=10, seed=1)
    exc_df = generate_excitation_data(plant=plant, scenario_df=scenario, seed=2)
    assert (exc_df["u"] >= 0.0).all(), (
        f"Excitation data contains negative u values: min={exc_df['u'].min():.4f}. "
        "Heating-only scope requires u in [0, 1]."
    )


def test_excitation_data_u_bounded_to_one():
    scenario = make_scenario(
        ScenarioSpec(name="sunny_day", steps=200, sampling_minutes=10),
        day_setpoint=22.0,
        night_setpoint=20.0,
        seed=5,
    )
    plant = ToyRoomPlant(dt_minutes=10, seed=6)
    exc_df = generate_excitation_data(plant=plant, scenario_df=scenario, seed=7)
    assert (exc_df["u"] <= 1.0).all()
    assert (exc_df["u"] >= 0.0).all()


def test_feature_columns_count_matches_feature_count():
    assert len(FEATURE_COLUMNS) == FEATURE_COUNT, (
        f"FEATURE_COLUMNS has {len(FEATURE_COLUMNS)} entries but "
        f"FEATURE_COUNT is {FEATURE_COUNT}. Update one to match the other."
    )


def test_feature_vector_length_matches_feature_count():
    from app.controllers import Disturbance, Observation

    model = LinearRoomModel(dt_minutes=10)
    ctrl = MPCController(
        model=model,
        horizon_steps=6,
        u_min=0.0,
        u_max=1.0,
        du_max=1.0,
        weights=MPCWeights(),
    )
    obs = Observation(
        y=21.0,
        setpoint=22.0,
        disturbance=Disturbance(t_out=5.0, solar=200.0, occupancy=2.0),
    )
    feat = ctrl._feature(obs)
    assert len(feat) == FEATURE_COUNT


def test_pure_mpc_residual_sequence_is_zero():
    model = LinearRoomModel(dt_minutes=10)
    ctrl = MPCController(
        model=model,
        horizon_steps=6,
        u_min=0.0,
        u_max=1.0,
        du_max=1.0,
        weights=MPCWeights(),
        residual_predictor=None,
    )
    from app.controllers import Disturbance, Observation

    obs = Observation(
        y=21.0,
        setpoint=22.0,
        disturbance=Disturbance(t_out=5.0, solar=200.0, occupancy=2.0),
    )
    seq = ctrl._residual_sequence(obs)
    assert np.allclose(seq, 0.0)


def test_solver_fallback_count_increments():
    import unittest.mock as mock

    model = LinearRoomModel(dt_minutes=10)
    ctrl = MPCController(
        model=model,
        horizon_steps=6,
        u_min=0.0,
        u_max=1.0,
        du_max=1.0,
        weights=MPCWeights(),
    )
    ctrl._u_prev = 0.5

    from app.controllers import Disturbance, Observation

    obs = Observation(
        y=21.0,
        setpoint=22.0,
        disturbance=Disturbance(t_out=5.0, solar=200.0, occupancy=2.0),
    )

    with mock.patch.object(ctrl.problem, "solve", side_effect=Exception("solver_error")):
        u = ctrl.act(obs)

    assert ctrl._solver_fallback_count == 1
    assert ctrl._solver_status_counts["fallback"] == 1
    assert abs(u - 0.5) < 1e-9


def test_solver_optimal_inaccurate_count_increments_when_no_optimal_found():
    import unittest.mock as mock

    model = LinearRoomModel(dt_minutes=10)
    ctrl = MPCController(
        model=model,
        horizon_steps=6,
        u_min=0.0,
        u_max=1.0,
        du_max=1.0,
        weights=MPCWeights(),
    )
    from app.controllers import Disturbance, Observation

    obs = Observation(
        y=21.0,
        setpoint=22.0,
        disturbance=Disturbance(t_out=5.0, solar=200.0, occupancy=2.0),
    )

    def fake_solve(*args, solver=None, **kwargs):
        ctrl.problem._status = cp.OPTIMAL_INACCURATE
        ctrl.u.value = np.full(ctrl.horizon_steps, 0.3)

    with mock.patch.object(ctrl.problem, "solve", side_effect=fake_solve):
        u = ctrl.act(obs)

    assert abs(u - 0.3) < 1e-9
    assert ctrl._solver_status_counts["optimal"] == 0
    assert ctrl._solver_status_counts["optimal_inaccurate"] == 1
    assert ctrl._solver_status_counts["fallback"] == 0


def test_solver_optimal_is_preferred_over_inaccurate():
    import unittest.mock as mock

    model = LinearRoomModel(dt_minutes=10)
    ctrl = MPCController(
        model=model,
        horizon_steps=6,
        u_min=0.0,
        u_max=1.0,
        du_max=1.0,
        weights=MPCWeights(),
    )
    from app.controllers import Disturbance, Observation

    obs = Observation(
        y=21.0,
        setpoint=22.0,
        disturbance=Disturbance(t_out=5.0, solar=200.0, occupancy=2.0),
    )

    def fake_solve(*args, solver=None, **kwargs):
        if solver == cp.OSQP:
            ctrl.problem._status = cp.OPTIMAL_INACCURATE
            ctrl.u.value = np.full(ctrl.horizon_steps, 0.4)
            return
        ctrl.problem._status = cp.OPTIMAL
        ctrl.u.value = np.full(ctrl.horizon_steps, 0.2)

    with mock.patch.object(ctrl.problem, "solve", side_effect=fake_solve):
        u = ctrl.act(obs)

    assert abs(u - 0.2) < 1e-9
    assert ctrl._solver_status_counts["optimal"] == 1
    assert ctrl._solver_status_counts["optimal_inaccurate"] == 0


def test_single_run_uses_canonical_artifact_layout(tmp_path: Path, monkeypatch):
    _fast_run_monkeypatch(monkeypatch)

    result = run_simple(
        config_path="configs/small_office.yaml",
        seed=42,
        show_plot=False,
        artifact_root=str(tmp_path),
        experiment_tag="smoke",
    )

    run_root = tmp_path / "smoke" / "small_office" / "runs" / "seed_042_variant_00"
    eval_dir = run_root / "eval_run"
    train_dir = run_root / "lstm_train"

    assert eval_dir.is_dir()
    assert train_dir.is_dir()
    assert (eval_dir / "provenance.json").exists()
    assert (train_dir / "provenance.json").exists()
    assert (run_root / "run_manifest.json").exists()
    assert result.run_summary["eval_out_dir"] == str(eval_dir)
    assert result.run_summary["train_out_dir"] == str(train_dir)
    assert result.run_summary["run_root"] == str(run_root)


def test_provenance_is_stage_specific_when_eval_and_train_share_dir(
    tmp_path: Path,
    monkeypatch,
):
    _fast_run_monkeypatch(monkeypatch)
    shared_dir = tmp_path / "shared_run"

    run_simple(
        config_path="configs/small_office.yaml",
        seed=42,
        show_plot=False,
        out_dir=shared_dir,
        train_out_dir=shared_dir,
    )

    eval_prov = shared_dir / "provenance_eval_run.json"
    train_prov = shared_dir / "provenance_lstm_train.json"
    assert eval_prov.exists()
    assert train_prov.exists()
    assert not (shared_dir / "provenance.json").exists()

    eval_payload = json.loads(eval_prov.read_text(encoding="utf-8"))
    train_payload = json.loads(train_prov.read_text(encoding="utf-8"))
    assert eval_payload["stage"] == "eval_run"
    assert train_payload["stage"] == "lstm_train"


def test_warmup_minutes_must_be_divisible_by_sampling_minutes():
    cfg = load_config("configs/small_office.yaml")
    with pytest.raises(ValueError, match="divisible by sampling_minutes"):
        _resolve_experiment_settings(cfg, warmup_minutes=125)


def test_snapshot_contains_effective_experiment_settings(
    tmp_path: Path,
    monkeypatch,
):
    _fast_run_monkeypatch(monkeypatch)

    run_simple(
        config_path="configs/small_office.yaml",
        seed=7,
        show_plot=False,
        artifact_root=str(tmp_path),
        experiment_tag="snapshot",
        warmup_minutes=60,
        lstm_seq_len=8,
        lstm_hidden_size=16,
        lstm_epochs=5,
    )

    snapshot_path = (
        tmp_path
        / "snapshot"
        / "small_office"
        / "runs"
        / "seed_007_variant_00"
        / "eval_run"
        / "experiment_config_snapshot.json"
    )
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["warmup_minutes"] == 60
    assert snapshot["warmup_steps"] == 6
    assert snapshot["lstm_seq_len"] == 8
    assert snapshot["lstm_hidden_size"] == 16
    assert snapshot["lstm_epochs"] == 5
    assert "config_hash" in snapshot
    assert "train_signature_hash" in snapshot
    assert "expected_signature_hash" in snapshot


def test_incompatible_cached_lstm_is_not_reused(tmp_path: Path, monkeypatch):
    cfg = load_config("configs/small_office.yaml")
    settings = _resolve_experiment_settings(cfg)
    config_hash = _read_config_hash("configs/small_office.yaml")
    expected_signature = _build_lstm_compatibility_signature(
        cfg=cfg,
        settings=settings,
        seed=42,
        config_hash=config_hash,
    )

    train_dir = tmp_path / "train_cache"
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "residual_lstm.pt").write_bytes(b"dummy")
    (train_dir / "residual_lstm_summary.json").write_text(
        json.dumps(
            {
                "samples": 64,
                "train_samples": 51,
                "val_samples": 13,
                "train_rmse": 0.1,
                "val_rmse": 0.2,
                "train_runtime_s": 3.0,
            }
        ),
        encoding="utf-8",
    )

    cached_signature = dict(expected_signature)
    cached_signature["hidden_size"] = 999
    cached_signature["lstm_signature_hash"] = "stale-signature"
    (train_dir / "residual_lstm_meta.json").write_text(
        json.dumps(
            {
                "seq_len": cached_signature["seq_len"],
                "hidden_size": cached_signature["hidden_size"],
                "epochs": cached_signature["epochs"],
                "horizon_steps": cached_signature["horizon_steps"],
                "feature_columns": cached_signature["feature_columns"],
                "room_id": cached_signature["room_id"],
                "sampling_minutes": cached_signature["sampling_minutes"],
                "seed": cached_signature["seed"],
                "config_hash": cached_signature["config_hash"],
                "lstm_signature_hash": cached_signature["lstm_signature_hash"],
                "training_signature": {
                    key: cached_signature[key]
                    for key in [
                        "room_id",
                        "sampling_minutes",
                        "horizon_steps",
                        "feature_columns",
                        "seq_len",
                        "hidden_size",
                        "epochs",
                        "seed",
                        "config_hash",
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_train(*, out_dir: Path, **kwargs):
        return _dummy_lstm_result(out_dir), 4.5

    import run_mvp

    monkeypatch.setattr(run_mvp, "_train_lstm_residual", fake_train)

    (
        _train_result,
        train_runtime_incremental,
        train_runtime_source,
        lstm_reused,
        reuse_info,
    ) = _ensure_lstm_residual(
        cfg=cfg,
        train_out_dir=train_dir,
        seed=42,
        settings=settings,
        lstm_signature=expected_signature,
        reuse_trained_model=True,
    )

    assert lstm_reused is False
    assert train_runtime_incremental == 4.5
    assert train_runtime_source == 4.5
    assert reuse_info["reuse_check_passed"] is False
    assert "hidden_size" in reuse_info["reuse_mismatch_fields"]
    assert reuse_info["train_signature_hash"] == expected_signature["lstm_signature_hash"]
    assert reuse_info["expected_signature_hash"] == expected_signature["lstm_signature_hash"]


def test_build_eval_scenarios_requires_complete_set(tmp_path: Path):
    cfg = load_config("configs/small_office.yaml")
    room_dir = tmp_path / cfg.room_id
    room_dir.mkdir(parents=True, exist_ok=True)

    for name in EXPECTED_SCENARIO_TYPES[:-1]:
        _make_valid_scenario_csv(
            room_dir / f"{name}_v00.csv",
            sampling_minutes=cfg.sampling_minutes,
        )

    with pytest.raises(FileNotFoundError, match="Incomplete scenario set"):
        _build_eval_scenarios(
            cfg=cfg,
            seed=42,
            scenarios_dir=str(tmp_path),
            scenario_variant=0,
        )
