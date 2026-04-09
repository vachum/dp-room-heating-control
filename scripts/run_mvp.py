import argparse
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(tempfile.gettempdir()) / "matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
if "XDG_CACHE_HOME" not in os.environ:
    cache_dir = Path(tempfile.gettempdir()) / "xdg-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)

import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from app.config import load_config
from app.experiment_utils import (
    build_provenance_payload,
    resolve_artifact_root,
    sha256_text,
    stable_hash,
    write_json,
)
from app.lstm import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    LSTMTrainResult,
    load_residual_predictor,
    train_residual_lstm,
)
from app.scenario_utils import (
    DEFAULT_SCENARIO_MODE,
    EXPECTED_SCENARIO_TYPES,
    load_scenario_csv,
    validate_scenario_df,
)
from app.controllers import (
    HybridMPCController,
    LinearRoomModel,
    MPCController,
    MPCWeights,
    OnOffController,
    PIDController,
)
from app.metrics import compute_kpi
from app.simulation import (
    ScenarioSpec,
    ToyRoomPlant,
    generate_excitation_data,
    make_scenario,
    run_closed_loop,
)

SCENARIO_SEED_OFFSETS = {
    "cold_day": 101,
    "sunny_day": 102,
    "spring_day": 103,
    "autumn_day": 104,
    "mixed_day": 105,
    "summer_heatwave": 106,
}


@dataclass
class RunResult:
    avg_df: pd.DataFrame
    scenario_kpi_df: pd.DataFrame
    runtime_df: pd.DataFrame
    chart_path: Path
    lstm_result: LSTMTrainResult
    run_summary: dict


@dataclass(frozen=True)
class RunDirectories:
    room_root: Path | None
    run_root: Path
    eval_out_dir: Path
    train_out_dir: Path


@dataclass(frozen=True)
class ExperimentSettings:
    warmup_minutes: int
    warmup_steps: int
    sampling_minutes: int
    horizon_minutes: int
    horizon_steps: int
    lstm_seq_len: int
    lstm_hidden_size: int
    lstm_epochs: int


@dataclass(frozen=True)
class LSTMCacheStatus:
    train_result: LSTMTrainResult | None
    train_runtime_s: float
    compatible: bool
    mismatch_fields: list[str]
    cached_model_found: bool
    train_signature_hash: str | None
    expected_signature_hash: str


LSTM_SIGNATURE_FIELDS = [
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


def _read_config_hash(config_path: str) -> str:
    return sha256_text(Path(config_path).read_text(encoding="utf-8"))


def _resolve_run_dirs(
    cfg,
    seed: int,
    scenario_variant: int,
    artifact_root: str,
    experiment_tag: str | None,
    out_dir: Path | None,
    train_out_dir: Path | None,
) -> RunDirectories:
    room_root = resolve_artifact_root(artifact_root, experiment_tag) / cfg.room_id

    if out_dir is None and train_out_dir is None:
        run_root = room_root / "runs" / f"seed_{seed:03d}_variant_{scenario_variant:02d}"
        return RunDirectories(
            room_root=room_root,
            run_root=run_root,
            eval_out_dir=run_root / "eval_run",
            train_out_dir=run_root / "lstm_train",
        )

    eval_out_dir = Path(out_dir) if out_dir is not None else None
    resolved_train_dir = Path(train_out_dir) if train_out_dir is not None else None

    if eval_out_dir is None:
        run_root = room_root / "runs" / f"seed_{seed:03d}_variant_{scenario_variant:02d}"
        eval_out_dir = run_root / "eval_run"
    else:
        run_root = eval_out_dir

    if resolved_train_dir is None:
        resolved_train_dir = eval_out_dir

    return RunDirectories(
        room_root=room_root if out_dir is None else None,
        run_root=run_root,
        eval_out_dir=eval_out_dir,
        train_out_dir=resolved_train_dir,
    )


def _resolve_experiment_settings(
    cfg,
    *,
    warmup_minutes: int | None = None,
    lstm_seq_len: int | None = None,
    lstm_hidden_size: int | None = None,
    lstm_epochs: int | None = None,
) -> ExperimentSettings:
    effective_warmup_minutes = (
        cfg.warmup_minutes if warmup_minutes is None else int(warmup_minutes)
    )
    effective_seq_len = cfg.lstm_seq_len if lstm_seq_len is None else int(lstm_seq_len)
    effective_hidden_size = (
        cfg.lstm_hidden_size if lstm_hidden_size is None else int(lstm_hidden_size)
    )
    effective_epochs = cfg.lstm_epochs if lstm_epochs is None else int(lstm_epochs)

    if effective_warmup_minutes < 0:
        raise ValueError("warmup_minutes must be >= 0")
    if effective_warmup_minutes % cfg.sampling_minutes != 0:
        raise ValueError("warmup_minutes must be divisible by sampling_minutes")
    if effective_seq_len < 1:
        raise ValueError("lstm_seq_len must be >= 1")
    if effective_hidden_size < 1:
        raise ValueError("lstm_hidden_size must be >= 1")
    if effective_epochs < 1:
        raise ValueError("lstm_epochs must be >= 1")

    return ExperimentSettings(
        warmup_minutes=effective_warmup_minutes,
        warmup_steps=effective_warmup_minutes // cfg.sampling_minutes,
        sampling_minutes=cfg.sampling_minutes,
        horizon_minutes=cfg.horizon_minutes,
        horizon_steps=cfg.horizon_steps,
        lstm_seq_len=effective_seq_len,
        lstm_hidden_size=effective_hidden_size,
        lstm_epochs=effective_epochs,
    )


def _same_directory(path_a: Path, path_b: Path) -> bool:
    return path_a.resolve(strict=False) == path_b.resolve(strict=False)


def _provenance_filename_for_stage(
    eval_out_dir: Path,
    train_out_dir: Path,
    stage: str,
) -> str:
    if _same_directory(eval_out_dir, train_out_dir):
        return f"provenance_{stage}.json"
    return "provenance.json"


def _capture_provenance(
    config_path: str,
    seed: int,
    scenario_variant: int,
    out_dir: Path,
    stage: str,
    filename: str | None = None,
    extra: dict | None = None,
) -> tuple[dict, Path]:
    payload = build_provenance_payload(
        {
            "config_path": config_path,
            "seed": seed,
            "scenario_variant": scenario_variant,
            "out_dir": str(out_dir),
            "stage": stage,
            **(extra or {}),
        }
    )
    provenance_path = out_dir / (filename or "provenance.json")
    write_json(provenance_path, payload)

    config_snapshot = Path(config_path)
    if config_snapshot.exists():
        (out_dir / "config_snapshot.yaml").write_text(
            config_snapshot.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    return payload, provenance_path


def _build_lstm_compatibility_signature(
    cfg,
    settings: ExperimentSettings,
    seed: int,
    config_hash: str,
) -> dict:
    signature = {
        "room_id": cfg.room_id,
        "sampling_minutes": cfg.sampling_minutes,
        "horizon_steps": cfg.horizon_steps,
        "feature_columns": FEATURE_COLUMNS,
        "seq_len": settings.lstm_seq_len,
        "hidden_size": settings.lstm_hidden_size,
        "epochs": settings.lstm_epochs,
        "seed": seed,
        "config_hash": config_hash,
    }
    signature["lstm_signature_hash"] = stable_hash(signature)
    return signature


def _extract_lstm_signature(meta: dict) -> dict:
    if isinstance(meta.get("training_signature"), dict):
        signature = dict(meta["training_signature"])
    else:
        signature = {
            "room_id": meta.get("room_id"),
            "sampling_minutes": meta.get("sampling_minutes"),
            "horizon_steps": meta.get("horizon_steps", meta.get("horizon")),
            "feature_columns": meta.get("feature_columns"),
            "seq_len": meta.get("seq_len"),
            "hidden_size": meta.get("hidden_size"),
            "epochs": meta.get("epochs"),
            "seed": meta.get("seed"),
            "config_hash": meta.get("config_hash"),
        }
    signature["lstm_signature_hash"] = meta.get("lstm_signature_hash") or stable_hash(
        {field: signature.get(field) for field in LSTM_SIGNATURE_FIELDS}
    )
    return signature


def _compare_lstm_signatures(expected: dict, actual: dict) -> list[str]:
    mismatch_fields: list[str] = []
    for field in LSTM_SIGNATURE_FIELDS:
        if actual.get(field) != expected.get(field):
            mismatch_fields.append(field)
    return mismatch_fields


def _write_run_manifest(
    run_dirs: RunDirectories,
    *,
    room_id: str,
    seed: int,
    scenario_variant: int,
    eval_provenance_path: Path,
    train_provenance_path: Path,
) -> Path:
    manifest_path = run_dirs.run_root / "run_manifest.json"
    write_json(
        manifest_path,
        {
            "room_id": room_id,
            "seed": seed,
            "scenario_variant": scenario_variant,
            "room_root": str(run_dirs.room_root) if run_dirs.room_root is not None else None,
            "run_root": str(run_dirs.run_root),
            "eval_out_dir": str(run_dirs.eval_out_dir),
            "train_out_dir": str(run_dirs.train_out_dir),
            "eval_provenance_path": str(eval_provenance_path),
            "train_provenance_path": str(train_provenance_path),
        },
    )
    return manifest_path


def _resolve_train_provenance_reference(
    eval_out_dir: Path,
    train_out_dir: Path,
) -> Path:
    preferred = train_out_dir / _provenance_filename_for_stage(
        eval_out_dir,
        train_out_dir,
        "lstm_train",
    )
    if preferred.exists():
        return preferred

    fallback = train_out_dir / "provenance.json"
    if fallback.exists():
        return fallback
    return preferred


def _build_controllers(cfg, model_dir: Path):
    model = LinearRoomModel(
        dt_minutes=cfg.sampling_minutes,
        heating_gain=cfg.mpc_heating_gain,
        leak_coef=cfg.mpc_leak_coef,
        solar_coef=cfg.mpc_solar_coef,
        occ_coef=cfg.mpc_occ_coef,
    )

    pure_weights = MPCWeights(q_track=20.0, r_energy=1.0, r_du=0.15, deadband=0.3)
    hybrid_weights = MPCWeights(q_track=20.0, r_energy=1.0, r_du=0.15, deadband=0.3)

    pure_mpc = MPCController(
        model=model,
        horizon_steps=cfg.horizon_steps,
        u_min=cfg.u_min,
        u_max=cfg.u_max,
        du_max=cfg.du_max,
        weights=pure_weights,
    )

    hybrid_predictor = load_residual_predictor(str(model_dir))
    hybrid_mpc = HybridMPCController(
        model=LinearRoomModel(
            dt_minutes=cfg.sampling_minutes,
            heating_gain=cfg.mpc_heating_gain,
            leak_coef=cfg.mpc_leak_coef,
            solar_coef=cfg.mpc_solar_coef,
            occ_coef=cfg.mpc_occ_coef,
        ),
        horizon_steps=cfg.horizon_steps,
        u_min=cfg.u_min,
        u_max=cfg.u_max,
        du_max=cfg.du_max,
        weights=hybrid_weights,
        residual_predictor=hybrid_predictor,
    )

    return {
        "onoff": OnOffController(deadband=0.4, u_min=cfg.u_min, u_max=cfg.u_max),
        "pid": PIDController(
            kp=0.30,
            ki=0.02,
            kd=0.03,
            u_min=cfg.u_min,
            u_max=cfg.u_max,
        ),
        "pure_mpc": pure_mpc,
        "hybrid_mpc": hybrid_mpc,
    }


def _residual_for_row(model: LinearRoomModel, row: pd.Series) -> float:
    y_hat = model.a * float(row["y"])
    y_hat += model.b_u * float(row["u"])
    y_hat += model.b_t_out * float(row["T_out"])
    y_hat += model.b_solar * float(row["solar"])
    y_hat += model.b_occ * float(row["occupancy"])
    return float(row["y_next"] - y_hat)


def _train_lstm_residual(
    cfg,
    out_dir: Path,
    seed: int,
    settings: ExperimentSettings,
    lstm_signature: dict,
) -> tuple[LSTMTrainResult, float]:
    t0 = time.perf_counter()
    steps_per_day = int((24 * 60) / cfg.sampling_minutes)
    train_segments = [
        ("cold_day", 10),
        ("sunny_day", 10),
        ("spring_day", 10),
        ("autumn_day", 10),
        ("mixed_day", 10),
        ("summer_heatwave", 10),
    ]
    scenario_parts: list[pd.DataFrame] = []
    for idx, (name, days) in enumerate(train_segments):
        part = make_scenario(
            ScenarioSpec(
                name=name,
                steps=days * steps_per_day,
                sampling_minutes=cfg.sampling_minutes,
            ),
            day_setpoint=cfg.day_setpoint,
            night_setpoint=cfg.night_setpoint,
            seed=seed + 5 + idx,
        )
        scenario_parts.append(part)

    train_scenario = pd.concat(scenario_parts, ignore_index=True)
    train_scenario["timestamp"] = pd.date_range(
        start="2025-01-01T00:00:00Z",
        periods=len(train_scenario),
        freq=f"{cfg.sampling_minutes}min",
    )

    plant = ToyRoomPlant(
        dt_minutes=cfg.sampling_minutes,
        seed=seed + 6,
        heating_gain=cfg.plant_heating_gain,
        leak_coef=cfg.plant_leak_coef,
        solar_coef=cfg.plant_solar_coef,
        occ_coef=cfg.plant_occ_coef,
    )
    exc_df = generate_excitation_data(
        plant=plant,
        scenario_df=train_scenario,
        seed=seed + 7,
    )

    model = LinearRoomModel(
        dt_minutes=cfg.sampling_minutes,
        heating_gain=cfg.mpc_heating_gain,
        leak_coef=cfg.mpc_leak_coef,
        solar_coef=cfg.mpc_solar_coef,
        occ_coef=cfg.mpc_occ_coef,
    )
    exc_df[TARGET_COLUMN] = exc_df.apply(
        lambda row: _residual_for_row(model, row),
        axis=1,
    )
    exc_df["u_prev"] = exc_df["u"].shift(1, fill_value=0.0)
    exc_df["residual_prev"] = exc_df[TARGET_COLUMN].shift(1, fill_value=0.0)
    exc_df["u"] = exc_df["u_prev"]
    exc_df["residual"] = exc_df["residual_prev"]

    exc_path = out_dir / "residual_training_data.csv"
    exc_df.to_csv(exc_path, index=False)

    train_result = train_residual_lstm(
        df=exc_df,
        out_dir=str(out_dir),
        seq_len=settings.lstm_seq_len,
        horizon=cfg.horizon_steps,
        hidden_size=settings.lstm_hidden_size,
        epochs=settings.lstm_epochs,
        seed=seed,
        metadata={
            "room_id": cfg.room_id,
            "sampling_minutes": cfg.sampling_minutes,
            "horizon_steps": cfg.horizon_steps,
            "epochs": settings.lstm_epochs,
            "seed": seed,
            "config_hash": lstm_signature["config_hash"],
            "lstm_signature_hash": lstm_signature["lstm_signature_hash"],
            "training_signature": {
                field: lstm_signature[field] for field in LSTM_SIGNATURE_FIELDS
            },
        },
    )

    train_runtime_s = time.perf_counter() - t0
    write_json(
        out_dir / "residual_lstm_summary.json",
        {
            "split_strategy": "time_ordered",
            "samples": train_result.samples,
            "train_samples": train_result.train_samples,
            "val_samples": train_result.val_samples,
            "train_rmse": train_result.train_rmse,
            "val_rmse": train_result.val_rmse,
            "training_data": str(exc_path),
            "model_path": train_result.model_path,
            "meta_path": train_result.meta_path,
            "train_runtime_s": train_runtime_s,
            "train_signature_hash": lstm_signature["lstm_signature_hash"],
        },
    )

    return train_result, train_runtime_s


def _load_existing_lstm_result(
    train_out_dir: Path,
    expected_signature: dict,
) -> LSTMCacheStatus:
    summary_path = train_out_dir / "residual_lstm_summary.json"
    model_path = train_out_dir / "residual_lstm.pt"
    meta_path = train_out_dir / "residual_lstm_meta.json"
    if not (summary_path.exists() and model_path.exists() and meta_path.exists()):
        return LSTMCacheStatus(
            train_result=None,
            train_runtime_s=0.0,
            compatible=False,
            mismatch_fields=["missing_cached_model"],
            cached_model_found=False,
            train_signature_hash=None,
            expected_signature_hash=expected_signature["lstm_signature_hash"],
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    actual_signature = _extract_lstm_signature(meta)
    mismatch_fields = _compare_lstm_signatures(expected_signature, actual_signature)
    return LSTMCacheStatus(
        train_result=LSTMTrainResult(
            model_path=str(model_path),
            meta_path=str(meta_path),
            samples=int(summary["samples"]),
            train_samples=int(summary["train_samples"]),
            val_samples=int(summary["val_samples"]),
            train_rmse=float(summary["train_rmse"]),
            val_rmse=float(summary["val_rmse"]),
        ),
        train_runtime_s=float(summary.get("train_runtime_s", 0.0)),
        compatible=not mismatch_fields,
        mismatch_fields=mismatch_fields,
        cached_model_found=True,
        train_signature_hash=actual_signature["lstm_signature_hash"],
        expected_signature_hash=expected_signature["lstm_signature_hash"],
    )


def _ensure_lstm_residual(
    cfg,
    train_out_dir: Path,
    seed: int,
    settings: ExperimentSettings,
    lstm_signature: dict,
    reuse_trained_model: bool,
) -> tuple[LSTMTrainResult, float, float, bool, dict]:
    cache_status = LSTMCacheStatus(
        train_result=None,
        train_runtime_s=0.0,
        compatible=False,
        mismatch_fields=[],
        cached_model_found=False,
        train_signature_hash=None,
        expected_signature_hash=lstm_signature["lstm_signature_hash"],
    )
    if reuse_trained_model:
        cache_status = _load_existing_lstm_result(
            train_out_dir,
            expected_signature=lstm_signature,
        )
        if cache_status.compatible and cache_status.train_result is not None:
            return (
                cache_status.train_result,
                0.0,
                cache_status.train_runtime_s,
                True,
                {
                    "reuse_requested": True,
                    "reuse_check_passed": True,
                    "reuse_mismatch_fields": [],
                    "train_signature_hash": cache_status.train_signature_hash,
                    "expected_signature_hash": cache_status.expected_signature_hash,
                },
            )

    train_result, train_runtime_s = _train_lstm_residual(
        cfg=cfg,
        out_dir=train_out_dir,
        seed=seed,
        settings=settings,
        lstm_signature=lstm_signature,
    )
    return (
        train_result,
        train_runtime_s,
        train_runtime_s,
        False,
        {
            "reuse_requested": reuse_trained_model,
            "reuse_check_passed": False,
            "reuse_mismatch_fields": cache_status.mismatch_fields
            if reuse_trained_model
            else [],
            "train_signature_hash": lstm_signature["lstm_signature_hash"],
            "expected_signature_hash": lstm_signature["lstm_signature_hash"],
        },
    )


def _plot(avg_df: pd.DataFrame, out_path: Path, show: bool) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    cols = [
        "comfort_violation_hours",
        "degree_minutes_outside_band",
        "rmse",
        "energy_proxy",
    ]
    titles = ["Comfort [h]", "Discomfort [deg-min]", "RMSE [degC]", "Energy proxy"]

    for ax, col, title in zip(axes, cols, titles):
        ax.bar(avg_df["controller"], avg_df[col])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)

    if show:
        plt.show()
    plt.close(fig)


def _build_eval_scenarios(
    cfg,
    seed: int,
    scenarios_dir: str | None,
    scenario_variant: int,
    scenario_mode: str = DEFAULT_SCENARIO_MODE,
) -> tuple[dict[str, pd.DataFrame], dict]:
    if scenarios_dir is None:
        scenarios: dict[str, pd.DataFrame] = {}
        scenario_sources: dict[str, str] = {}
        for s_type in EXPECTED_SCENARIO_TYPES:
            scenario_df = make_scenario(
                ScenarioSpec(
                    name=s_type,
                    steps=192,
                    sampling_minutes=cfg.sampling_minutes,
                ),
                day_setpoint=cfg.day_setpoint,
                night_setpoint=cfg.night_setpoint,
                seed=seed + SCENARIO_SEED_OFFSETS.get(s_type, 100),
            )
            validate_scenario_df(
                scenario_df,
                mode=scenario_mode,
                expected_sampling_minutes=cfg.sampling_minutes,
                source=s_type,
            )
            scenarios[s_type] = scenario_df
            scenario_sources[s_type] = "generated_in_memory"

        loaded = list(scenarios.keys())
        return scenarios, {
            "expected_scenarios": EXPECTED_SCENARIO_TYPES,
            "loaded_scenarios": loaded,
            "n_scenarios_loaded": len(loaded),
            "scenario_sources": scenario_sources,
        }

    base = Path(scenarios_dir) / cfg.room_id
    if not base.exists():
        raise FileNotFoundError(
            f"Scenario directory not found: {base}. "
            "Generate it first with generate_scenarios.py."
        )

    missing = [
        s_type
        for s_type in EXPECTED_SCENARIO_TYPES
        if not (base / f"{s_type}_v{scenario_variant:02d}.csv").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Incomplete scenario set for room={cfg.room_id}, variant=v{scenario_variant:02d}. "
            f"Missing scenarios: {missing}. Expected complete set: {EXPECTED_SCENARIO_TYPES}"
        )

    scenarios = {}
    scenario_sources = {}
    for s_type in EXPECTED_SCENARIO_TYPES:
        csv_path = base / f"{s_type}_v{scenario_variant:02d}.csv"
        scenarios[s_type] = load_scenario_csv(
            csv_path,
            mode=scenario_mode,
            expected_sampling_minutes=cfg.sampling_minutes,
        )
        scenario_sources[s_type] = str(csv_path)

    return scenarios, {
        "expected_scenarios": EXPECTED_SCENARIO_TYPES,
        "loaded_scenarios": list(scenarios.keys()),
        "n_scenarios_loaded": len(scenarios),
        "scenario_sources": scenario_sources,
    }


def run_simple(
    config_path: str,
    seed: int,
    show_plot: bool,
    scenarios_dir: str | None = None,
    scenario_variant: int = 0,
    out_dir: Path | None = None,
    artifact_root: str = "artifacts",
    experiment_tag: str | None = None,
    train_out_dir: Path | None = None,
    scenario_mode: str = DEFAULT_SCENARIO_MODE,
    reuse_trained_model: bool = False,
    warmup_minutes: int | None = None,
    lstm_seq_len: int | None = None,
    lstm_hidden_size: int | None = None,
    lstm_epochs: int | None = None,
) -> RunResult:
    cfg = load_config(config_path)
    settings = _resolve_experiment_settings(
        cfg,
        warmup_minutes=warmup_minutes,
        lstm_seq_len=lstm_seq_len,
        lstm_hidden_size=lstm_hidden_size,
        lstm_epochs=lstm_epochs,
    )
    config_hash = _read_config_hash(config_path)
    run_dirs = _resolve_run_dirs(
        cfg=cfg,
        seed=seed,
        scenario_variant=scenario_variant,
        artifact_root=artifact_root,
        experiment_tag=experiment_tag,
        out_dir=out_dir,
        train_out_dir=train_out_dir,
    )
    run_dirs.eval_out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs.train_out_dir.mkdir(parents=True, exist_ok=True)

    _, eval_provenance_path = _capture_provenance(
        config_path=config_path,
        seed=seed,
        scenario_variant=scenario_variant,
        out_dir=run_dirs.eval_out_dir,
        stage="eval_run",
        filename=_provenance_filename_for_stage(
            run_dirs.eval_out_dir,
            run_dirs.train_out_dir,
            "eval_run",
        ),
        extra={
            "scenario_mode": scenario_mode,
            "train_out_dir": str(run_dirs.train_out_dir),
            "eval_out_dir": str(run_dirs.eval_out_dir),
            "run_root": str(run_dirs.run_root),
        },
    )
    lstm_signature = _build_lstm_compatibility_signature(
        cfg=cfg,
        settings=settings,
        seed=seed,
        config_hash=config_hash,
    )

    (
        lstm_result,
        train_runtime_s_incremental,
        train_runtime_s_source,
        lstm_reused,
        reuse_info,
    ) = _ensure_lstm_residual(
        cfg=cfg,
        train_out_dir=run_dirs.train_out_dir,
        seed=seed,
        settings=settings,
        lstm_signature=lstm_signature,
        reuse_trained_model=reuse_trained_model,
    )

    if not lstm_reused:
        _, train_provenance_path = _capture_provenance(
            config_path=config_path,
            seed=seed,
            scenario_variant=scenario_variant,
            out_dir=run_dirs.train_out_dir,
            stage="lstm_train",
            filename=_provenance_filename_for_stage(
                run_dirs.eval_out_dir,
                run_dirs.train_out_dir,
                "lstm_train",
            ),
            extra={
                "scenario_mode": scenario_mode,
                "eval_out_dir": str(run_dirs.eval_out_dir),
                "train_out_dir": str(run_dirs.train_out_dir),
                "run_root": str(run_dirs.run_root),
            },
        )
    else:
        train_provenance_path = _resolve_train_provenance_reference(
            run_dirs.eval_out_dir,
            run_dirs.train_out_dir,
        )

    manifest_path = _write_run_manifest(
        run_dirs,
        room_id=cfg.room_id,
        seed=seed,
        scenario_variant=scenario_variant,
        eval_provenance_path=eval_provenance_path,
        train_provenance_path=train_provenance_path,
    )

    scenarios, scenario_meta = _build_eval_scenarios(
        cfg=cfg,
        seed=seed,
        scenarios_dir=scenarios_dir,
        scenario_variant=scenario_variant,
        scenario_mode=scenario_mode,
    )

    experiment_config = {
        "room_id": cfg.room_id,
        "seed": seed,
        "scenario_variant": scenario_variant,
        "scenario_mode": scenario_mode,
        "scope": "heating_only",
        "u_min": cfg.u_min,
        "u_max": cfg.u_max,
        "split_strategy": "time_ordered",
        **asdict(settings),
        "controllers": ["onoff", "pid", "pure_mpc", "hybrid_mpc"],
        "expected_scenarios": EXPECTED_SCENARIO_TYPES,
        "loaded_scenarios": scenario_meta["loaded_scenarios"],
        "n_scenarios_loaded": scenario_meta["n_scenarios_loaded"],
        "scenario_sources": scenario_meta["scenario_sources"],
        "train_out_dir": str(run_dirs.train_out_dir),
        "eval_out_dir": str(run_dirs.eval_out_dir),
        "run_root": str(run_dirs.run_root),
        "run_manifest_path": str(manifest_path),
        "eval_provenance_path": str(eval_provenance_path),
        "train_provenance_path": str(train_provenance_path),
        "lstm_reused": lstm_reused,
        "reuse_requested": reuse_info["reuse_requested"],
        "reuse_check_passed": reuse_info["reuse_check_passed"],
        "reuse_mismatch_fields": reuse_info["reuse_mismatch_fields"],
        "train_signature_hash": reuse_info["train_signature_hash"],
        "expected_signature_hash": reuse_info["expected_signature_hash"],
        "config_path": config_path,
        "config_hash": config_hash,
    }
    write_json(
        run_dirs.eval_out_dir / "experiment_config_snapshot.json",
        experiment_config,
    )

    rows: list[dict] = []
    runtime_rows: list[dict] = []

    for scenario_name, scenario_df in scenarios.items():
        controllers = _build_controllers(cfg=cfg, model_dir=run_dirs.train_out_dir)
        for name, controller in controllers.items():
            plant = ToyRoomPlant(
                dt_minutes=cfg.sampling_minutes,
                seed=seed + 100,
                heating_gain=cfg.plant_heating_gain,
                leak_coef=cfg.plant_leak_coef,
                solar_coef=cfg.plant_solar_coef,
                occ_coef=cfg.plant_occ_coef,
            )
            log_df = run_closed_loop(
                plant,
                controller,
                scenario_df,
                warmup_steps=settings.warmup_steps,
            )
            log_df.to_csv(
                run_dirs.eval_out_dir / f"{scenario_name}_{name}_log.csv",
                index=False,
            )

            kpi = compute_kpi(
                log_df,
                sampling_minutes=cfg.sampling_minutes,
                skip_steps=settings.warmup_steps,
            )
            solver_status_counts = getattr(controller, "_solver_status_counts", {})
            rows.append(
                {
                    "scenario": scenario_name,
                    "controller": name,
                    "comfort_violation_hours": kpi.comfort_violation_hours,
                    "rmse": kpi.rmse,
                    "mae": kpi.mae,
                    "energy_proxy": kpi.energy_proxy,
                    "degree_minutes_outside_band": kpi.degree_minutes_outside_band,
                    "underheating_degree_minutes": kpi.underheating_degree_minutes,
                    "overheating_degree_minutes": kpi.overheating_degree_minutes,
                    "solver_optimal_steps": int(solver_status_counts.get("optimal", 0)),
                    "solver_optimal_inaccurate_steps": int(
                        solver_status_counts.get("optimal_inaccurate", 0)
                    ),
                    "solver_fallbacks": int(solver_status_counts.get("fallback", 0)),
                }
            )

            eval_df = (
                log_df.iloc[settings.warmup_steps :]
                if settings.warmup_steps > 0
                else log_df
            )
            runtime_rows.append(
                {
                    "controller": name,
                    "scenario": scenario_name,
                    "mean_step_runtime_ms": float(eval_df["step_runtime_ms"].mean()),
                    "max_step_runtime_ms": float(eval_df["step_runtime_ms"].max()),
                    "p95_step_runtime_ms": float(
                        eval_df["step_runtime_ms"].quantile(0.95)
                    ),
                    "train_runtime_s_incremental": train_runtime_s_incremental
                    if name == "hybrid_mpc"
                    else 0.0,
                    "train_runtime_s_source": train_runtime_s_source
                    if name == "hybrid_mpc"
                    else 0.0,
                    "lstm_reused": lstm_reused if name == "hybrid_mpc" else False,
                }
            )

    kpi_df = pd.DataFrame(rows)
    kpi_df.to_csv(run_dirs.eval_out_dir / "kpi_simple_table.csv", index=False)

    runtime_df = pd.DataFrame(runtime_rows)
    runtime_df.to_csv(run_dirs.eval_out_dir / "runtime_table.csv", index=False)

    avg_df = (
        kpi_df.groupby("controller", as_index=False)
        .agg(
            comfort_violation_hours=("comfort_violation_hours", "mean"),
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            energy_proxy=("energy_proxy", "mean"),
            degree_minutes_outside_band=("degree_minutes_outside_band", "mean"),
            underheating_degree_minutes=("underheating_degree_minutes", "mean"),
            overheating_degree_minutes=("overheating_degree_minutes", "mean"),
            solver_optimal_steps=("solver_optimal_steps", "sum"),
            solver_optimal_inaccurate_steps=(
                "solver_optimal_inaccurate_steps",
                "sum",
            ),
            solver_fallbacks=("solver_fallbacks", "sum"),
            n_scenarios=("scenario", "count"),
        )
        .sort_values("rmse")
        .reset_index(drop=True)
    )

    best_rmse = float(avg_df.iloc[0]["rmse"])
    avg_df["rmse_vs_best_pct"] = (
        (avg_df["rmse"] - best_rmse) / max(best_rmse, 1e-9) * 100.0
    ).round(1)

    metric_cols = ["comfort_violation_hours", "rmse", "mae", "energy_proxy"]
    for col in metric_cols:
        best_col = float(avg_df[col].min())
        avg_df[f"{col}_vs_best_pct"] = (
            (avg_df[col] - best_col) / max(best_col, 1e-9) * 100.0
        )

    avg_df["overall_vs_best_pct"] = (
        avg_df[[f"{c}_vs_best_pct" for c in metric_cols]].mean(axis=1).round(1)
    )
    avg_df = avg_df.sort_values("overall_vs_best_pct").reset_index(drop=True)
    avg_df = avg_df.drop(columns=[f"{c}_vs_best_pct" for c in metric_cols])
    avg_df.to_csv(run_dirs.eval_out_dir / "kpi_simple_avg.csv", index=False)

    chart = run_dirs.eval_out_dir / "comparison_simple.png"
    _plot(avg_df, chart, show_plot)

    run_summary = {
        "room_id": cfg.room_id,
        "seed": seed,
        "scenario_variant": scenario_variant,
        "scenario_mode": scenario_mode,
        "expected_scenarios": EXPECTED_SCENARIO_TYPES,
        "loaded_scenarios": scenario_meta["loaded_scenarios"],
        "n_scenarios_loaded": scenario_meta["n_scenarios_loaded"],
        "scenario_sources": scenario_meta["scenario_sources"],
        "warmup_minutes": settings.warmup_minutes,
        "warmup_steps": settings.warmup_steps,
        "lstm_seq_len": settings.lstm_seq_len,
        "lstm_hidden_size": settings.lstm_hidden_size,
        "lstm_epochs": settings.lstm_epochs,
        "config_path": config_path,
        "config_hash": config_hash,
        "run_root": str(run_dirs.run_root),
        "run_manifest_path": str(manifest_path),
        "eval_provenance_path": str(eval_provenance_path),
        "train_provenance_path": str(train_provenance_path),
        "train_out_dir": str(run_dirs.train_out_dir),
        "eval_out_dir": str(run_dirs.eval_out_dir),
        "lstm_reused": lstm_reused,
        "reuse_requested": reuse_info["reuse_requested"],
        "reuse_check_passed": reuse_info["reuse_check_passed"],
        "reuse_mismatch_fields": reuse_info["reuse_mismatch_fields"],
        "train_signature_hash": reuse_info["train_signature_hash"],
        "expected_signature_hash": reuse_info["expected_signature_hash"],
        "train_runtime_s_incremental": train_runtime_s_incremental,
        "train_runtime_s_source": train_runtime_s_source,
        "lstm_model_path": lstm_result.model_path,
        "solver_status_totals": {
            "optimal": int(kpi_df["solver_optimal_steps"].sum()),
            "optimal_inaccurate": int(
                kpi_df["solver_optimal_inaccurate_steps"].sum()
            ),
            "fallback": int(kpi_df["solver_fallbacks"].sum()),
        },
        "solver_status_by_controller": {
            row["controller"]: {
                "optimal": int(row["solver_optimal_steps"]),
                "optimal_inaccurate": int(row["solver_optimal_inaccurate_steps"]),
                "fallback": int(row["solver_fallbacks"]),
            }
            for row in avg_df.to_dict(orient="records")
        },
    }
    write_json(run_dirs.eval_out_dir / "run_summary.json", run_summary)

    return RunResult(
        avg_df=avg_df,
        scenario_kpi_df=kpi_df,
        runtime_df=runtime_df,
        chart_path=chart,
        lstm_result=lstm_result,
        run_summary=run_summary,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Controller comparison: OnOff, PID, pure MPC, Hybrid MPC+LSTM"
    )
    parser.add_argument("--config", default="configs/small_office.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scenarios-dir",
        default=None,
        help=(
            "Folder with pre-generated scenarios "
            "(expects <dir>/<room_id>/<scenario>_vXX.csv for all expected scenarios)"
        ),
    )
    parser.add_argument(
        "--scenario-variant",
        type=int,
        default=0,
        help="Scenario variant index XX used in *_vXX.csv",
    )
    parser.add_argument(
        "--scenario-mode",
        choices=["standard", "hidden_occupancy"],
        default=DEFAULT_SCENARIO_MODE,
        help="Scenario validation mode. Benchmark defaults to hidden occupancy.",
    )
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--experiment-tag", default=None)
    parser.add_argument("--warmup-minutes", type=int, default=None)
    parser.add_argument("--lstm-seq-len", type=int, default=None)
    parser.add_argument("--lstm-hidden-size", type=int, default=None)
    parser.add_argument("--lstm-epochs", type=int, default=None)
    parser.add_argument("--reuse-trained-model", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    result = run_simple(
        config_path=args.config,
        seed=args.seed,
        show_plot=not args.no_show,
        scenarios_dir=args.scenarios_dir,
        scenario_variant=args.scenario_variant,
        artifact_root=args.artifact_root,
        experiment_tag=args.experiment_tag,
        scenario_mode=args.scenario_mode,
        reuse_trained_model=args.reuse_trained_model,
        warmup_minutes=args.warmup_minutes,
        lstm_seq_len=args.lstm_seq_len,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_epochs=args.lstm_epochs,
    )

    print("\nLSTM training:")
    print(
        f"samples={result.lstm_result.samples}, "
        f"train_rmse={result.lstm_result.train_rmse:.4f}, "
        f"val_rmse={result.lstm_result.val_rmse:.4f}"
    )

    print("\nController comparison (mean across scenarios):")
    print(result.avg_df.to_string(index=False))
    print(f"\nGraf ulozen: {result.chart_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
