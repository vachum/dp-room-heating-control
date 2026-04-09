import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from app.config import load_config
from app.experiment_utils import build_provenance_payload, resolve_artifact_root, write_json
from run_mvp import run_simple
from app.scenario_utils import DEFAULT_SCENARIO_MODE, EXPECTED_SCENARIO_TYPES

DEFAULT_CONFIGS = [
    "configs/small_office.yaml",
    "configs/large_office.yaml",
    "configs/meeting_room.yaml",
]


def _parse_int_list(value: str) -> list[int]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise ValueError("Expected at least one integer value.")
    return [int(x) for x in items]


def _aggregate_summary(detailed_df: pd.DataFrame) -> pd.DataFrame:
    def p25(x):
        return x.quantile(0.25)

    def p75(x):
        return x.quantile(0.75)

    summary_df = (
        detailed_df.groupby("controller", as_index=False)
        .agg(
            comfort_violation_hours_mean=("comfort_violation_hours", "mean"),
            comfort_violation_hours_std=("comfort_violation_hours", "std"),
            comfort_violation_hours_median=("comfort_violation_hours", "median"),
            comfort_violation_hours_min=("comfort_violation_hours", "min"),
            comfort_violation_hours_max=("comfort_violation_hours", "max"),
            comfort_violation_hours_p25=("comfort_violation_hours", p25),
            comfort_violation_hours_p75=("comfort_violation_hours", p75),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            rmse_median=("rmse", "median"),
            rmse_min=("rmse", "min"),
            rmse_max=("rmse", "max"),
            rmse_p25=("rmse", p25),
            rmse_p75=("rmse", p75),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mae_median=("mae", "median"),
            mae_min=("mae", "min"),
            mae_max=("mae", "max"),
            mae_p25=("mae", p25),
            mae_p75=("mae", p75),
            energy_proxy_mean=("energy_proxy", "mean"),
            energy_proxy_std=("energy_proxy", "std"),
            energy_proxy_median=("energy_proxy", "median"),
            energy_proxy_min=("energy_proxy", "min"),
            energy_proxy_max=("energy_proxy", "max"),
            energy_proxy_p25=("energy_proxy", p25),
            energy_proxy_p75=("energy_proxy", p75),
            degree_minutes_outside_band_mean=("degree_minutes_outside_band", "mean"),
            degree_minutes_outside_band_std=("degree_minutes_outside_band", "std"),
            degree_minutes_outside_band_median=("degree_minutes_outside_band", "median"),
            degree_minutes_outside_band_min=("degree_minutes_outside_band", "min"),
            degree_minutes_outside_band_max=("degree_minutes_outside_band", "max"),
            degree_minutes_outside_band_p25=("degree_minutes_outside_band", p25),
            degree_minutes_outside_band_p75=("degree_minutes_outside_band", p75),
            underheating_degree_minutes_mean=("underheating_degree_minutes", "mean"),
            underheating_degree_minutes_std=("underheating_degree_minutes", "std"),
            underheating_degree_minutes_median=("underheating_degree_minutes", "median"),
            underheating_degree_minutes_min=("underheating_degree_minutes", "min"),
            underheating_degree_minutes_max=("underheating_degree_minutes", "max"),
            underheating_degree_minutes_p25=("underheating_degree_minutes", p25),
            underheating_degree_minutes_p75=("underheating_degree_minutes", p75),
            overheating_degree_minutes_mean=("overheating_degree_minutes", "mean"),
            overheating_degree_minutes_std=("overheating_degree_minutes", "std"),
            overheating_degree_minutes_median=("overheating_degree_minutes", "median"),
            overheating_degree_minutes_min=("overheating_degree_minutes", "min"),
            overheating_degree_minutes_max=("overheating_degree_minutes", "max"),
            overheating_degree_minutes_p25=("overheating_degree_minutes", p25),
            overheating_degree_minutes_p75=("overheating_degree_minutes", p75),
            overall_vs_best_pct_mean=("overall_vs_best_pct", "mean"),
            overall_vs_best_pct_std=("overall_vs_best_pct", "std"),
            overall_vs_best_pct_median=("overall_vs_best_pct", "median"),
            overall_vs_best_pct_min=("overall_vs_best_pct", "min"),
            overall_vs_best_pct_max=("overall_vs_best_pct", "max"),
            overall_vs_best_pct_p25=("overall_vs_best_pct", p25),
            overall_vs_best_pct_p75=("overall_vs_best_pct", p75),
            solver_optimal_steps=("solver_optimal_steps", "sum"),
            solver_optimal_inaccurate_steps=(
                "solver_optimal_inaccurate_steps",
                "sum",
            ),
            solver_fallbacks=("solver_fallbacks", "sum"),
            n_runs=("controller", "size"),
        )
        .sort_values("overall_vs_best_pct_mean")
        .reset_index(drop=True)
    )

    run_keys = ["seed", "variant"]
    if "room_id" in detailed_df.columns:
        run_keys = ["room_id", "seed", "variant"]
    winners = (
        detailed_df.sort_values(run_keys + ["overall_vs_best_pct"])
        .groupby(run_keys, as_index=False)
        .first()[["controller"]]
        .value_counts()
        .rename("wins")
        .reset_index()
    )
    summary_df = summary_df.merge(winners, on="controller", how="left")
    summary_df["wins"] = summary_df["wins"].fillna(0).astype(int)
    return summary_df


def _run_benchmark_for_config(
    config_path: str,
    scenarios_dir: str,
    seeds: list[int],
    variants: list[int],
    benchmark_root: Path,
    scenario_mode: str,
) -> dict:
    cfg = load_config(config_path)
    room_root = benchmark_root / cfg.room_id

    run_rows: list[dict] = []
    scenario_rows: list[dict] = []
    run_summaries: list[dict] = []

    for seed in seeds:
        train_dir = room_root / "runs" / f"train_seed_{seed:03d}"
        for variant in variants:
            run_dir = train_dir / f"eval_variant_{variant:02d}"
            result = run_simple(
                config_path=config_path,
                seed=seed,
                show_plot=False,
                scenarios_dir=scenarios_dir,
                scenario_variant=variant,
                out_dir=run_dir,
                train_out_dir=train_dir,
                scenario_mode=scenario_mode,
                reuse_trained_model=True,
            )

            avg_df = result.avg_df.copy()
            avg_df["seed"] = seed
            avg_df["variant"] = variant
            avg_df["room_id"] = cfg.room_id
            avg_df["train_dir"] = str(train_dir)
            avg_df["eval_dir"] = str(run_dir)
            avg_df["n_scenarios_loaded"] = result.run_summary["n_scenarios_loaded"]
            run_rows.extend(avg_df.to_dict(orient="records"))

            scenario_df = result.scenario_kpi_df.copy()
            scenario_df["seed"] = seed
            scenario_df["variant"] = variant
            scenario_df["room_id"] = cfg.room_id
            scenario_df["train_dir"] = str(train_dir)
            scenario_df["eval_dir"] = str(run_dir)
            scenario_rows.extend(scenario_df.to_dict(orient="records"))

            run_summaries.append(result.run_summary)

    return {
        "room_id": cfg.room_id,
        "detailed_df": pd.DataFrame(run_rows),
        "scenario_detailed_df": pd.DataFrame(scenario_rows),
        "run_summaries": run_summaries,
    }


def _build_benchmark_manifest(
    benchmark_root: Path,
    config_paths: list[str],
    seeds: list[int],
    variants: list[int],
    room_results: list[dict],
    combined_run_df: pd.DataFrame | None,
    combined_summary_df: pd.DataFrame | None,
    combined_scenario_df: pd.DataFrame | None,
    provenance: dict,
) -> dict:
    room_row_counts = {
        item["room_id"]: {
            "run_level_rows": int(len(item["detailed_df"])),
            "scenario_level_rows": int(len(item["scenario_detailed_df"])),
        }
        for item in room_results
    }
    coverage_runs = []
    for item in room_results:
        for summary in item["run_summaries"]:
            coverage_runs.append(
                {
                    "room_id": summary["room_id"],
                    "seed": summary["seed"],
                    "variant": summary["scenario_variant"],
                    "n_scenarios_loaded": summary["n_scenarios_loaded"],
                    "loaded_scenarios": summary["loaded_scenarios"],
                    "train_out_dir": summary["train_out_dir"],
                    "eval_out_dir": summary["eval_out_dir"],
                }
            )

    wins_total = None
    if combined_summary_df is not None:
        wins_total = int(combined_summary_df["wins"].sum())
    elif room_results:
        wins_total = int(
            _aggregate_summary(room_results[0]["detailed_df"])["wins"].sum()
        )

    return {
        "artifact_root": str(benchmark_root),
        "config_paths": config_paths,
        "rooms": [item["room_id"] for item in room_results],
        "seeds": seeds,
        "variants": variants,
        "expected_scenarios": EXPECTED_SCENARIO_TYPES,
        "row_counts": {
            "per_room": room_row_counts,
            "combined_run_level_rows": int(len(combined_run_df))
            if combined_run_df is not None
            else None,
            "combined_scenario_level_rows": int(len(combined_scenario_df))
            if combined_scenario_df is not None
            else None,
        },
        "wins_total": wins_total,
        "coverage_runs": coverage_runs,
        "provenance": provenance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run benchmark across multiple rooms and export summary."
    )
    parser.add_argument(
        "--configs",
        default=",".join(DEFAULT_CONFIGS),
        help="Comma-separated list of config paths (default: all 3 rooms).",
    )
    parser.add_argument("--scenarios-dir", default="data/scenarios")
    parser.add_argument(
        "--seeds",
        default="40,41,42,43,44",
        help="Comma-separated list of seeds.",
    )
    parser.add_argument(
        "--variants",
        default="0,1,2,3",
        help="Comma-separated list of scenario variants.",
    )
    parser.add_argument(
        "--scenario-mode",
        choices=["standard", "hidden_occupancy"],
        default=DEFAULT_SCENARIO_MODE,
        help="Scenario validation mode used by all benchmark runs.",
    )
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--experiment-tag", default=None)
    args = parser.parse_args()

    seeds = _parse_int_list(args.seeds)
    variants = _parse_int_list(args.variants)
    config_paths = [p.strip() for p in args.configs.split(",") if p.strip()]
    benchmark_root = resolve_artifact_root(args.artifact_root, args.experiment_tag)

    provenance = build_provenance_payload(
        {
            "stage": "benchmark",
            "artifact_root": str(benchmark_root),
            "config_paths": config_paths,
            "seeds": seeds,
            "variants": variants,
            "scenario_mode": args.scenario_mode,
            "scenarios_dir": args.scenarios_dir,
        }
    )
    write_json(benchmark_root / "benchmark_provenance.json", provenance)

    all_detailed: list[pd.DataFrame] = []
    all_scenario_detailed: list[pd.DataFrame] = []
    room_results: list[dict] = []

    for config_path in config_paths:
        cfg = load_config(config_path)
        print(f"\n{'=' * 60}")
        print(f"Benchmarking room: {cfg.room_id}  ({config_path})")
        print(f"{'=' * 60}")

        room_result = _run_benchmark_for_config(
            config_path=config_path,
            scenarios_dir=args.scenarios_dir,
            seeds=seeds,
            variants=variants,
            benchmark_root=benchmark_root,
            scenario_mode=args.scenario_mode,
        )
        detailed_df = room_result["detailed_df"]
        scenario_detailed_df = room_result["scenario_detailed_df"]
        if detailed_df.empty:
            print(f"WARNING: No results for {cfg.room_id}, skipping.")
            continue

        benchmark_dir = benchmark_root / cfg.room_id / "benchmark"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        detailed_path = benchmark_dir / "benchmark_detailed.csv"
        scenario_detailed_path = benchmark_dir / "benchmark_detailed_scenarios.csv"
        summary_path = benchmark_dir / "benchmark_summary.csv"

        room_summary = _aggregate_summary(detailed_df)
        detailed_df.to_csv(detailed_path, index=False)
        scenario_detailed_df.to_csv(scenario_detailed_path, index=False)
        room_summary.to_csv(summary_path, index=False)
        print(f"  Detailed run-level    : {detailed_path}")
        print(f"  Detailed scenario-level: {scenario_detailed_path}")
        print(f"  Summary               : {summary_path}")
        print(f"\n  Room summary ({cfg.room_id}):")
        print(
            room_summary[
                [
                    "controller",
                    "overall_vs_best_pct_mean",
                    "solver_optimal_inaccurate_steps",
                    "solver_fallbacks",
                    "wins",
                ]
            ].to_string(index=False)
        )

        room_results.append(room_result)
        all_detailed.append(detailed_df)
        all_scenario_detailed.append(scenario_detailed_df)

    if not all_detailed:
        print("No benchmark data produced.")
        return 1

    combined_df: pd.DataFrame | None = None
    combined_summary: pd.DataFrame | None = None
    combined_scenario_df: pd.DataFrame | None = None
    if len(all_detailed) > 1:
        combined_df = pd.concat(all_detailed, ignore_index=True)
        combined_scenario_df = pd.concat(all_scenario_detailed, ignore_index=True)
        combined_dir = benchmark_root / "benchmark_combined"
        combined_dir.mkdir(parents=True, exist_ok=True)

        combined_detailed_path = combined_dir / "benchmark_detailed_all_rooms.csv"
        combined_scenario_path = (
            combined_dir / "benchmark_detailed_scenarios_all_rooms.csv"
        )
        combined_summary_path = combined_dir / "benchmark_summary_all_rooms.csv"

        combined_df.to_csv(combined_detailed_path, index=False)
        combined_scenario_df.to_csv(combined_scenario_path, index=False)
        combined_summary = _aggregate_summary(combined_df)
        combined_summary.to_csv(combined_summary_path, index=False)

        print(f"\n{'=' * 60}")
        print("Combined benchmark (all rooms):")
        print(f"{'=' * 60}")
        print(f"  Detailed run-level    : {combined_detailed_path}")
        print(f"  Detailed scenario-level: {combined_scenario_path}")
        print(f"  Summary               : {combined_summary_path}")
        selected = combined_summary[
            [
                "controller",
                "overall_vs_best_pct_mean",
                "solver_optimal_inaccurate_steps",
                "solver_fallbacks",
                "wins",
            ]
        ]
        print(f"\n{selected.to_string(index=False)}")

    manifest = _build_benchmark_manifest(
        benchmark_root=benchmark_root,
        config_paths=config_paths,
        seeds=seeds,
        variants=variants,
        room_results=room_results,
        combined_run_df=combined_df,
        combined_summary_df=combined_summary,
        combined_scenario_df=combined_scenario_df,
        provenance=provenance,
    )
    write_json(benchmark_root / "benchmark_manifest.json", manifest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
