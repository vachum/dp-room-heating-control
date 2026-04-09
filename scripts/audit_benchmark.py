import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from app.config import load_config
from app.scenario_utils import EXPECTED_SCENARIO_TYPES

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


def _resolve_root(artifact_root: str, experiment_tag: str | None) -> Path:
    root = Path(artifact_root)
    if experiment_tag:
        root = root / experiment_tag
    return root


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit benchmark artifact row counts, wins, and coverage metadata."
    )
    parser.add_argument(
        "--configs",
        default=",".join(DEFAULT_CONFIGS),
        help="Comma-separated config paths.",
    )
    parser.add_argument("--seeds", default="40,41,42,43,44")
    parser.add_argument("--variants", default="0,1,2,3")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--experiment-tag", default=None)
    args = parser.parse_args()

    config_paths = [p.strip() for p in args.configs.split(",") if p.strip()]
    seeds = _parse_int_list(args.seeds)
    variants = _parse_int_list(args.variants)
    root = _resolve_root(args.artifact_root, args.experiment_tag)

    expected_runs_per_room = len(seeds) * len(variants)
    expected_run_rows_per_room = expected_runs_per_room * 4
    expected_scenario_rows_per_room = (
        expected_runs_per_room * len(EXPECTED_SCENARIO_TYPES) * 4
    )

    errors: list[str] = []

    for config_path in config_paths:
        cfg = load_config(config_path)
        benchmark_dir = root / cfg.room_id / "benchmark"
        detailed_path = benchmark_dir / "benchmark_detailed.csv"
        scenario_path = benchmark_dir / "benchmark_detailed_scenarios.csv"
        summary_path = benchmark_dir / "benchmark_summary.csv"

        if not detailed_path.exists():
            errors.append(f"Missing run-level detailed CSV: {detailed_path}")
            continue
        if not scenario_path.exists():
            errors.append(f"Missing scenario-level detailed CSV: {scenario_path}")
            continue
        if not summary_path.exists():
            errors.append(f"Missing summary CSV: {summary_path}")
            continue

        detailed_df = pd.read_csv(detailed_path)
        scenario_df = pd.read_csv(scenario_path)
        summary_df = pd.read_csv(summary_path)

        if len(detailed_df) != expected_run_rows_per_room:
            errors.append(
                f"{cfg.room_id}: expected {expected_run_rows_per_room} run-level rows, "
                f"got {len(detailed_df)}"
            )
        if len(scenario_df) != expected_scenario_rows_per_room:
            errors.append(
                f"{cfg.room_id}: expected {expected_scenario_rows_per_room} scenario-level rows, "
                f"got {len(scenario_df)}"
            )

        wins_total = int(summary_df["wins"].sum())
        if wins_total != expected_runs_per_room:
            errors.append(
                f"{cfg.room_id}: expected wins sum {expected_runs_per_room}, got {wins_total}"
            )

    if len(config_paths) > 1:
        combined_dir = root / "benchmark_combined"
        combined_detailed_path = combined_dir / "benchmark_detailed_all_rooms.csv"
        combined_scenario_path = (
            combined_dir / "benchmark_detailed_scenarios_all_rooms.csv"
        )
        combined_summary_path = combined_dir / "benchmark_summary_all_rooms.csv"

        expected_combined_run_rows = expected_run_rows_per_room * len(config_paths)
        expected_combined_scenario_rows = (
            expected_scenario_rows_per_room * len(config_paths)
        )
        expected_combined_wins = expected_runs_per_room * len(config_paths)

        if not combined_detailed_path.exists():
            errors.append(f"Missing combined run-level CSV: {combined_detailed_path}")
        else:
            combined_df = pd.read_csv(combined_detailed_path)
            if len(combined_df) != expected_combined_run_rows:
                errors.append(
                    f"Combined: expected {expected_combined_run_rows} run-level rows, "
                    f"got {len(combined_df)}"
                )

        if not combined_scenario_path.exists():
            errors.append(
                f"Missing combined scenario-level CSV: {combined_scenario_path}"
            )
        else:
            combined_scenario_df = pd.read_csv(combined_scenario_path)
            if len(combined_scenario_df) != expected_combined_scenario_rows:
                errors.append(
                    f"Combined: expected {expected_combined_scenario_rows} scenario-level rows, "
                    f"got {len(combined_scenario_df)}"
                )

        if not combined_summary_path.exists():
            errors.append(f"Missing combined summary CSV: {combined_summary_path}")
        else:
            combined_summary_df = pd.read_csv(combined_summary_path)
            combined_wins = int(combined_summary_df["wins"].sum())
            if combined_wins != expected_combined_wins:
                errors.append(
                    f"Combined: expected wins sum {expected_combined_wins}, got {combined_wins}"
                )

    manifest_path = root / "benchmark_manifest.json"
    if not manifest_path.exists():
        errors.append(f"Missing benchmark manifest: {manifest_path}")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        coverage_runs = manifest.get("coverage_runs", [])
        expected_coverage_rows = len(config_paths) * expected_runs_per_room
        if len(coverage_runs) != expected_coverage_rows:
            errors.append(
                f"Manifest: expected {expected_coverage_rows} coverage runs, "
                f"got {len(coverage_runs)}"
            )
        for coverage in coverage_runs:
            loaded = coverage.get("loaded_scenarios", [])
            if coverage.get("n_scenarios_loaded") != len(EXPECTED_SCENARIO_TYPES):
                errors.append(
                    f"Manifest: incomplete scenario count for "
                    f"{coverage.get('room_id')} seed={coverage.get('seed')} "
                    f"variant={coverage.get('variant')}"
                )
            if list(loaded) != EXPECTED_SCENARIO_TYPES:
                errors.append(
                    f"Manifest: scenario ordering/coverage mismatch for "
                    f"{coverage.get('room_id')} seed={coverage.get('seed')} "
                    f"variant={coverage.get('variant')}: {loaded}"
                )

    if errors:
        print("Benchmark audit failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print(f"Benchmark audit passed for {root}")
    print(
        f"Per-room rows: run-level={expected_run_rows_per_room}, "
        f"scenario-level={expected_scenario_rows_per_room}, wins={expected_runs_per_room}"
    )
    if len(config_paths) > 1:
        print(
            f"Combined rows: run-level={expected_run_rows_per_room * len(config_paths)}, "
            f"scenario-level={expected_scenario_rows_per_room * len(config_paths)}, "
            f"wins={expected_runs_per_room * len(config_paths)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
