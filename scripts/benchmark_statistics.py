import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

PRIMARY_KPIS = [
    "comfort_violation_hours",
    "degree_minutes_outside_band",
]
SECONDARY_KPIS = [
    "rmse",
    "mae",
    "energy_proxy",
]
BASELINE_CONTROLLERS = ["pure_mpc", "pid", "onoff"]
ANALYSIS_UNITS = ("run", "scenario")


def _pair_key_columns(analysis_unit: str) -> list[str]:
    if analysis_unit == "run":
        return ["room_id", "seed", "variant"]
    if analysis_unit == "scenario":
        return ["room_id", "seed", "variant", "scenario"]
    raise ValueError(f"Unsupported analysis_unit: {analysis_unit}")


def _resolve_input_csv(
    input_csv: str | None,
    artifact_root: str,
    experiment_tag: str | None,
    analysis_unit: str,
) -> Path:
    if input_csv is not None:
        return Path(input_csv)

    root = Path(artifact_root)
    if experiment_tag:
        root = root / experiment_tag

    filename = (
        "benchmark_detailed_all_rooms.csv"
        if analysis_unit == "run"
        else "benchmark_detailed_scenarios_all_rooms.csv"
    )
    return root / "benchmark_combined" / filename


def _paired_bootstrap_ci(
    deltas: np.ndarray,
    stat_fn,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    if len(deltas) == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        sample = deltas[rng.integers(0, len(deltas), size=len(deltas))]
        stats.append(float(stat_fn(sample)))
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def _rank_biserial(deltas: np.ndarray) -> float:
    nonzero = deltas[np.abs(deltas) > 1e-12]
    if len(nonzero) == 0:
        return 0.0

    ranks = rankdata(np.abs(nonzero))
    w_plus = float(ranks[nonzero > 0].sum())
    w_minus = float(ranks[nonzero < 0].sum())
    denom = w_plus + w_minus
    if denom == 0.0:
        return 0.0
    return float((w_plus - w_minus) / denom)


def _holm_correct(p_values: pd.Series) -> pd.Series:
    if p_values.empty:
        return p_values

    m = len(p_values)
    order = np.argsort(p_values.to_numpy(dtype=float))
    sorted_p = p_values.to_numpy(dtype=float)[order]
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0
    for idx, p_value in enumerate(sorted_p):
        corrected = min(1.0, (m - idx) * p_value)
        running_max = max(running_max, corrected)
        adjusted[idx] = running_max

    result = np.empty(m, dtype=float)
    result[order] = adjusted
    return pd.Series(result, index=p_values.index)


def _require_columns(
    df: pd.DataFrame,
    columns: list[str],
    *,
    label: str,
) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _validate_pairing_inputs(
    df: pd.DataFrame,
    *,
    analysis_unit: str,
    key_cols: list[str],
) -> None:
    required_cols = key_cols + ["controller", *PRIMARY_KPIS, *SECONDARY_KPIS]
    _require_columns(
        df,
        required_cols,
        label=f"{analysis_unit}-level benchmark data",
    )

    duplicate_mask = df.duplicated(subset=key_cols + ["controller"], keep=False)
    if duplicate_mask.any():
        sample_rows = df.loc[duplicate_mask, key_cols + ["controller"]].head(5)
        raise ValueError(
            f"{analysis_unit}-level data must be unique for pair key + controller. "
            f"Duplicated rows detected for {key_cols + ['controller']}: "
            f"{sample_rows.to_dict(orient='records')}"
        )


def _compute_scope_stats(
    df: pd.DataFrame,
    *,
    scope: str,
    analysis_unit: str,
    key_cols: list[str],
) -> pd.DataFrame:
    _validate_pairing_inputs(df, analysis_unit=analysis_unit, key_cols=key_cols)
    rows: list[dict] = []

    for baseline in BASELINE_CONTROLLERS:
        hybrid_df = df[df["controller"] == "hybrid_mpc"]
        baseline_df = df[df["controller"] == baseline]
        if hybrid_df.empty or baseline_df.empty:
            continue

        hybrid_keys = hybrid_df[key_cols].drop_duplicates()
        baseline_keys = baseline_df[key_cols].drop_duplicates()
        expected_pairs = hybrid_keys.merge(baseline_keys, on=key_cols)

        merged = hybrid_df[key_cols + PRIMARY_KPIS + SECONDARY_KPIS].merge(
            baseline_df[key_cols + PRIMARY_KPIS + SECONDARY_KPIS],
            on=key_cols,
            suffixes=("_hybrid", "_baseline"),
        )
        if merged.empty:
            continue

        merged_pair_count = len(merged[key_cols].drop_duplicates())
        if merged_pair_count != len(expected_pairs):
            raise ValueError(
                f"{analysis_unit}-level merge dropped paired rows for baseline={baseline}. "
                f"Expected {len(expected_pairs)} pairs, got {merged_pair_count}."
            )

        for kpi in [*PRIMARY_KPIS, *SECONDARY_KPIS]:
            deltas = (
                merged[f"{kpi}_baseline"].to_numpy(dtype=float)
                - merged[f"{kpi}_hybrid"].to_numpy(dtype=float)
            )
            nonzero = deltas[np.abs(deltas) > 1e-12]
            if len(nonzero) == 0:
                statistic = 0.0
                p_value = 1.0
            else:
                result = wilcoxon(
                    deltas,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                statistic = float(result.statistic)
                p_value = float(result.pvalue)

            delta_mean_ci_low, delta_mean_ci_high = _paired_bootstrap_ci(
                deltas,
                np.mean,
            )
            delta_median_ci_low, delta_median_ci_high = _paired_bootstrap_ci(
                deltas,
                np.median,
            )
            rows.append(
                {
                    "analysis_unit": analysis_unit,
                    "scope": scope,
                    "controller_vs": baseline,
                    "kpi": kpi,
                    "kpi_tier": "primary" if kpi in PRIMARY_KPIS else "secondary",
                    "n_pairs": int(len(deltas)),
                    "pair_key_columns": ",".join(key_cols),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "effect_size": _rank_biserial(deltas),
                    "delta_mean": float(np.mean(deltas)),
                    "delta_median": float(np.median(deltas)),
                    "delta_mean_ci_low": delta_mean_ci_low,
                    "delta_mean_ci_high": delta_mean_ci_high,
                    "delta_median_ci_low": delta_median_ci_low,
                    "delta_median_ci_high": delta_median_ci_high,
                }
            )

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return stats_df

    stats_df["p_value_corrected"] = _holm_correct(stats_df["p_value"])
    return stats_df


def _analyze_unit(
    *,
    analysis_unit: str,
    input_csv: Path,
    output_dir: Path,
) -> tuple[Path, Path, pd.DataFrame]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Benchmark detailed CSV not found: {input_csv}")

    detailed_df = pd.read_csv(input_csv)
    key_cols = _pair_key_columns(analysis_unit)

    combined_stats = _compute_scope_stats(
        detailed_df,
        scope="combined",
        analysis_unit=analysis_unit,
        key_cols=key_cols,
    )

    per_room_frames = [
        _compute_scope_stats(
            room_df,
            scope=str(room_id),
            analysis_unit=analysis_unit,
            key_cols=key_cols,
        )
        for room_id, room_df in detailed_df.groupby("room_id")
    ]
    per_room_stats = (
        pd.concat(per_room_frames, ignore_index=True)
        if per_room_frames
        else pd.DataFrame()
    )

    combined_path = output_dir / f"benchmark_stats_{analysis_unit}_combined.csv"
    per_room_path = output_dir / f"benchmark_stats_{analysis_unit}_by_room.csv"
    combined_stats.to_csv(combined_path, index=False)
    per_room_stats.to_csv(per_room_path, index=False)
    return combined_path, per_room_path, combined_stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute paired benchmark statistics for hybrid_mpc vs baselines."
    )
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--experiment-tag", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--analysis-unit",
        choices=["run", "scenario", "both"],
        default="both",
    )
    args = parser.parse_args()

    if args.analysis_unit == "both" and args.input_csv is not None:
        raise ValueError(
            "--input-csv can only be used with --analysis-unit run or scenario."
        )

    units = ANALYSIS_UNITS if args.analysis_unit == "both" else (args.analysis_unit,)
    for analysis_unit in units:
        input_csv = _resolve_input_csv(
            input_csv=args.input_csv,
            artifact_root=args.artifact_root,
            experiment_tag=args.experiment_tag,
            analysis_unit=analysis_unit,
        )
        output_dir = Path(args.output_dir) if args.output_dir else input_csv.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        combined_path, per_room_path, combined_stats = _analyze_unit(
            analysis_unit=analysis_unit,
            input_csv=input_csv,
            output_dir=output_dir,
        )

        print(f"{analysis_unit} combined stats: {combined_path}")
        print(f"{analysis_unit} per-room stats: {per_room_path}")
        if not combined_stats.empty:
            print(
                combined_stats[
                    [
                        "analysis_unit",
                        "controller_vs",
                        "kpi",
                        "n_pairs",
                        "pair_key_columns",
                        "p_value",
                        "p_value_corrected",
                        "effect_size",
                        "delta_mean",
                        "delta_median",
                    ]
                ].to_string(index=False)
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
