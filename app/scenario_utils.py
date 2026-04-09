from pathlib import Path

import pandas as pd

EXPECTED_SCENARIO_TYPES = [
    "cold_day",
    "sunny_day",
    "spring_day",
    "autumn_day",
    "mixed_day",
    "summer_heatwave",
]

SCENARIO_REQUIRED_COLUMNS = ["timestamp", "T_out", "solar", "occupancy", "setpoint"]
SCENARIO_OPTIONAL_COLUMNS = ["occupancy_actual", "unexpected_occupancy"]
HIDDEN_OCC_REQUIRED_COLUMNS = ["occupancy_actual", "unexpected_occupancy"]
DEFAULT_SCENARIO_MODE = "hidden_occupancy"


def _source_label(source: str | Path | None) -> str:
    if source is None:
        return "Scenario"
    return f"Scenario {source}"


def validate_scenario_df(
    df: pd.DataFrame,
    mode: str = DEFAULT_SCENARIO_MODE,
    expected_sampling_minutes: int | None = None,
    source: str | Path | None = None,
) -> None:
    label = _source_label(source)

    missing_base = [c for c in SCENARIO_REQUIRED_COLUMNS if c not in df.columns]
    if missing_base:
        raise ValueError(f"{label} is missing required columns: {missing_base}")

    if mode == "hidden_occupancy":
        missing_hidden = [c for c in HIDDEN_OCC_REQUIRED_COLUMNS if c not in df.columns]
        if missing_hidden:
            raise ValueError(
                f"{label} requires hidden-occupancy columns: {missing_hidden}. "
                "Regenerate scenarios with occupancy_actual included."
            )
    elif mode != "standard":
        raise ValueError(f"Unsupported scenario validation mode: {mode}")

    numeric_cols = ["T_out", "solar", "occupancy", "setpoint"]
    if mode == "hidden_occupancy":
        numeric_cols.extend(HIDDEN_OCC_REQUIRED_COLUMNS)

    nan_cols = [col for col in numeric_cols if df[col].isna().any()]
    if nan_cols:
        raise ValueError(f"{label} contains NaN values in columns: {nan_cols}")

    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{label} contains invalid timestamps")
    if not timestamps.is_monotonic_increasing:
        raise ValueError(f"{label} timestamps are not strictly non-decreasing")

    diffs = timestamps.diff().dropna().dt.total_seconds() / 60.0
    if len(diffs) == 0:
        return

    expected_dt = (
        float(expected_sampling_minutes)
        if expected_sampling_minutes is not None
        else float(diffs.mode().iloc[0])
    )
    if not diffs.sub(expected_dt).abs().lt(0.1).all():
        raise ValueError(
            f"{label} has irregular timestamp sampling "
            f"(expected {expected_dt:g} min steps)"
        )


def load_scenario_csv(
    path: str | Path,
    mode: str = DEFAULT_SCENARIO_MODE,
    expected_sampling_minutes: int | None = None,
) -> pd.DataFrame:
    csv_path = Path(path)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    validate_scenario_df(
        df,
        mode=mode,
        expected_sampling_minutes=expected_sampling_minutes,
        source=csv_path,
    )
    keep_cols = [
        *SCENARIO_REQUIRED_COLUMNS,
        *[c for c in SCENARIO_OPTIONAL_COLUMNS if c in df.columns],
    ]
    return df[keep_cols].copy()
