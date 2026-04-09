# Metrics, Statistics & Artifacts

## KPI Definitions

KPIs are computed by `app/metrics.py:compute_kpi()` on the **eval phase only**
(warm-up steps are excluded). The comfort band is ±0.5 °C around the active setpoint.

| KPI | Unit | Formula | Notes |
|-----|------|---------|-------|
| `comfort_violation_hours` | h | `sum(violations) × dt_min / 60` | steps where `|y - setpoint| > 0.5` |
| `degree_minutes_outside_band` | °C·min | `sum(|y - setpoint| - 0.5, 0) × dt_min` | primary KPI; integrates magnitude of violations |
| `underheating_degree_minutes` | °C·min | `sum(max(setpoint - 0.5 - y, 0)) × dt_min` | below-band component |
| `overheating_degree_minutes` | °C·min | `sum(max(y - setpoint - 0.5, 0)) × dt_min` | above-band component |
| `rmse` | °C | `sqrt(mean((y - setpoint)²))` | includes steps inside band |
| `mae` | °C | `mean(|y - setpoint|)` | includes steps inside band |
| `energy_proxy` | kW·h (norm.) | `sum(u) × dt_min / 60` | normalized to the room config; not comparable across rooms |

**Primary KPIs** for hypothesis testing: `comfort_violation_hours` and
`degree_minutes_outside_band`. The others are reported for completeness.

---

## Scenario Types

| Name | Base T_out | Base solar | Notes |
|------|-----------|------------|-------|
| `cold_day` | −3 °C | 180 W/m² | high heating demand |
| `sunny_day` | 4 °C | 450 W/m² | strong solar gain; tests overheating |
| `spring_day` | 12 °C | 350 W/m² | mild heating demand |
| `autumn_day` | 8 °C | 200 W/m² | weather fronts + cloud spikes |
| `mixed_day` | 5 °C | 250 W/m² | increased noise, manual setpoint overrides |
| `summer_heatwave` | 28 °C | 600 W/m² | heating-only systems cannot cool; violations expected |

Each type has 5 variants (`v00`–`v04`) generated with different seeds. Variant adds:
- day-level randomization of temperature mean, amplitude, and phase
- multi-day temperature drift
- beta-distributed cloudiness modulating solar gain
- Poisson occupancy with daily scaling
- stochastic unplanned occupancy events (hidden from controllers)

---

## Scenario CSV Columns

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime (UTC) | starts 2025-01-01T00:00:00Z, step = `sampling_minutes` |
| `T_out` | float °C | outdoor temperature |
| `solar` | float W/m² | solar irradiance (non-negative) |
| `occupancy` | float persons | Poisson forecast; visible to controllers |
| `occupancy_actual` | float persons | forecast + unplanned events; seen by plant only |
| `unexpected_occupancy` | float persons | `occupancy_actual - occupancy`; logged only |
| `setpoint` | float °C | comfort setpoint at each step |

---

## Artifact CSV Columns

### `kpi_simple_table.csv` (per controller × scenario)

| Column | Description |
|--------|-------------|
| `controller` | `onoff`, `pid`, `pure_mpc`, `hybrid_mpc` |
| `scenario` | scenario type name |
| `comfort_violation_hours` | |
| `degree_minutes_outside_band` | |
| `underheating_degree_minutes` | |
| `overheating_degree_minutes` | |
| `rmse` | |
| `mae` | |
| `energy_proxy` | |

### `benchmark_detailed_all_rooms.csv` (run-level, all rooms)

| Column | Description |
|--------|-------------|
| `room_id` | room identifier |
| `seed` | training seed |
| `variant` | scenario variant index |
| `controller` | controller name |
| `comfort_violation_hours` | mean across all scenario types in this run |
| `degree_minutes_outside_band` | mean |
| `energy_proxy` | mean |
| `rmse` | mean |
| `mae` | mean |
| `wins` | 1 if this controller had the best composite score in this run |
| `overall_vs_best_pct` | composite score relative to the best controller in this run (overview only; not for statistical inference) |

### `benchmark_stats_run_combined.csv` (Wilcoxon results)

| Column | Description |
|--------|-------------|
| `analysis_unit` | `run` or `scenario` |
| `pair_key_columns` | columns used to form pairs |
| `controller_vs` | baseline compared against `hybrid_mpc` |
| `kpi` | KPI name |
| `n_pairs` | number of matched pairs |
| `p_value` | raw two-sided Wilcoxon p-value |
| `p_value_corrected` | Holm-Bonferroni corrected p-value |
| `effect_size` | rank-biserial correlation |
| `delta_mean` | mean(baseline − hybrid_mpc); positive = hybrid is better for lower-is-better KPIs |
| `delta_median` | median(baseline − hybrid_mpc) |
| `delta_mean_ci_low`, `delta_mean_ci_high` | 95 % bootstrap CI on delta_mean (2000 resamples, seed 42) |
| `delta_median_ci_low`, `delta_median_ci_high` | 95 % bootstrap CI on delta_median (2000 resamples, seed 42) |

---

## Statistical Methodology

`benchmark_statistics.py` applies a **Wilcoxon signed-rank test** — non-parametric,
paired, appropriate for the small sample sizes here (~60 run-level pairs per baseline).

**Test setup:**
- Null hypothesis: median difference KPI(baseline) − KPI(hybrid_mpc) = 0
- Direction: two-sided (tests whether hybrid_mpc differs from baseline; use sign of delta_mean to determine direction)
- Baselines tested: `onoff`, `pid`, `pure_mpc`
- KPIs tested: all seven per baseline

**Multiple comparisons:**
Holm-Bonferroni correction is applied across all `(baseline × KPI)` combinations
within each analysis unit and room grouping.

**Effect size:**
Rank-biserial correlation: +1 means hybrid always wins, −1 means it always loses,
0 means random.

**Confidence intervals:**
Bootstrap CIs (2000 resamples, seed 42) on `delta_mean` and `delta_median`.

**Analysis units:**
- `run`: each `(room_id, seed, variant)` triple is one observation (KPI averaged over scenarios)
- `scenario`: each `(room_id, seed, variant, scenario_type)` is one observation

---

## Test Suite

| Module | What it tests |
|--------|---------------|
| `test_guardrails.py` | Critical correctness invariants (see below) |
| `test_kpi.py` | KPI formula correctness: comfort band, energy proxy, degree-minutes |
| `test_scenario_schema.py` | Scenario CSV structure: required columns, NaN, timestamp drift |
| `test_seed_reproducibility.py` | Same seed → identical outputs |
| `test_benchmark_statistics.py` | Wilcoxon test, rank-biserial, bootstrap CI |

### Guardrail tests

| Test | Invariant |
|------|-----------|
| `test_excitation_data_no_negative_u` | LSTM training data must never contain `u < 0` (heating-only scope) |
| `test_feature_columns_count_matches_feature_count` | `FEATURE_COLUMNS` in `lstm.py` and `FEATURE_COUNT` in `controllers.py` must agree |
| `test_solver_fallback_count_increments` | fallback counter increments when all solvers fail |
| `test_incompatible_cached_lstm_is_not_reused` | stale model is rejected and retrained when the compatibility hash changes |
| `test_build_eval_scenarios_requires_complete_set` | missing any scenario CSV causes an immediate fail-fast before any run starts |
| `test_single_run_uses_canonical_artifact_layout` | artifact directory structure matches the documented spec |
| `test_snapshot_contains_effective_experiment_settings` | `experiment_config_snapshot.json` captures the correct CLI-overridden values |
