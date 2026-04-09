# Running Guide

## 1. Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. Generate Scenarios

Scenarios must be generated before any run. The benchmark pipeline will fail-fast if
any scenario file is missing.

```bash
python scripts/generate_scenarios.py --config configs/small_office.yaml --out-dir data/scenarios
python scripts/generate_scenarios.py --config configs/large_office.yaml --out-dir data/scenarios
python scripts/generate_scenarios.py --config configs/meeting_room.yaml --out-dir data/scenarios
```

This creates `data/scenarios/<room_id>/` with 30 CSV files per room
(6 scenario types × 5 variants each: `cold_day_v00.csv` … `summer_heatwave_v04.csv`).

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | required | YAML config for this room |
| `--out-dir` | `data/scenarios` | output root; `<room_id>/` is appended |
| `--variants` | `5` | number of variants per scenario type |
| `--steps` | `192` | time steps per scenario |
| `--seed` | `42` | RNG seed for reproducibility |

---

## 3. Validate Config (optional)

```bash
python scripts/validate_config.py configs/small_office.yaml
python scripts/validate_config.py configs/meeting_room.yaml --json   # also dumps JSON
```

Prints all parsed fields and derived values (`horizon_steps`, `warmup_steps`).
Exits non-zero if any validation rule fails.

---

## 4. Single Run

Runs all four controllers against all six scenario types for one
`(room, seed, variant)` combination.

```bash
python scripts/run_mvp.py \
  --config configs/small_office.yaml \
  --scenarios-dir data/scenarios \
  --scenario-variant 0 \
  --seed 42 \
  --warmup-minutes 120 \
  --lstm-seq-len 12 \
  --lstm-hidden-size 24 \
  --lstm-epochs 20 \
  --artifact-root artifacts \
  --experiment-tag smoke_test \
  --no-show
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | required | room YAML config |
| `--scenarios-dir` | required | root of generated scenario files |
| `--scenario-variant` | required | variant index (0–4) |
| `--seed` | required | RNG seed |
| `--warmup-minutes` | from YAML | overrides `experiment.warmup_minutes` |
| `--lstm-seq-len` | from YAML | overrides `experiment.lstm_seq_len` |
| `--lstm-hidden-size` | from YAML | overrides `experiment.lstm_hidden_size` |
| `--lstm-epochs` | from YAML | overrides `experiment.lstm_epochs` |
| `--artifact-root` | `artifacts` | output root directory |
| `--experiment-tag` | `None` | subdirectory label for this experiment |
| `--no-show` | flag | skip matplotlib window (use in CI/headless) |

**Key outputs** under `artifacts/<tag>/<room_id>/runs/seed_042_variant_00/`:

```
eval_run/
  provenance.json                  git commit, Python version, packages
  experiment_config_snapshot.json  effective settings + config hash
  run_summary.json                 KPI summary, solver stats, file paths
  kpi_simple_table.csv             KPI per controller x scenario
  kpi_simple_avg.csv               mean KPI per controller
  runtime_table.csv                step/solver/forecast timing stats
  comparison_simple.png            comparison plot

lstm_train/
  residual_lstm.pt                 model weights (PyTorch state_dict)
  residual_lstm_meta.json          normalization stats, hyperparameters, feature_columns
  residual_lstm_summary.json       train/val RMSE, sample counts, runtime
```

---

## 5. Full Benchmark

Runs all rooms × seeds × variants and aggregates results.
LSTM is trained once per `(room_id, seed)` and reused across variants.

```bash
python scripts/benchmark_report.py \
  --configs configs/small_office.yaml,configs/large_office.yaml,configs/meeting_room.yaml \
  --scenarios-dir data/scenarios \
  --seeds 40,41,42,43,44 \
  --variants 0,1,2,3 \
  --artifact-root artifacts \
  --experiment-tag final_20260321
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--configs` | required | comma-separated list of YAML configs |
| `--scenarios-dir` | required | scenario root directory |
| `--seeds` | required | comma-separated seed list |
| `--variants` | required | comma-separated variant index list |
| `--artifact-root` | `artifacts` | output root |
| `--experiment-tag` | required | experiment label |

**Key outputs** under `artifacts/<tag>/`:

```
benchmark_manifest.json                         coverage metadata for all runs
benchmark_combined/
  benchmark_detailed_all_rooms.csv              run-level KPI, all rooms
  benchmark_detailed_scenarios_all_rooms.csv    scenario-level KPI, all rooms
  benchmark_summary_all_rooms.csv               mean/std/median/p25/p75 per controller
<room_id>/benchmark/
  benchmark_detailed_<room_id>.csv
  benchmark_detailed_scenarios_<room_id>.csv
```

---

## 6. Audit

Verifies benchmark output integrity before running statistics.

```bash
python scripts/audit_benchmark.py \
  --configs configs/small_office.yaml,configs/large_office.yaml,configs/meeting_room.yaml \
  --seeds 40,41,42,43,44 \
  --variants 0,1,2,3 \
  --artifact-root artifacts \
  --experiment-tag final_20260321
```

Checks:
- run-level and scenario-level row counts match `n_rooms × n_seeds × n_variants × 4`
- `wins` column sums correctly (exactly one winner per run)
- all runs recorded in `benchmark_manifest.json`

Exits non-zero if any check fails.

---

## 7. Statistics

Paired Wilcoxon signed-rank test comparing `hybrid_mpc` against each baseline,
with Holm-Bonferroni correction and bootstrap confidence intervals.

```bash
python scripts/benchmark_statistics.py \
  --artifact-root artifacts \
  --experiment-tag final_20260321 \
  --analysis-unit both
```

Alternatively, point directly at a CSV:

```bash
python scripts/benchmark_statistics.py \
  --input-csv artifacts/final_20260321/benchmark_combined/benchmark_detailed_all_rooms.csv \
  --analysis-unit run
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--artifact-root` + `--experiment-tag` | — | auto-discovers combined CSV |
| `--input-csv` | — | explicit input path (alternative to above) |
| `--analysis-unit` | `both` | `run`, `scenario`, or `both` |

**Outputs** in `artifacts/<tag>/benchmark_combined/`:

```
benchmark_stats_run_combined.csv       all rooms, run-level pairs
benchmark_stats_run_by_room.csv        per room, run-level pairs
benchmark_stats_scenario_combined.csv  all rooms, scenario-level pairs
benchmark_stats_scenario_by_room.csv   per room, scenario-level pairs
```

Each row: `analysis_unit`, `pair_key_columns`, `controller_vs`, `kpi`,
`n_pairs`, `p_value`, `p_value_corrected`, `effect_size`, `delta_mean`, `delta_median`.

Pairing keys:
- `run` unit: `(room_id, seed, variant)`
- `scenario` unit: `(room_id, seed, variant, scenario)`

---

## 8. Tests

```bash
pytest -q             # fast summary
pytest -v tests/      # verbose per-test output
pytest tests/test_guardrails.py   # critical correctness tests only
```

See [docs/METRICS.md](METRICS.md) for what each test module covers.

---

## Troubleshooting

**`FileNotFoundError: scenario CSV missing`**
Run `generate_scenarios.py` for all three room configs before starting the benchmark.
The pipeline validates the complete scenario set upfront and fails immediately if anything
is missing.

**`ValueError: Not enough samples for LSTM training`**
The excitation dataset is too small. Increase `--steps` when generating scenarios
(default 192 is well above the 40-sample minimum) or check that the scenario CSV was
loaded correctly.

**High `solver_fallback_count` in run summary**
Fallbacks indicate the QP could not be solved optimally. Common causes:
- `du_max` is too tight relative to the required control effort
- `horizon_minutes` is very long with a short `sampling_minutes` (large `H`)
- Conflicting bound constraints (`u_min`/`u_max`)

A fallback rate above ~1 % of steps is worth investigating.

**`AssertionError: Feature vector has N elements, LSTM expects 7`**
`FEATURE_COUNT` in `controllers.py` and `FEATURE_COLUMNS` in `lstm.py` are out of sync.
Both must list the same 7 features in the same order. This is guarded by
`test_feature_columns_count_matches_feature_count` in the test suite.

**LSTM cache mismatch warning during benchmark**
The cached model's compatibility hash does not match current hyperparameters or config.
The benchmark will retrain automatically — this is the intended fail-safe. Check that
the YAML and CLI flags match what was used for the original training run.

**`KeyError` or column missing when loading scenarios**
Scenarios generated with an older script version may be missing new columns
(e.g. `unexpected_occupancy`). Regenerate with the current `generate_scenarios.py`.
