# Architecture

## Components

```
app/
  config.py           AppConfig dataclass; validates YAML, computes derived fields
  controllers.py      OnOffController, PIDController, MPCController, HybridMPCController
                      LinearRoomModel (prediction model used inside MPC)
  simulation.py       ToyRoomPlant (physics), make_scenario(), run_closed_loop()
  lstm.py             ResidualLSTM (PyTorch), train_residual_lstm(), ResidualLSTMPredictor
  metrics.py          compute_kpi() --> KPIResult
  scenario_utils.py   validate_scenario_df(), load_scenario_csv()
  experiment_utils.py provenance recording, artifact path helpers, config hashing

scripts/
  generate_scenarios.py    generates reproducible CSV scenario files
  validate_config.py       parses a YAML config and prints all values
  run_mvp.py               single end-to-end run (one room x seed x variant)
  benchmark_report.py      full benchmark (rooms x seeds x variants)
  benchmark_statistics.py  paired Wilcoxon test, hybrid_mpc vs baselines
  audit_benchmark.py       integrity checks on benchmark outputs
```

### Component dependencies

```
AppConfig
  used by: run_mvp, benchmark_report, generate_scenarios, validate_config

ToyRoomPlant                  <-- ground truth physics
LinearRoomModel               <-- MPC prediction model (intentionally worse than plant)

MPCController
  solves QP via cvxpy: OSQP -> ECOS -> SCS (fallback chain)
  optionally accepts a ResidualLSTMPredictor for feedforward correction

HybridMPCController
  MPCController with residual_predictor required (otherwise identical)

ResidualLSTMPredictor
  maintains a rolling feature history deque
  calls ResidualLSTM.forward() at each step

run_closed_loop
  drives controller <-> plant interaction step by step
  calls controller.observe_transition() so LSTM state stays current

run_simple (run_mvp.py)
  loads config and scenarios
  trains LSTM once
  calls run_closed_loop for each controller x scenario combination
  writes KPI tables, comparison plot, provenance JSONs

benchmark_report.py
  calls run_simple across rooms x seeds x variants
  trains LSTM once per (room_id x seed), reuses across variants
  aggregates into per-room and combined CSV exports
```

---

## Data Flow

```
configs/<room>.yaml
        |
        v
generate_scenarios.py
        |
        v
data/scenarios/<room_id>/<type>_vNN.csv     (5 variants x 6 types x 3 rooms = 90 files)
        |
        v
run_mvp.py  OR  benchmark_report.py
  |
  +-- generate_excitation_data()   open-loop dataset for LSTM training
  |
  +-- train_residual_lstm()        trains on excitation data, saves .pt + meta.json
  |
  +-- for each controller x scenario:
  |     run_closed_loop()
  |       --> controller.act(obs)
  |       --> plant.step(u, disturbance)
  |       --> controller.observe_transition(obs, u, y_next)   updates LSTM history
  |
  +-- compute_kpi()                warm-up steps excluded
  |
  +-- write artifacts (CSV, JSON, PNG)
        |
        v
audit_benchmark.py                validates row counts, win sums, coverage metadata
        |
        v
benchmark_statistics.py           Wilcoxon tests, effect sizes, bootstrap CIs
```

---

## Key Design Decisions

### 1. Intentional model mismatch

`ToyRoomPlant` and `LinearRoomModel` use different coefficients on purpose.
The MPC model systematically underestimates heat loss and solar/occupancy gain:

| Parameter       | Plant (reality) | MPC model | Underestimation |
|-----------------|-----------------|-----------|-----------------|
| `leak_coef`     | 0.15–0.28       | 0.08–0.18 | ~35–47 %        |
| `solar_coef`    | 0.003–0.010     | 0.0012–0.006 | ~40–60 %     |
| `occ_coef`      | 0.07–0.15       | 0.03–0.10 | ~33–57 %        |

This means Pure MPC will chronically underestimate heat loss and overshoot when solar
or occupancy is high. The LSTM learns to predict these systematic residuals and feeds
a correction sequence into the QP at every step.

### 2. Hidden occupancy

Each scenario carries two occupancy signals:

| Column               | Who sees it   | Contents                                    |
|----------------------|---------------|---------------------------------------------|
| `occupancy`          | controller    | deterministic Poisson forecast              |
| `occupancy_actual`   | plant         | forecast + random unplanned events          |
| `unexpected_occupancy` | logging only | `occupancy_actual - occupancy`              |

Controllers are given the forecast; the plant experiences the actual value. This
simulates a calendar-based prediction with imperfect real-time occupancy sensing.
Unplanned meetings and late visits produce thermal spikes the MPC cannot anticipate
from its model alone — the LSTM infers them from the residual signal.

### 3. LSTM residual feedback loop

At every simulation step the hybrid controller:

1. Calls `_feature()` to build a 7-element vector using **previous** u and residual
   (using `u[k-1]` rather than `u[k]` keeps the QP linear and convex)
2. Passes the feature to `ResidualLSTMPredictor.forecast()` which returns a residual
   sequence of length `horizon_steps`
3. Feeds that sequence as `p_residual_seq` into the CVXPY parameter, so each predicted
   state in the QP is shifted by the expected model error

After the plant responds, `observe_transition()` computes the actual residual
(`y_next_real - y_next_predicted`) and appends it to the predictor's rolling history.

Feature vector (order is fixed — must match `FEATURE_COLUMNS` in `lstm.py`):

```
[ y,  u_prev,  T_out,  solar,  occupancy,  setpoint,  residual_prev ]
```

### 4. Solver fallback chain

MPC uses CVXPY with a three-solver cascade: **OSQP → ECOS → SCS**.

- OSQP is tried first — fastest for warm-started QPs
- ECOS and SCS are fallbacks if OSQP returns an infeasible or inaccurate status
- If all three fail, the controller repeats `u_prev` (safe hold)

`solver_fallback_count` is logged in every run summary. A count above ~1 % of total
steps usually indicates a poorly conditioned problem — check `du_max`, horizon length,
and whether `u_min`/`u_max` constraints are compatible with the scenario.

### 5. LSTM reuse across benchmark variants

For each `(room_id, seed)` pair the benchmark trains one LSTM and reuses it across
all scenario variants. Reuse is safe only when hyperparameters and room config are
identical. A compatibility hash is computed from:

```
room_id, sampling_minutes, horizon_steps, feature_columns,
seq_len, hidden_size, epochs, seed, config_hash
```

If the hash does not match, the model is retrained unconditionally (fail-safe).

### 6. Heating-only scope

`u_min: 0.0` in all production configs means the system can only add heat, not remove
it. This is deliberate: the scope is a heating-only building system. The `summer_heatwave`
scenario will therefore show temperature violations for all controllers — this is
expected and documented behavior, not a bug.
