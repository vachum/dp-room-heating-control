# Configuration Reference

Each room has a YAML configuration file in `configs/`. The file controls the room's
physical model parameters, MPC prediction model, comfort setpoints, sampling rate, and
default experiment hyperparameters.

## Sections

### `metadata`

```yaml
metadata:
  room_id: "small_office"   # string; used as the artifact directory name
  sampling_minutes: 10      # int; simulation time step [min]
  horizon_minutes: 180      # int; MPC prediction horizon [min]
```

| Field | Type | Constraint | Notes |
|-------|------|-----------|-------|
| `room_id` | string | unique | used in artifact paths and CSV `room_id` column |
| `sampling_minutes` | int | > 0 | determines `dt` for the plant and the MPC model |
| `horizon_minutes` | int | divisible by `sampling_minutes` | `horizon_steps = horizon_minutes // sampling_minutes` |

---

### `constraints`

```yaml
constraints:
  u_min: 0.0    # lower bound on control action
  u_max: 1.0    # upper bound on control action
  du_max: 1.2   # max change per step (rate limiter)
```

| Field | Type | Constraint | Notes |
|-------|------|-----------|-------|
| `u_min` | float | -1.0 ≤ u_min < u_max | `0.0` enforces heating-only scope |
| `u_max` | float | u_min < u_max ≤ 1.0 | `1.0` = full heating power |
| `du_max` | float | > 0 | tighter values smooth output, but increase fallback risk |

`u_min: 0.0` means the system can only add heat (no cooling). All production configs
use this setting. Negative values would enable cooling but are outside the thesis scope.

---

### `comfort`

```yaml
comfort:
  day_setpoint: 22.0     # target temperature 07:00–20:00 [°C]
  night_setpoint: 20.5   # target temperature 20:00–07:00 [°C]
```

The KPI comfort band is ±0.5 °C around the active setpoint.
Scenarios can also apply short manual override events (mixed_day type only).

---

### `experiment`

```yaml
experiment:
  warmup_minutes: 120   # simulation warm-up phase excluded from KPIs [min]
  lstm_seq_len: 12      # LSTM input sequence length [steps]
  lstm_hidden_size: 24  # LSTM hidden state size
  lstm_epochs: 20       # training epochs
```

These are defaults; all four fields can be overridden with CLI flags in `run_mvp.py`
and `benchmark_report.py`.

| Field | Notes |
|-------|-------|
| `warmup_minutes` | must be divisible by `sampling_minutes`; `warmup_steps = warmup_minutes // sampling_minutes` |
| `lstm_seq_len` | shorter sequences mean less context but faster inference; 12 steps works well at 10–20 min sampling |
| `lstm_hidden_size` | 24 units balances capacity against overfitting on the ~150-sample training set |
| `lstm_epochs` | more epochs can improve fit but risk overfitting on the short excitation dataset |

---

### `model`

Defines separate parameter sets for the plant (virtual reality) and the MPC prediction
model. The mismatch between them is intentional — see [ARCHITECTURE.md](ARCHITECTURE.md).

```yaml
model:
  # ToyRoomPlant — ground truth physics
  plant_heating_gain: 3.5   # °C/h at full power (u=1)
  plant_leak_coef: 0.15     # heat loss rate (proportional to T_room - T_out)
  plant_solar_coef: 0.003   # solar heat gain [°C / (h · W/m²)]
  plant_occ_coef: 0.07      # occupancy heat gain [°C / (h · person)]

  # LinearRoomModel — MPC uses this for QP prediction
  mpc_heating_gain: 3.5     # can match plant or differ
  mpc_leak_coef: 0.08       # intentionally underestimated
  mpc_solar_coef: 0.0012    # intentionally underestimated
  mpc_occ_coef: 0.03        # intentionally underestimated
```

All eight fields are required. Values must be positive floats.

---

## Room Configurations

### small_office

```yaml
metadata:
  room_id: "small_office"
  sampling_minutes: 10
  horizon_minutes: 180       # horizon_steps = 18

constraints:
  u_min: 0.0
  u_max: 1.0
  du_max: 1.2                # permissive rate limit

comfort:
  day_setpoint: 22.0
  night_setpoint: 20.5

experiment:
  warmup_minutes: 120        # warmup_steps = 12
  lstm_seq_len: 12
  lstm_hidden_size: 24
  lstm_epochs: 20

model:
  plant_heating_gain: 3.5
  plant_leak_coef: 0.15
  plant_solar_coef: 0.003
  plant_occ_coef: 0.07

  mpc_heating_gain: 3.5
  mpc_leak_coef: 0.08        # ~47% underestimation
  mpc_solar_coef: 0.0012     # ~60% underestimation
  mpc_occ_coef: 0.03         # ~57% underestimation
```

### large_office

```yaml
metadata:
  room_id: "large_office"
  sampling_minutes: 15
  horizon_minutes: 180       # horizon_steps = 12

constraints:
  u_min: 0.0
  u_max: 1.0
  du_max: 0.3                # tight rate limit (larger thermal mass)

comfort:
  day_setpoint: 22.5
  night_setpoint: 21.0

experiment:
  warmup_minutes: 120        # warmup_steps = 8
  lstm_seq_len: 12
  lstm_hidden_size: 24
  lstm_epochs: 20

model:
  plant_heating_gain: 5.8
  plant_leak_coef: 0.20
  plant_solar_coef: 0.005
  plant_occ_coef: 0.09

  mpc_heating_gain: 5.0      # ~14% underestimation
  mpc_leak_coef: 0.15        # ~25% underestimation
  mpc_solar_coef: 0.003      # ~40% underestimation
  mpc_occ_coef: 0.05         # ~44% underestimation
```

### meeting_room

```yaml
metadata:
  room_id: "meeting_room"
  sampling_minutes: 20
  horizon_minutes: 200       # horizon_steps = 10

constraints:
  u_min: 0.0
  u_max: 1.0
  du_max: 0.8

comfort:
  day_setpoint: 21.0
  night_setpoint: 18.0       # larger night setback

experiment:
  warmup_minutes: 120        # warmup_steps = 6
  lstm_seq_len: 12
  lstm_hidden_size: 24
  lstm_epochs: 20

model:
  # Large windows (high solar gain), older leaky building, high occupancy density
  plant_heating_gain: 4.2
  plant_leak_coef: 0.28
  plant_solar_coef: 0.010
  plant_occ_coef: 0.15

  mpc_heating_gain: 4.2      # identical (heating response is modelled accurately)
  mpc_leak_coef: 0.18        # ~36% underestimation
  mpc_solar_coef: 0.006      # ~40% underestimation
  mpc_occ_coef: 0.10         # ~33% underestimation
```

---

## Validation Rules

`config.py` enforces these rules at load time:

- `horizon_minutes % sampling_minutes == 0`
- `warmup_minutes % sampling_minutes == 0`
- `-1.0 <= u_min < u_max <= 1.0`
- `du_max > 0`
- all `*_gain` and `*_coef` values > 0

`validate_config.py` runs these checks and prints derived fields without starting a run.

---

## Derived Fields

These are computed automatically and do not appear in YAML:

| Field | Formula |
|-------|---------|
| `horizon_steps` | `horizon_minutes // sampling_minutes` |
| `warmup_steps` | `warmup_minutes // sampling_minutes` |

Both are written to `experiment_config_snapshot.json` for audit purposes.
