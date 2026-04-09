from dataclasses import dataclass

import numpy as np
import pandas as pd

from .controllers import Disturbance, Observation


@dataclass
class ScenarioSpec:
    name: str
    steps: int
    sampling_minutes: int


class ToyRoomPlant:
    """Ground-truth room physics model.

    Uses deliberately higher coefficients than LinearRoomModel so that
    MPC operating on the nominal model accumulates systematic residuals.
    The LSTM is trained to predict and correct those residuals.
    """

    def __init__(
        self,
        dt_minutes: int,
        seed: int = 0,
        heating_gain: float = 2.8,
        leak_coef: float = 0.15,
        solar_coef: float = 0.003,
        occ_coef: float = 0.07,
    ) -> None:
        self.dt_hours = dt_minutes / 60.0
        self.rng = np.random.default_rng(seed)
        self.t_room = 21.0
        self.heating_gain = float(heating_gain)
        self.leak_coef = float(leak_coef)
        self.solar_coef = float(solar_coef)
        self.occ_coef = float(occ_coef)

    def reset(self) -> float:
        self.t_room = 21.0
        return self.t_room

    def step(self, u: float, d: Disturbance) -> float:
        # Euler integration of a first-order RC-like thermal model.
        # Small Gaussian noise (σ=0.03 °C) simulates sensor jitter.
        leak = self.leak_coef * (d.t_out - self.t_room)
        heat = self.heating_gain * float(np.clip(u, 0.0, 1.0))
        solar = self.solar_coef * d.solar
        people = self.occ_coef * d.occupancy
        noise = float(self.rng.normal(0.0, 0.03))
        self.t_room += self.dt_hours * (leak + heat + solar + people) + noise
        return self.t_room


def _setpoint(hour: float, day_setpoint: float, night_setpoint: float) -> float:
    if 7.0 <= hour < 20.0:
        return day_setpoint
    return night_setpoint


def _controller_occupancy(row: pd.Series) -> float:
    # Controllers receive only the forecast occupancy, not the actual value.
    # This enforces the information asymmetry that motivates LSTM correction.
    if "occupancy" in row.index:
        return float(row["occupancy"])
    if "occupancy_actual" in row.index:
        return float(row["occupancy_actual"])
    return 0.0


def _plant_occupancy(row: pd.Series) -> float:
    # The plant experiences actual occupancy including unplanned events.
    if "occupancy_actual" in row.index:
        return float(row["occupancy_actual"])
    return _controller_occupancy(row)


def make_scenario(
    spec: ScenarioSpec,
    day_setpoint: float,
    night_setpoint: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = spec.steps
    dt = pd.Timedelta(minutes=spec.sampling_minutes)
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    timestamps = [start + i * dt for i in range(n)]

    steps_per_day = int((24 * 60) / spec.sampling_minutes)
    hours = np.array([(i % steps_per_day) / steps_per_day * 24.0 for i in range(n)])
    day_idx = np.arange(n) // steps_per_day
    n_days = int(day_idx.max()) + 1 if n > 0 else 0

    if spec.name == "cold_day":
        base_temp, base_amp, base_solar = -3.0, 4.0, 180.0
    elif spec.name == "sunny_day":
        base_temp, base_amp, base_solar = 4.0, 6.0, 450.0
    elif spec.name == "spring_day":
        base_temp, base_amp, base_solar = 12.0, 6.0, 350.0
    elif spec.name == "autumn_day":
        base_temp, base_amp, base_solar = 8.0, 4.0, 200.0
    elif spec.name == "mixed_day":
        base_temp, base_amp, base_solar = 5.0, 8.0, 250.0
    elif spec.name == "summer_heatwave":
        base_temp, base_amp, base_solar = 28.0, 8.0, 600.0
    else:
        base_temp, base_amp, base_solar = 1.0, 5.0, 300.0

    # Per-day randomization: each day draws its own mean temperature, amplitude,
    # and phase shift so scenarios don't look like a repeated single-day pattern.
    daily_temp_mean = rng.normal(loc=base_temp, scale=1.6, size=n_days)
    daily_temp_amp = np.clip(
        rng.normal(loc=base_amp, scale=1.2, size=n_days),
        0.45 * base_amp,
        1.8 * base_amp,
    )
    daily_phase_h = rng.normal(loc=0.0, scale=0.8, size=n_days)
    multi_day_drift = np.cumsum(rng.normal(loc=0.0, scale=0.08, size=n_days))

    noise_scale = 1.5 if spec.name == "mixed_day" else 0.35
    temp_noise = rng.normal(loc=0.0, scale=noise_scale, size=n)

    t_out = (
        daily_temp_mean[day_idx]
        + daily_temp_amp[day_idx]
        * np.sin(2 * np.pi * (hours + daily_phase_h[day_idx]) / 24.0)
        + multi_day_drift[day_idx]
        + temp_noise
    )

    if spec.name in ("mixed_day", "autumn_day"):
        fronts = rng.random(n) < 0.01
        t_out -= fronts * rng.normal(loc=5.0, scale=2.0, size=n)

    # Daily cloudiness attenuates solar gain.
    cloudiness = np.clip(rng.beta(a=2.0, b=2.5, size=n_days), 0.05, 0.95)
    daily_solar_scale = np.clip(
        1.05 - 0.85 * cloudiness + 0.15 * rng.normal(size=n_days),
        0.10,
        1.25,
    )
    daily_day_shift = rng.normal(loc=0.0, scale=0.35, size=n_days)
    solar_shape = np.maximum(
        0.0,
        np.sin(np.pi * (hours - (6.0 + daily_day_shift[day_idx])) / 12.0),
    )
    solar = base_solar * daily_solar_scale[day_idx] * solar_shape
    solar += rng.normal(loc=0.0, scale=12.0, size=n)

    if spec.name in ("mixed_day", "autumn_day"):
        cloud_spikes = rng.random(n) < 0.05
        solar = np.where(cloud_spikes, solar * rng.uniform(0.1, 0.4, size=n), solar)

    solar = np.clip(solar, 0.0, None)

    # Occupancy with day-level variability and occasional daytime spikes.
    occ_day_scale = np.clip(rng.normal(loc=1.0, scale=0.25, size=n_days), 0.35, 1.9)
    work_hours = (hours >= 7.5) & (hours <= 18.5)
    is_weekend = (day_idx % 7) >= 5
    lam = np.where(
        work_hours, 2.4 * occ_day_scale[day_idx], 0.25 * occ_day_scale[day_idx]
    )
    lam = np.where(is_weekend, lam * 0.6, lam)
    occupancy_forecast = rng.poisson(lam).astype(float)

    # Hidden occupancy: unplanned meetings and late-stay events are injected into
    # the actual occupancy signal but are not visible to any controller.
    # The LSTM must learn to infer these from the thermal residual signal alone.
    unexpected_occupancy = np.zeros(n, dtype=float)
    event_prob = 0.10 if spec.name == "mixed_day" else 0.05
    event_starts = np.flatnonzero((rng.random(n) < event_prob) & work_hours)
    event_sizes = rng.integers(2, 8, size=len(event_starts))
    event_durations = rng.integers(1, 5, size=len(event_starts))
    for idx, size, duration in zip(event_starts, event_sizes, event_durations):
        end = min(n, idx + int(duration))
        unexpected_occupancy[idx:end] += float(size)

    if spec.name in ("mixed_day", "autumn_day"):
        off_hours = ~work_hours
        late_visits = np.flatnonzero((rng.random(n) < 0.02) & off_hours)
        if len(late_visits) > 0:
            unexpected_occupancy[late_visits] += rng.integers(
                1, 4, size=len(late_visits)
            )

    occupancy = np.clip(occupancy_forecast, 0.0, 15.0)
    occupancy_actual = np.clip(occupancy + unexpected_occupancy, 0.0, 15.0)
    unexpected_occupancy = occupancy_actual - occupancy

    setpoint = np.array([_setpoint(h, day_setpoint, night_setpoint) for h in hours])

    if spec.name == "mixed_day":
        manual_overrides = rng.random(n) < 0.02
        setpoint = np.where(
            manual_overrides, setpoint + rng.choice([-1.0, 1.0, 2.0], size=n), setpoint
        )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "T_out": t_out,
            "solar": solar,
            "occupancy": occupancy,
            "occupancy_actual": occupancy_actual,
            "unexpected_occupancy": unexpected_occupancy,
            "setpoint": setpoint,
        }
    )


def generate_excitation_data(
    plant: ToyRoomPlant,
    scenario_df: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Generate open-loop-ish dataset for residual-model training.

    Control input changes every 4 steps (pseudo-random step sequence) to
    excite a wide range of operating conditions without closing the loop.
    The resulting (y, u, disturbance, residual) traces are used to train
    ResidualLSTM to predict model error one step ahead.
    """

    rng = np.random.default_rng(seed)
    y = plant.reset()
    u = 0.0

    rows: list[dict] = []
    for i, row in scenario_df.reset_index(drop=True).iterrows():
        if i % 4 == 0:
            # Heating-only scope: control is in [0, 1]; negative values are
            # never issued by any controller and must not appear in training data.
            u = float(np.clip(0.5 + 0.4 * rng.normal(), 0.0, 1.0))

        controller_occ = _controller_occupancy(row)
        plant_occ = _plant_occupancy(row)
        d = Disturbance(
            t_out=float(row["T_out"]),
            solar=float(row["solar"]),
            occupancy=plant_occ,
        )
        y_next = plant.step(u=u, d=d)

        rows.append(
            {
                "timestamp": row["timestamp"],
                "y": float(y),
                "y_next": float(y_next),
                "u": u,
                "setpoint": float(row["setpoint"]),
                "T_out": d.t_out,
                "solar": d.solar,
                "occupancy": controller_occ,
                "occupancy_actual": plant_occ,
                "unexpected_occupancy": plant_occ - controller_occ,
            }
        )
        y = y_next

    return pd.DataFrame(rows)


def run_closed_loop(
    plant: ToyRoomPlant, controller, scenario_df: pd.DataFrame, warmup_steps: int = 0
) -> pd.DataFrame:
    """Run a single closed-loop simulation for one controller on one scenario.

    The plant and controller use separate occupancy signals: the controller
    sees only the forecast while the plant uses actual occupancy.  KPIs are
    computed on rows where phase == 'eval' (steps >= warmup_steps).
    """
    controller.reset()
    y = plant.reset()

    rows: list[dict] = []
    n_steps = len(scenario_df)

    for i, row in scenario_df.reset_index(drop=True).iterrows():
        controller_occ = _controller_occupancy(row)
        plant_occ = _plant_occupancy(row)
        d = Disturbance(
            t_out=float(row["T_out"]),
            solar=float(row["solar"]),
            occupancy=controller_occ,
        )
        plant_d = Disturbance(
            t_out=float(row["T_out"]),
            solar=float(row["solar"]),
            occupancy=plant_occ,
        )

        H = getattr(controller, "horizon_steps", 0)
        future_disturbances = []
        future_setpoints = []
        if H > 0:
            # Build the forecast window.  Past the end of the scenario we
            # hold the last available row (zero-order hold) rather than
            # extrapolating, which keeps disturbances bounded.
            for k in range(H):
                idx = min(i + k, n_steps - 1)
                future_row = scenario_df.iloc[idx]
                future_disturbances.append(
                    Disturbance(
                        t_out=float(future_row["T_out"]),
                        solar=float(future_row["solar"]),
                        occupancy=_controller_occupancy(future_row),
                    )
                )
                future_setpoints.append(float(future_row["setpoint"]))

        obs = Observation(
            y=float(y),
            setpoint=float(row["setpoint"]),
            disturbance=d,
            future_disturbances=future_disturbances if H > 0 else None,
            future_setpoints=future_setpoints if H > 0 else None,
        )
        u = float(np.clip(controller.act(obs), 0.0, 1.0))
        y_next = plant.step(u=u, d=plant_d)

        if hasattr(controller, "observe_transition"):
            controller.observe_transition(obs, u, y_next)

        rows.append(
            {
                "timestamp": row["timestamp"],
                "y": float(obs.y),
                "u": u,
                "setpoint": float(obs.setpoint),
                "T_out": d.t_out,
                "solar": d.solar,
                "occupancy": d.occupancy,
                "occupancy_actual": plant_d.occupancy,
                "unexpected_occupancy": plant_d.occupancy - d.occupancy,
                "residual_pred": float(getattr(controller, "_last_residual_pred", 0.0)),
                "residual_obs": float(getattr(controller, "_last_residual_obs", 0.0)),
                "unexpected_occupancy_pred": float(
                    getattr(controller, "_last_unexpected_occupancy_pred", 0.0)
                ),
                "unexpected_occupancy_obs": float(
                    getattr(controller, "_last_unexpected_occupancy_obs", 0.0)
                ),
                "step_runtime_ms": float(
                    getattr(controller, "_last_step_runtime_ms", 0.0)
                ),
                "solver_runtime_ms": float(
                    getattr(controller, "_last_solver_runtime_ms", 0.0)
                ),
                "forecast_runtime_ms": float(
                    getattr(controller, "_last_forecast_runtime_ms", 0.0)
                ),
                "phase": "warmup" if i < warmup_steps else "eval",
            }
        )

        y = y_next

    return pd.DataFrame(rows)
