import logging
import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

logger = logging.getLogger(__name__)

ALLOWED_SOLVER_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}

# Must match FEATURE_COLUMNS in residual_lstm.py.
# Any change to _feature() must be mirrored in residual_lstm.FEATURE_COLUMNS.
# Guarded by test_feature_columns_count_matches_feature_count.
FEATURE_COUNT = 7


@dataclass
class Disturbance:
    t_out: float
    solar: float
    occupancy: float


@dataclass
class Observation:
    y: float
    setpoint: float
    disturbance: Disturbance
    future_disturbances: list[Disturbance] | None = None
    future_setpoints: list[float] | None = None


@dataclass
class MPCWeights:
    q_track: float = 2.0
    r_energy: float = 0.20
    r_du: float = 0.15
    deadband: float = 0.4


class LinearRoomModel:
    """Linear model used by MPC for prediction."""

    def __init__(
        self,
        dt_minutes: int,
        heating_gain: float = 1.8,
        leak_coef: float = 0.08,
        solar_coef: float = 0.0012,
        occ_coef: float = 0.03,
    ) -> None:
        dt_h = dt_minutes / 60.0
        # Nominal model is intentionally simpler than the plant.
        # Hybrid MPC uses LSTM residuals to compensate this mismatch.
        self.a = 1.0 - float(leak_coef) * dt_h
        self.b_u = float(heating_gain) * dt_h
        self.b_t_out = float(leak_coef) * dt_h
        self.b_solar = float(solar_coef) * dt_h
        self.b_occ = float(occ_coef) * dt_h

    def predict_next(
        self,
        y: float,
        u: float,
        d: Disturbance,
        residual: float = 0.0,
    ) -> float:
        return (
            self.a * float(y)
            + self.b_u * float(u)
            + self.b_t_out * float(d.t_out)
            + self.b_solar * float(d.solar)
            + self.b_occ * float(d.occupancy)
            + float(residual)
        )


class OnOffController:
    name = "onoff"

    def __init__(
        self, deadband: float = 0.4, u_min: float = 0.0, u_max: float = 1.0
    ) -> None:
        self.deadband = deadband
        self.u_min = u_min
        self.u_max = u_max
        self._u = 0.0
        self._last_step_runtime_ms = 0.0

    def reset(self) -> None:
        self._u = 0.0
        self._last_step_runtime_ms = 0.0

    def act(self, obs: Observation) -> float:
        t0 = time.perf_counter()
        # Classic hysteresis: switch states only when crossing the band edges,
        # never inside the deadband.  No output in the neutral zone prevents
        # chattering and keeps the baseline simple.
        if obs.y < obs.setpoint - self.deadband:
            self._u = self.u_max
        elif obs.y > obs.setpoint + self.deadband:
            self._u = self.u_min
        else:
            self._u = 0.0
        self._last_step_runtime_ms = (time.perf_counter() - t0) * 1000.0
        return self._u


class PIDController:
    name = "pid"

    def __init__(
        self,
        kp: float = 0.25,
        ki: float = 0.01,
        kd: float = 0.02,
        u_min: float = 0.0,
        u_max: float = 1.0,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.u_min = u_min
        self.u_max = u_max
        self._integral = 0.0
        self._prev_err = 0.0
        self._last_step_runtime_ms = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_err = 0.0
        self._last_step_runtime_ms = 0.0

    def act(self, obs: Observation) -> float:
        t0 = time.perf_counter()
        err = obs.setpoint - obs.y
        derivative = err - self._prev_err

        # Conditional integration anti-windup:
        # freeze integral when actuator is saturated and error pushes further
        # into the same saturation direction.
        u_pre = self.kp * err + self.ki * self._integral + self.kd * derivative
        pushing_high = u_pre >= self.u_max and err > 0.0
        pushing_low = u_pre <= self.u_min and err < 0.0
        if not (pushing_high or pushing_low):
            self._integral += err

        self._prev_err = err
        u = self.kp * err + self.ki * self._integral + self.kd * derivative
        self._last_step_runtime_ms = (time.perf_counter() - t0) * 1000.0
        return float(np.clip(u, self.u_min, self.u_max))


class MPCController:
    name = "pure_mpc"

    def __init__(
        self,
        model: LinearRoomModel,
        horizon_steps: int,
        u_min: float,
        u_max: float,
        du_max: float,
        weights: MPCWeights | None = None,
        residual_predictor=None,
    ) -> None:
        self.model = model
        self.horizon_steps = horizon_steps
        self.u_min = u_min
        self.u_max = u_max
        self.du_max = du_max
        self.weights = weights or MPCWeights()
        self.residual_predictor = residual_predictor

        self._u_prev = 0.0
        self._last_feature: np.ndarray | None = None
        self._last_residual_pred = 0.0
        self._last_residual_obs = 0.0
        self._last_unexpected_occupancy_pred = 0.0
        self._last_unexpected_occupancy_obs = 0.0

        self._last_step_runtime_ms = 0.0
        self._last_forecast_runtime_ms = 0.0
        self._last_solver_runtime_ms = 0.0

        self._step = 0
        self._solver_fallback_count = 0
        self._solver_status_counts = {
            cp.OPTIMAL: 0,
            cp.OPTIMAL_INACCURATE: 0,
            "fallback": 0,
        }
        self._solver_attempt_status_counts: dict[str, int] = {}
        self._last_solver_status = "not_run"

        # Build the CVXPY problem once at construction time; use cp.Parameter
        # for all values that change step-to-step so the problem can be
        # re-solved without recompiling the computation graph.
        self._setup_problem()

    def _setup_problem(self):
        H = self.horizon_steps
        self.y = cp.Variable(H + 1)
        self.u = cp.Variable(H)

        self.p_y0 = cp.Parameter()
        self.p_u_prev = cp.Parameter()
        self.p_d_t_out = cp.Parameter(H)
        self.p_d_solar = cp.Parameter(H)
        self.p_d_occ = cp.Parameter(H)
        self.p_sp = cp.Parameter(H)
        self.p_residual_seq = cp.Parameter(H)

        cons = [self.y[0] == self.p_y0]
        obj = 0.0

        for k in range(H):
            cons += [
                self.y[k + 1]
                == self.model.a * self.y[k]
                + self.model.b_u * self.u[k]
                + self.model.b_t_out * self.p_d_t_out[k]
                + self.model.b_solar * self.p_d_solar[k]
                + self.model.b_occ * self.p_d_occ[k]
                + self.p_residual_seq[k],
                self.u[k] >= self.u_min,
                self.u[k] <= self.u_max,
            ]

            if k == 0:
                du = self.u[k] - self.p_u_prev
            else:
                du = self.u[k] - self.u[k - 1]

            cons += [cp.abs(du) <= self.du_max]

            if self.weights.deadband > 0:
                # Dead-band penalty: only penalize deviations that exceed
                # the comfort band.  This keeps the QP convex (cp.pos is
                # equivalent to max(0, ·)) while tolerating small offsets
                # without driving the heater constantly.
                slack_high = cp.pos(
                    self.y[k + 1] - (self.p_sp[k] + self.weights.deadband)
                )
                slack_low = cp.pos(
                    (self.p_sp[k] - self.weights.deadband) - self.y[k + 1]
                )
                obj += self.weights.q_track * (
                    cp.square(slack_high) + cp.square(slack_low)
                )
            else:
                obj += self.weights.q_track * cp.square(self.y[k + 1] - self.p_sp[k])

            obj += self.weights.r_energy * cp.square(self.u[k])
            obj += self.weights.r_du * cp.square(du)

        self.problem = cp.Problem(cp.Minimize(obj), cons)

    def reset(self) -> None:
        self._u_prev = 0.0
        self._last_feature = None
        self._last_residual_pred = 0.0
        self._last_residual_obs = 0.0
        self._last_unexpected_occupancy_pred = 0.0
        self._last_unexpected_occupancy_obs = 0.0
        self._last_step_runtime_ms = 0.0
        self._last_forecast_runtime_ms = 0.0
        self._last_solver_runtime_ms = 0.0
        self._step = 0
        self._solver_fallback_count = 0
        self._solver_status_counts = {
            cp.OPTIMAL: 0,
            cp.OPTIMAL_INACCURATE: 0,
            "fallback": 0,
        }
        self._solver_attempt_status_counts = {}
        self._last_solver_status = "not_run"
        if self.residual_predictor is not None:
            self.residual_predictor.reset()

    def _feature(self, obs: Observation) -> np.ndarray:
        # Deliberate design tradeoff:
        # use previous control action to keep residual model independent
        # of current optimization variables and preserve convex MPC.
        feat = np.array(
            [
                obs.y,
                self._u_prev,               # u[k-1], matches "u" column in training data
                obs.disturbance.t_out,
                obs.disturbance.solar,
                obs.disturbance.occupancy,
                obs.setpoint,
                self._last_residual_obs,    # residual[k-1], matches "residual" column
            ],
            dtype=float,
        )
        assert len(feat) == FEATURE_COUNT, (
            f"Feature vector has {len(feat)} elements, LSTM expects {FEATURE_COUNT}. "
            f"Check FEATURE_COLUMNS in residual_lstm.py."
        )
        return feat

    def _residual_sequence(self, obs: Observation) -> np.ndarray:
        if self.residual_predictor is None:
            return np.zeros(self.horizon_steps, dtype=float)
        feat = self._feature(obs)
        seq = self.residual_predictor.forecast(self.horizon_steps, feat)
        return np.asarray(seq, dtype=float)

    def _residual_to_unexpected_occupancy(self, residual: float) -> float:
        if abs(self.model.b_occ) < 1e-8:
            return 0.0
        return float(residual / self.model.b_occ)

    def _record_solver_attempt_status(self, status: str) -> None:
        self._solver_attempt_status_counts[status] = (
            self._solver_attempt_status_counts.get(status, 0) + 1
        )

    def _solve(self, obs: Observation, residual_seq: np.ndarray) -> float:
        H = self.horizon_steps

        d_t_out = np.zeros(H)
        d_solar = np.zeros(H)
        d_occ = np.zeros(H)
        sp = np.zeros(H)

        if obs.future_disturbances is not None and len(obs.future_disturbances) > 0:
            for k in range(min(H, len(obs.future_disturbances))):
                d_t_out[k] = obs.future_disturbances[k].t_out
                d_solar[k] = obs.future_disturbances[k].solar
                d_occ[k] = obs.future_disturbances[k].occupancy
            # Fill the rest with the last available if horizon is longer
            for k in range(len(obs.future_disturbances), H):
                d_t_out[k] = d_t_out[k - 1]
                d_solar[k] = d_solar[k - 1]
                d_occ[k] = d_occ[k - 1]
        else:
            d_t_out.fill(obs.disturbance.t_out)
            d_solar.fill(obs.disturbance.solar)
            d_occ.fill(obs.disturbance.occupancy)

        if obs.future_setpoints is not None and len(obs.future_setpoints) > 0:
            for k in range(min(H, len(obs.future_setpoints))):
                sp[k] = obs.future_setpoints[k]
            for k in range(len(obs.future_setpoints), H):
                sp[k] = sp[k - 1]
        else:
            sp.fill(obs.setpoint)

        self.p_y0.value = obs.y
        self.p_u_prev.value = self._u_prev
        self.p_d_t_out.value = d_t_out
        self.p_d_solar.value = d_solar
        self.p_d_occ.value = d_occ
        self.p_sp.value = sp
        self.p_residual_seq.value = residual_seq

        # Solver cascade: OSQP is the primary choice (fast warm-start for QPs).
        # ECOS and SCS are fallbacks — slower but more robust to near-infeasibility.
        # Prefer the first OPTIMAL solution; accept OPTIMAL_INACCURATE only if
        # no OPTIMAL result was found across all three solvers.
        inaccurate_candidate: float | None = None
        for solver in (cp.OSQP, cp.ECOS, cp.SCS):
            try:
                self.problem.solve(solver=solver, warm_start=True, verbose=False)
            except Exception as e:
                self._record_solver_attempt_status(f"exception:{type(e).__name__}")
                logger.debug("Solver %s failed at step %d: %s", solver, self._step, e)
                continue
            status = str(self.problem.status or "unknown")
            self._record_solver_attempt_status(status)

            if status not in ALLOWED_SOLVER_STATUSES or self.u.value is None:
                logger.debug(
                    "Solver %s returned unusable status %s at step %d.",
                    solver,
                    status,
                    self._step,
                )
                continue

            u0 = float(np.clip(self.u.value[0], self.u_min, self.u_max))
            if status == cp.OPTIMAL:
                self._solver_status_counts[cp.OPTIMAL] += 1
                self._last_solver_status = cp.OPTIMAL
                return u0

            if inaccurate_candidate is None:
                inaccurate_candidate = u0
            logger.warning(
                "MPC step %d: solver %s returned %s, trying fallback solvers.",
                self._step,
                solver,
                status,
            )

        if inaccurate_candidate is not None:
            self._solver_status_counts[cp.OPTIMAL_INACCURATE] += 1
            self._last_solver_status = cp.OPTIMAL_INACCURATE
            logger.warning(
                "MPC step %d: accepting %s solution after no optimal solve was found.",
                self._step,
                cp.OPTIMAL_INACCURATE,
            )
            return inaccurate_candidate

        self._solver_status_counts["fallback"] += 1
        self._last_solver_status = "fallback"
        if self.u.value is not None:
            self.u.value = None
        self._solver_fallback_count += 1
        logger.warning(
            "MPC step %d: all solvers failed (total %d fallbacks). "
            "Returning u_prev=%.3f.",
            self._step,
            self._solver_fallback_count,
            self._u_prev,
        )
        return float(np.clip(self._u_prev, self.u_min, self.u_max))

    def act(self, obs: Observation) -> float:
        t0 = time.perf_counter()
        self._step += 1

        tf0 = time.perf_counter()
        residual_seq = self._residual_sequence(obs)
        self._last_forecast_runtime_ms = (time.perf_counter() - tf0) * 1000.0

        ts0 = time.perf_counter()
        u = self._solve(obs, residual_seq)
        self._last_solver_runtime_ms = (time.perf_counter() - ts0) * 1000.0

        self._last_feature = self._feature(obs)
        self._last_residual_pred = (
            float(residual_seq[0]) if len(residual_seq) > 0 else 0.0
        )
        self._last_unexpected_occupancy_pred = self._residual_to_unexpected_occupancy(
            self._last_residual_pred
        )
        self._u_prev = u

        self._last_step_runtime_ms = (time.perf_counter() - t0) * 1000.0
        return u

    def observe_transition(self, obs: Observation, u: float, y_next: float) -> None:
        # Compute the one-step model error after the plant has responded.
        # This residual feeds into the LSTM history so the predictor has
        # ground-truth correction signal for the next step.
        y_hat = self.model.predict_next(obs.y, u, obs.disturbance, residual=0.0)
        residual = float(y_next - y_hat)
        self._last_residual_obs = residual
        self._last_unexpected_occupancy_obs = self._residual_to_unexpected_occupancy(
            residual
        )

        if self.residual_predictor is not None and self._last_feature is not None:
            self.residual_predictor.update(self._last_feature, residual)


class HybridMPCController(MPCController):
    name = "hybrid_mpc"

    def __init__(
        self,
        model: LinearRoomModel,
        horizon_steps: int,
        u_min: float,
        u_max: float,
        du_max: float,
        residual_predictor,
        weights: MPCWeights | None = None,
    ) -> None:
        super().__init__(
            model=model,
            horizon_steps=horizon_steps,
            u_min=u_min,
            u_max=u_max,
            du_max=du_max,
            weights=weights,
            residual_predictor=residual_predictor,
        )
