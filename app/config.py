from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_WARMUP_MINUTES = 120
DEFAULT_LSTM_SEQ_LEN = 12
DEFAULT_LSTM_HIDDEN_SIZE = 24
DEFAULT_LSTM_EPOCHS = 20


class ConfigError(RuntimeError):
    pass


@dataclass
class AppConfig:
    room_id: str
    sampling_minutes: int
    horizon_minutes: int
    u_min: float
    u_max: float
    du_max: float
    day_setpoint: float
    night_setpoint: float
    plant_heating_gain: float
    plant_leak_coef: float
    plant_solar_coef: float
    plant_occ_coef: float
    mpc_heating_gain: float
    mpc_leak_coef: float
    mpc_solar_coef: float
    mpc_occ_coef: float
    warmup_minutes: int
    lstm_seq_len: int
    lstm_hidden_size: int
    lstm_epochs: int

    @property
    def horizon_steps(self) -> int:
        return self.horizon_minutes // self.sampling_minutes


def _get_required(section: dict, key: str):
    if key not in section:
        raise ConfigError(f"Missing key: {key}")
    return section[key]


def load_config(path: str) -> AppConfig:
    file_path = Path(path)
    if not file_path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("Top-level config must be a dictionary")

    metadata = raw.get("metadata", {})
    constraints = raw.get("constraints", {})
    comfort = raw.get("comfort", {})
    model = raw.get("model", {})
    experiment = raw.get("experiment", {})

    cfg = AppConfig(
        room_id=str(_get_required(metadata, "room_id")),
        sampling_minutes=int(_get_required(metadata, "sampling_minutes")),
        horizon_minutes=int(_get_required(metadata, "horizon_minutes")),
        u_min=float(_get_required(constraints, "u_min")),
        u_max=float(_get_required(constraints, "u_max")),
        du_max=float(_get_required(constraints, "du_max")),
        day_setpoint=float(comfort.get("day_setpoint", 22.0)),
        night_setpoint=float(comfort.get("night_setpoint", 20.5)),
        plant_heating_gain=float(model.get("plant_heating_gain", 2.8)),
        plant_leak_coef=float(model.get("plant_leak_coef", 0.15)),
        plant_solar_coef=float(model.get("plant_solar_coef", 0.003)),
        plant_occ_coef=float(model.get("plant_occ_coef", 0.07)),
        mpc_heating_gain=float(model.get("mpc_heating_gain", 1.8)),
        mpc_leak_coef=float(model.get("mpc_leak_coef", 0.08)),
        mpc_solar_coef=float(model.get("mpc_solar_coef", 0.0012)),
        mpc_occ_coef=float(model.get("mpc_occ_coef", 0.03)),
        warmup_minutes=int(
            experiment.get("warmup_minutes", DEFAULT_WARMUP_MINUTES)
        ),
        lstm_seq_len=int(experiment.get("lstm_seq_len", DEFAULT_LSTM_SEQ_LEN)),
        lstm_hidden_size=int(
            experiment.get("lstm_hidden_size", DEFAULT_LSTM_HIDDEN_SIZE)
        ),
        lstm_epochs=int(experiment.get("lstm_epochs", DEFAULT_LSTM_EPOCHS)),
    )

    # Basic checks kept explicit for readability.
    if cfg.sampling_minutes <= 0:
        raise ConfigError("sampling_minutes must be > 0")
    if cfg.horizon_minutes < cfg.sampling_minutes:
        raise ConfigError("horizon_minutes must be >= sampling_minutes")
    if cfg.horizon_minutes % cfg.sampling_minutes != 0:
        raise ConfigError("horizon_minutes must be divisible by sampling_minutes")
    if cfg.warmup_minutes < 0:
        raise ConfigError("experiment.warmup_minutes must be >= 0")
    if cfg.warmup_minutes % cfg.sampling_minutes != 0:
        raise ConfigError(
            "experiment.warmup_minutes must be divisible by sampling_minutes"
        )
    if not (-1.0 <= cfg.u_min < cfg.u_max <= 1.0):
        raise ConfigError("u_min/u_max must satisfy -1 <= u_min < u_max <= 1")
    if cfg.du_max <= 0.0:
        raise ConfigError("du_max must be > 0")
    if cfg.plant_heating_gain <= 0.0:
        raise ConfigError("model.plant_heating_gain must be > 0")
    if cfg.mpc_heating_gain <= 0.0:
        raise ConfigError("model.mpc_heating_gain must be > 0")
    if cfg.lstm_seq_len < 1:
        raise ConfigError("experiment.lstm_seq_len must be >= 1")
    if cfg.lstm_hidden_size < 1:
        raise ConfigError("experiment.lstm_hidden_size must be >= 1")
    if cfg.lstm_epochs < 1:
        raise ConfigError("experiment.lstm_epochs must be >= 1")

    return cfg
