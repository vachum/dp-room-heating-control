import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import ConfigError, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate room YAML config")
    parser.add_argument("config_path", nargs="?")
    parser.add_argument("--config", dest="config_opt")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    config_path = args.config_opt or args.config_path
    if config_path is None:
        parser.error("Provide config path as positional argument or --config <path>")

    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        f"Config OK: room_id={cfg.room_id}, sampling={cfg.sampling_minutes}min, horizon={cfg.horizon_minutes}min"
    )
    print(f"Control bounds: u=[{cfg.u_min}, {cfg.u_max}], du_max={cfg.du_max}")
    print(f"Heating gains: plant={cfg.plant_heating_gain}, mpc={cfg.mpc_heating_gain}")

    if args.json:
        print(json.dumps(cfg.__dict__, indent=2, ensure_ascii=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
