import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import load_config
from app.scenario_utils import EXPECTED_SCENARIO_TYPES
from app.simulation import ScenarioSpec, make_scenario


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate reproducible multi-day scenario CSV files for evaluation"
    )
    parser.add_argument("--config", default="configs/small_office.yaml")
    parser.add_argument("--out-dir", default="data/scenarios")
    parser.add_argument("--variants", type=int, default=5)
    parser.add_argument("--steps", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir) / cfg.room_id
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_count = 0
    for idx in range(args.variants):
        for s_idx, s_type in enumerate(EXPECTED_SCENARIO_TYPES):
            df = make_scenario(
                ScenarioSpec(
                    name=s_type,
                    steps=args.steps,
                    sampling_minutes=cfg.sampling_minutes,
                ),
                day_setpoint=cfg.day_setpoint,
                night_setpoint=cfg.night_setpoint,
                seed=args.seed + s_idx * 1000 + idx,
            )

            out_path = out_dir / f"{s_type}_v{idx:02d}.csv"
            df.to_csv(out_path, index=False)
            print(f"Saved: {out_path}")
            generated_count += 1

    print(f"\nDone. Generated {generated_count} files in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
