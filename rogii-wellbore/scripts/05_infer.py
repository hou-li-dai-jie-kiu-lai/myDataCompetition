"""Infer test wells and write submission.csv."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii.config import N_PARTICLES, N_SEEDS
from rogii.infer import infer_topline, write_submission


def main() -> None:
    ap = argparse.ArgumentParser(description="Infer ROGII test wells (defaults from config.py).")
    ap.add_argument("--data", type=Path, default=None, help="data root containing test/")
    ap.add_argument("--artifact", type=Path, default=ROOT / "outputs" / "topline_artifact.joblib")
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    ap.add_argument("--n-particles", type=int, default=N_PARTICLES)
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument(
        "--fast",
        action="store_true",
        help="smoke PF settings (4 seeds / 40 particles); must match train --fast",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "submission.csv")
    args = ap.parse_args()

    if args.fast:
        args.n_seeds = 4
        args.n_particles = 40
        print("[fast] smoke PF overrides", flush=True)

    data = args.data
    if data is None:
        env = os.environ.get("ROGII_DATA_ROOT")
        if not env:
            raise SystemExit("Pass --data or set ROGII_DATA_ROOT")
        data = Path(env)
    data = Path(data)
    test_dir = data / "test" if (data / "test").is_dir() else data
    sample = data / "sample_submission.csv"
    sample = sample if sample.exists() else None

    print(f"[cfg] seeds={args.n_seeds} particles={args.n_particles}", flush=True)
    pred = infer_topline(
        test_dir,
        args.artifact,
        n_seeds=args.n_seeds,
        n_particles=args.n_particles,
        max_wells=args.max_wells,
    )
    path = write_submission(pred, sample, args.out)
    print(
        "wrote",
        path,
        "rows",
        len(pred),
        "tvt_mean",
        float(pred["tvt"].mean()),
        flush=True,
    )


if __name__ == "__main__":
    main()
