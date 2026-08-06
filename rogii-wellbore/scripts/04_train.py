"""Build features + train topline OOF (requires ROGII_DATA_ROOT or --data)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii.dataset import build_dataset
from rogii.train import save_artifacts, train_topline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None, help="train/ directory parent or train dir")
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--n-particles", type=int, default=120)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--cat-iterations", type=int, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "topline_artifact.joblib")
    args = ap.parse_args()

    data = args.data
    if data is None:
        env = os.environ.get("ROGII_DATA_ROOT")
        if not env:
            raise SystemExit("Pass --data or set ROGII_DATA_ROOT")
        data = Path(env)
    data = Path(data)
    train_dir = data / "train" if (data / "train").is_dir() else data

    df = build_dataset(
        train_dir,
        n_seeds=args.n_seeds,
        n_particles=args.n_particles,
        require_label=True,
        max_wells=args.max_wells,
    )
    print("rows", len(df), "wells", df["well"].nunique(), flush=True)
    art = train_topline(df, n_splits=args.n_splits, cat_iterations=args.cat_iterations)
    save_artifacts(art, args.out)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
