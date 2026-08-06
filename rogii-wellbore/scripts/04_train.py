"""Build features + train topline OOF (requires ROGII_DATA_ROOT or --data)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii.config import N_PARTICLES, N_SEEDS, N_SPLITS, PP_BLEND, PP_TAU
from rogii.dataset import build_dataset
from rogii.train import save_artifacts, train_topline


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train final-recipe topline (defaults = private-LB submit settings)."
    )
    ap.add_argument("--data", type=Path, default=None, help="data root or train/ directory")
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS, help=f"default {N_SEEDS} (final)")
    ap.add_argument(
        "--n-particles", type=int, default=N_PARTICLES, help=f"default {N_PARTICLES} (final)"
    )
    ap.add_argument("--n-splits", type=int, default=N_SPLITS)
    ap.add_argument("--cat-iterations", type=int, default=None, help="override Cat iterations")
    ap.add_argument("--gpu", action="store_true", help="train CatBoost on GPU if available")
    ap.add_argument(
        "--fast",
        action="store_true",
        help="smoke settings (not final): 4 seeds, 40 particles, 150 cat iters",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "topline_artifact.joblib")
    args = ap.parse_args()

    if args.fast:
        args.n_seeds = 4
        args.n_particles = 40
        if args.cat_iterations is None:
            args.cat_iterations = 150
        if args.max_wells is None:
            args.max_wells = 8
        print("[fast] smoke overrides enabled (not final-score settings)", flush=True)

    data = args.data
    if data is None:
        env = os.environ.get("ROGII_DATA_ROOT")
        if not env:
            raise SystemExit("Pass --data or set ROGII_DATA_ROOT")
        data = Path(env)
    data = Path(data)
    train_dir = data / "train" if (data / "train").is_dir() else data

    print(
        f"[cfg] seeds={args.n_seeds} particles={args.n_particles} splits={args.n_splits} "
        f"pp=({PP_TAU},{PP_BLEND})",
        flush=True,
    )
    df = build_dataset(
        train_dir,
        n_seeds=args.n_seeds,
        n_particles=args.n_particles,
        require_label=True,
        max_wells=args.max_wells,
    )
    print("rows", len(df), "wells", df["well"].nunique(), flush=True)
    art = train_topline(
        df,
        n_splits=args.n_splits,
        pp_tau=PP_TAU,
        pp_blend=PP_BLEND,
        cat_iterations=args.cat_iterations,
        use_gpu=args.gpu,
    )
    save_artifacts(art, args.out)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
