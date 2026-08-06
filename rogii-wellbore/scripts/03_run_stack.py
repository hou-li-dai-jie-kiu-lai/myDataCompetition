"""
Run residual stacking on user-provided OOF channels.

NPZ keys (aligned length n):
  y, wells, md, main, alt_a, alt_b, hard, stock, anchor

Example:
  python scripts/03_run_stack.py --npz outputs/channels.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii import channels_from_npz, run_topline_stack, well_rmse_table


def main() -> None:
    ap = argparse.ArgumentParser(description="Stack OOF channels from an NPZ file.")
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "stack_out")
    ap.add_argument("--tau", type=float, default=85.0)
    args = ap.parse_args()

    channels = channels_from_npz(args.npz)
    res = run_topline_stack(channels, tau=args.tau)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out.with_suffix(".npz"),
        ridge_bag=res.stack.ridge_bag,
        blend=res.stack.blend,
        blend_warmup=res.blend_warmup,
        blend_weights=np.asarray(res.stack.blend_weights, dtype=np.float64),
    )
    metrics = {k: v for k, v in res.metrics.items() if k != "well_rmse_head"}
    args.out.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    well_rmse_table(channels["y"], res.stack.blend, channels["wells"]).to_csv(
        args.out.with_name(args.out.name + "_wells.csv"), index=False
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
