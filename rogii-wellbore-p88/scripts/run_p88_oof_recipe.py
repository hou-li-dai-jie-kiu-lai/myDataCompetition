"""
Run P88 ridge blend on user-supplied OOF arrays.

Expected NPZ keys (all length n, residual space):
  y, wells, md, p20h, p12f, p50e, hard, stock, p69

Example:
  python scripts/run_p88_oof_recipe.py --npz path/to/channels.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_share import fit_predict_p88_oof, rmse, well_rmse_table, warmup_delta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True, help="NPZ with OOF channels")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "p88_oof_result")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    required = ["y", "wells", "md", "p20h", "p12f", "p50e", "hard", "stock", "p69"]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise SystemExit(f"NPZ missing keys: {missing}")

    X = np.column_stack([z[k] for k in ["p20h", "p12f", "p50e", "hard", "stock", "md", "p69"]])
    y = z["y"]
    wells = z["wells"]
    res = fit_predict_p88_oof(X, y, wells, p20h=z["p20h"], p69=z["p69"])
    pp = warmup_delta(res.blend, z["md"], tau=85.0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out.with_suffix(".npz"),
        p81b=res.p81b,
        blend=res.blend,
        blend_warmup=pp,
        blend_weights=np.asarray(res.blend_weights, dtype=np.float64),
    )
    summary = {
        "rmse_p20h": rmse(y, z["p20h"]),
        "rmse_p69": rmse(y, z["p69"]),
        "rmse_p81b": rmse(y, res.p81b),
        "rmse_blend": rmse(y, res.blend),
        "rmse_blend_warmup": rmse(y, pp),
        "blend_w_mean": float(np.mean(res.blend_weights)),
        "blend_weights": res.blend_weights,
        "fold_va_rmse": res.fold_va_rmse,
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    well_rmse_table(y, res.blend, wells).to_csv(args.out.with_name(args.out.name + "_wells.csv"), index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
