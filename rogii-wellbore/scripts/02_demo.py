"""Synthetic end-to-end smoke test (no competition data)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii import run_topline_stack, well_rmse_table


def make_synthetic(n_wells: int = 40, n_per: int = 80, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    wells = np.repeat([f"w{i:02d}" for i in range(n_wells)], n_per)
    n = len(wells)
    truth = rng.normal(0, 8.0, size=n)
    noise = rng.normal(0, 2.0, size=(n, 5))
    main = truth + noise[:, 0]
    alt_a = truth + noise[:, 1]
    alt_b = truth + noise[:, 2]
    hard = truth + noise[:, 3] + 1.0
    stock = truth + noise[:, 4] - 0.5
    anchor = 0.55 * main + 0.45 * alt_b + rng.normal(0, 0.4, n)
    md = np.tile(np.linspace(0, 400, n_per), n_wells)
    return {
        "y": truth,
        "wells": wells,
        "md": md,
        "main": main,
        "alt_a": alt_a,
        "alt_b": alt_b,
        "hard": hard,
        "stock": stock,
        "anchor": anchor,
    }


def main() -> None:
    ch = make_synthetic()
    res = run_topline_stack(ch)
    print(json.dumps({k: v for k, v in res.metrics.items() if k != "well_rmse_head"}, indent=2))
    print(well_rmse_table(ch["y"], res.stack.blend, ch["wells"]).head(5).to_string(index=False))
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (out / "demo_metrics.json").write_text(
        json.dumps(res.metrics, indent=2, default=float), encoding="utf-8"
    )
    print("ok ->", out / "demo_metrics.json")


if __name__ == "__main__":
    main()
