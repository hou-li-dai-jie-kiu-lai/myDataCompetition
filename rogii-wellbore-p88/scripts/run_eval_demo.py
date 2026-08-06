"""Synthetic demo: metrics + P88 ridge blend without competition data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_share import fit_predict_p88_oof, rmse, well_rmse_table, warmup_delta


def main() -> None:
    rng = np.random.default_rng(42)
    n_wells, n_per = 40, 80
    wells = np.repeat([f"w{i:02d}" for i in range(n_wells)], n_per)
    n = len(wells)

    # simulate residual target and correlated channel predictions
    truth = rng.normal(0, 8.0, size=n)
    noise = rng.normal(0, 2.0, size=(n, 5))
    p20h = truth + noise[:, 0]
    p12f = truth + noise[:, 1]
    p50e = truth + noise[:, 2]
    hard = truth + noise[:, 3] + 1.0
    stock = truth + noise[:, 4] - 0.5
    p69 = 0.6 * p20h + 0.4 * p50e + rng.normal(0, 0.5, n)
    md = np.tile(np.linspace(0, 400, n_per), n_wells)

    X = np.column_stack([p20h, p12f, p50e, hard, stock, md, p69])
    y = truth

    res = fit_predict_p88_oof(X, y, wells, p20h=p20h, p69=p69)
    print("p20h", rmse(y, p20h))
    print("p69 ", rmse(y, p69))
    print("p81b", rmse(y, res.p81b))
    print("blend", rmse(y, res.blend), "w_mean", float(np.mean(res.blend_weights)))

    pp = warmup_delta(res.blend, md, tau=85.0)
    print("blend+warmup", rmse(y, pp))
    print(well_rmse_table(y, res.blend, wells).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
