from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .cv import iter_group_kfold
from .metrics import rmse


DEFAULT_ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0)


@dataclass
class P88OOFResult:
    p81b: np.ndarray
    blend: np.ndarray
    blend_weights: list[float]
    fold_va_rmse: list[float]


def fit_predict_p88_oof(
    X: np.ndarray,
    y: np.ndarray,
    wells: np.ndarray,
    p20h: np.ndarray,
    p69: np.ndarray,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    n_splits: int = 5,
    blend_grid: int = 41,
) -> P88OOFResult:
    """
    Reproduce the P88 core stacking step on aligned OOF matrices.

    X: shape [n, f] — typically columns [p20h, p12f, p50e, hard, stock, md, p69]
    y: residual target (TVT - last_known) used in training
    p20h / p69: base OOF predictions in the same residual space
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p20h = np.asarray(p20h, dtype=np.float64)
    p69 = np.asarray(p69, dtype=np.float64)
    wells = np.asarray(wells)

    p81b = np.zeros(len(y), dtype=np.float32)
    fold_va_rmse: list[float] = []
    for fold, tr, va in iter_group_kfold(wells, n_splits=n_splits):
        acc = np.zeros(len(va), dtype=np.float64)
        for alpha in alphas:
            sc = StandardScaler()
            Xt = sc.fit_transform(X[tr])
            Xv = sc.transform(X[va])
            reg = Ridge(alpha=alpha).fit(Xt, (y - p20h)[tr])
            acc += p20h[va] + reg.predict(Xv)
        pred = (acc / len(alphas)).astype(np.float32)
        p81b[va] = pred
        fold_va_rmse.append(rmse(y[va], pred))
        print(f"fold{fold} va_rmse={fold_va_rmse[-1]:.4f}")

    blend = np.zeros(len(y), dtype=np.float32)
    weights: list[float] = []
    for _, tr, va in iter_group_kfold(wells, n_splits=n_splits):
        best_w, best_r = 1.0, 1e9
        for w in np.linspace(0.0, 1.0, blend_grid):
            r = rmse(y[tr], w * p69[tr] + (1.0 - w) * p81b[tr])
            if r < best_r:
                best_r, best_w = r, float(w)
        blend[va] = (best_w * p69[va] + (1.0 - best_w) * p81b[va]).astype(np.float32)
        weights.append(best_w)

    return P88OOFResult(
        p81b=p81b,
        blend=blend,
        blend_weights=weights,
        fold_va_rmse=fold_va_rmse,
    )
