from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .config import BLEND_GRID, STACK_ALPHAS
from .cv import iter_group_kfold
from .metrics import rmse

DEFAULT_ALPHAS = STACK_ALPHAS


@dataclass
class StackOOFResult:
    ridge_bag: np.ndarray
    blend: np.ndarray
    blend_weights: list[float]
    fold_va_rmse: list[float]

    def summary(self, y: np.ndarray, main: np.ndarray, anchor: np.ndarray) -> dict:
        return {
            "rmse_main": rmse(y, main),
            "rmse_anchor": rmse(y, anchor),
            "rmse_ridge_bag": rmse(y, self.ridge_bag),
            "rmse_blend": rmse(y, self.blend),
            "blend_w_mean": float(np.mean(self.blend_weights)),
            "blend_weights": list(map(float, self.blend_weights)),
            "fold_va_rmse": list(map(float, self.fold_va_rmse)),
        }


def fit_predict_stack_oof(
    X: np.ndarray,
    y: np.ndarray,
    wells: np.ndarray,
    main: np.ndarray,
    anchor: np.ndarray,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    n_splits: int = 5,
    blend_grid: int = BLEND_GRID,
) -> StackOOFResult:
    """
    Residual stacking:

    1) Multi-alpha Ridge bag on residual (y - main), added back to main
    2) Fold-wise blend with anchor: w * anchor + (1-w) * ridge_bag
       (w chosen on fold-train wells only)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    main = np.asarray(main, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    wells = np.asarray(wells)

    ridge_bag = np.zeros(len(y), dtype=np.float32)
    fold_va_rmse: list[float] = []

    for fold, tr, va in iter_group_kfold(wells, n_splits=n_splits):
        acc = np.zeros(len(va), dtype=np.float64)
        for alpha in alphas:
            sc = StandardScaler()
            Xt = sc.fit_transform(X[tr])
            Xv = sc.transform(X[va])
            reg = Ridge(alpha=alpha).fit(Xt, (y - main)[tr])
            acc += main[va] + reg.predict(Xv)
        pred = (acc / len(alphas)).astype(np.float32)
        ridge_bag[va] = pred
        fold_va_rmse.append(rmse(y[va], pred))
        print(f"[stack] fold{fold} va_rmse={fold_va_rmse[-1]:.4f}")

    blend = np.zeros(len(y), dtype=np.float32)
    weights: list[float] = []
    for _, tr, va in iter_group_kfold(wells, n_splits=n_splits):
        best_w, best_r = 1.0, 1e9
        for w in np.linspace(0.0, 1.0, blend_grid):
            r = rmse(y[tr], w * anchor[tr] + (1.0 - w) * ridge_bag[tr])
            if r < best_r:
                best_r, best_w = r, float(w)
        blend[va] = (best_w * anchor[va] + (1.0 - best_w) * ridge_bag[va]).astype(np.float32)
        weights.append(best_w)

    return StackOOFResult(
        ridge_bag=ridge_bag,
        blend=blend,
        blend_weights=weights,
        fold_va_rmse=fold_va_rmse,
    )
