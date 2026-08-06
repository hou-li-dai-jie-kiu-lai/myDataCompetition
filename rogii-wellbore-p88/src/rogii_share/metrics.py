from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error


def rmse(y_true, y_pred) -> float:
    return float(root_mean_squared_error(y_true, y_pred))


def well_rmse_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    wells: np.ndarray,
) -> pd.DataFrame:
    """Per-well RMSE plus global row."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    wells = np.asarray(wells)
    rows = []
    for w in np.unique(wells):
        m = wells == w
        rows.append(
            {
                "well": str(w),
                "n": int(m.sum()),
                "rmse": rmse(y_true[m], y_pred[m]),
                "mean_err": float(np.mean(y_pred[m] - y_true[m])),
            }
        )
    out = pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)
    return out
