from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import rmse, well_rmse_table
from .postprocess import warmup_delta
from .stack import fit_predict_stack_oof, StackOOFResult


CHANNEL_ORDER = ("main", "alt_a", "alt_b", "hard", "stock", "md", "anchor")


@dataclass
class ToplineResult:
    stack: StackOOFResult
    blend_warmup: np.ndarray
    metrics: dict


def build_design_matrix(channels: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """channels must include CHANNEL_ORDER keys (+ y, wells)."""
    missing = [k for k in CHANNEL_ORDER if k not in channels]
    if missing:
        raise KeyError(f"missing channels: {missing}")
    X = np.column_stack([np.asarray(channels[k], dtype=np.float64) for k in CHANNEL_ORDER])
    y = np.asarray(channels["y"], dtype=np.float64)
    wells = np.asarray(channels["wells"])
    return X, y, wells


def run_topline_stack(
    channels: dict[str, np.ndarray],
    tau: float = 85.0,
    n_splits: int = 5,
) -> ToplineResult:
    """
    Full topline after base OOFs exist:
      design matrix -> ridge bag + anchor blend -> warm-up PP
    """
    X, y, wells = build_design_matrix(channels)
    main = np.asarray(channels["main"], dtype=np.float64)
    anchor = np.asarray(channels["anchor"], dtype=np.float64)
    md = np.asarray(channels["md"], dtype=np.float64)

    stack = fit_predict_stack_oof(
        X, y, wells, main=main, anchor=anchor, n_splits=n_splits
    )
    blend_wu = warmup_delta(stack.blend, md, tau=tau)
    metrics = stack.summary(y, main, anchor)
    metrics["rmse_blend_warmup"] = rmse(y, blend_wu)
    metrics["well_rmse_head"] = well_rmse_table(y, stack.blend, wells).head(10).to_dict(
        orient="records"
    )
    return ToplineResult(stack=stack, blend_warmup=blend_wu, metrics=metrics)


def channels_from_npz(path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}
