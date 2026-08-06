from __future__ import annotations

import numpy as np
import pandas as pd


def robfit(s: np.ndarray, y: np.ndarray, deg: int = 4) -> np.ndarray:
    if len(s) < deg + 2:
        return y.copy()
    c = np.polyfit(s, y, deg)
    for _ in range(4):
        r = y - np.polyval(c, s)
        sc = np.median(np.abs(r)) * 1.4826 + 1e-6
        c = np.polyfit(s, y, deg, w=1.0 / (1.0 + (r / (2.0 * sc)) ** 2))
    return np.polyval(c, s)


def warmup_delta(delta: np.ndarray, md_since: np.ndarray, tau: float = 85.0) -> np.ndarray:  # final PP_TAU
    w = 1.0 - np.exp(-np.maximum(md_since, 0.0) / max(tau, 1e-6))
    return (w * np.asarray(delta, dtype=np.float64)).astype(np.float32)


def project_well_absolute(
    md: np.ndarray,
    z: np.ndarray,
    tvt_abs: np.ndarray,
    md_anchor: float,
    z_anchor: float,
    tvt_anchor: float,
    md_end: float,
    blend: float = 0.7,
    deg: int = 4,
) -> np.ndarray:
    """Project absolute TVT via U = TVT + Z continuity."""
    anchor_u = float(tvt_anchor) + float(z_anchor)
    s = (md - md_anchor) / max(md_end - md_anchor, 1e-6)
    fit = robfit(s, (tvt_abs + z) - anchor_u, deg=deg)
    tvt_fit = (anchor_u + fit) - z
    out = (1.0 - blend) * tvt_abs + blend * tvt_fit
    if not np.all(np.isfinite(out)):
        return np.asarray(tvt_abs, dtype=np.float32)
    return out.astype(np.float32)


def apply_pp_delta(
    delta: np.ndarray,
    wells: np.ndarray,
    md_since: np.ndarray,
    well_frames: dict[str, pd.DataFrame],
    ids: np.ndarray | None = None,
    tau: float = 85.0,
    blend: float = 0.7,
    do_warmup: bool = True,
    do_proj: bool = True,
) -> np.ndarray:
    """
    Warm-up + optional per-well projection on residual predictions.

    well_frames[well] needs MD, Z, TVT_input.
    """
    out = np.asarray(delta, dtype=np.float32).copy()
    if do_warmup:
        out = warmup_delta(out, md_since, tau=tau)
    if not do_proj:
        return out

    wells = np.asarray(wells)
    for w in np.unique(wells):
        m = wells == w
        hw = well_frames.get(str(w))
        if hw is None or "TVT_input" not in hw.columns:
            continue
        kn = hw[hw["TVT_input"].notna()]
        if len(kn) < 5:
            continue
        last = kn.iloc[-1]
        last_tvt = float(last["TVT_input"])
        if ids is not None:
            row_idx = np.array([int(str(i).rsplit("_", 1)[1]) for i in ids[m]], dtype=np.int32)
        else:
            row_idx = np.arange(int(m.sum()), dtype=np.int32)
            row_idx = np.clip(row_idx, 0, len(hw) - 1)
        md = hw["MD"].to_numpy(float)[row_idx]
        z = hw["Z"].to_numpy(float)[row_idx]
        tvt_abs = last_tvt + out[m]
        tvt2 = project_well_absolute(
            md=md,
            z=z,
            tvt_abs=tvt_abs,
            md_anchor=float(last["MD"]),
            z_anchor=float(last["Z"]),
            tvt_anchor=last_tvt,
            md_end=float(hw["MD"].iloc[-1]),
            blend=blend,
        )
        out[m] = (tvt2 - last_tvt).astype(np.float32)
    return out
