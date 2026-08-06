"""Build row-level feature tables from raw well CSVs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .features_pf import TRACK_KEYS, pf_track


def _well_id_from_horizontal(path: Path) -> str:
    return path.name.split("__", 1)[0]


def list_wells(data_dir: Path) -> list[str]:
    data_dir = Path(data_dir)
    wells = []
    for p in sorted(data_dir.glob("*__horizontal_well.csv")):
        wid = _well_id_from_horizontal(p)
        if (data_dir / f"{wid}__typewell.csv").exists():
            wells.append(wid)
    return wells


def build_well_frame(
    data_dir: Path,
    wid: str,
    n_seeds: int = 8,
    n_particles: int = 120,
    require_label: bool = True,
) -> pd.DataFrame | None:
    """
    One row per lateral sample (TVT_input is NaN).
    Residual target y = TVT - last_known when label exists.
    """
    data_dir = Path(data_dir)
    hw = pd.read_csv(data_dir / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(data_dir / f"{wid}__typewell.csv")
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 5:
        return None
    last = float(kn.iloc[-1]["TVT_input"])
    md_last = float(kn.iloc[-1]["MD"])

    raw = pf_track(hw, tw, dual=False, use_affine=False, n_seeds=n_seeds, n_particles=n_particles)
    aff = pf_track(hw, tw, dual=False, use_affine=True, n_seeds=n_seeds, n_particles=n_particles)
    dual = pf_track(hw, tw, dual=True, use_affine=False, n_seeds=n_seeds, n_particles=n_particles)
    if raw is None:
        return None

    idx = raw["idx"]
    rows = []
    for j, i in enumerate(idx):
        r = hw.loc[int(i)]
        tvt = r["TVT"] if "TVT" in hw.columns else np.nan
        if require_label and not np.isfinite(tvt):
            continue
        gr = r["GR"] if "GR" in hw.columns else np.nan
        item = {
            "well": wid,
            "row_idx": int(i),
            "id": f"{wid}_{int(i)}",
            "md": float(r["MD"]),
            "z": float(r["Z"]),
            "x": float(r["X"]) if "X" in hw.columns else 0.0,
            "y": float(r["Y"]) if "Y" in hw.columns else 0.0,
            "gr": float(gr) if np.isfinite(gr) else 0.0,
            "md_since": max(float(r["MD"]) - md_last, 0.0),
            "last_known": last,
            "frac": float(j + 1) / max(len(idx), 1),
        }
        if np.isfinite(tvt):
            item["tvt"] = float(tvt)
            item["y"] = float(tvt) - last
        for k in TRACK_KEYS:
            item[f"trk_r_{k}"] = float(raw["deltas"][k][j])
            if aff is not None:
                item[f"trk_a_{k}"] = float(aff["deltas"][k][j])
            else:
                item[f"trk_a_{k}"] = item[f"trk_r_{k}"]
            if dual is not None:
                item[f"trk_d_{k}"] = float(dual["deltas"][k][j])
            else:
                item[f"trk_d_{k}"] = item[f"trk_r_{k}"]
        item["stock"] = item["trk_r_ms_mean"]
        item["pf_dual_gap"] = item["trk_d_ms_mean"] - item["trk_r_ms_mean"]
        item["pf_aff_gap"] = item["trk_a_ms_mean"] - item["trk_r_ms_mean"]
        rows.append(item)
    if not rows:
        return None
    return pd.DataFrame(rows)


def build_dataset(
    data_dir: Path,
    wells: list[str] | None = None,
    n_seeds: int = 8,
    n_particles: int = 120,
    require_label: bool = True,
    max_wells: int | None = None,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    if wells is None:
        wells = list_wells(data_dir)
    if max_wells is not None:
        wells = wells[:max_wells]
    frames = []
    for wid in tqdm(wells, desc="features"):
        fr = build_well_frame(
            data_dir,
            wid,
            n_seeds=n_seeds,
            n_particles=n_particles,
            require_label=require_label,
        )
        if fr is not None and len(fr):
            frames.append(fr)
    if not frames:
        raise RuntimeError(f"no usable wells under {data_dir}")
    return pd.concat(frames, ignore_index=True)


def default_track_cols() -> list[str]:
    cols = []
    for tag in ("r", "a", "d"):
        for k in TRACK_KEYS:
            cols.append(f"trk_{tag}_{k}")
    return cols


def default_cat_cols(track_cols: list[str] | None = None) -> list[str]:
    base = [
        "md",
        "z",
        "x",
        "y",
        "gr",
        "md_since",
        "frac",
        "stock",
        "pf_dual_gap",
        "pf_aff_gap",
    ]
    track_cols = track_cols or default_track_cols()
    return base + track_cols
