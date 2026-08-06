"""Inference: average fold models -> stack -> PP -> submission.csv."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .dataset import build_dataset
from .features_pf import track_matrix
from .postprocess import apply_pp_delta, warmup_delta


def _predict_fold(df: pd.DataFrame, fold: dict, track_cols: list[str], cat_cols: list[str], s2_cols: list[str]):
    M = track_matrix(df, track_cols)
    hard = fold["hard_ridge"].predict(fold["hard_scaler"].transform(M)).astype(np.float32)
    X = df[cat_cols].copy()
    X["hard"] = hard
    feat_cols = fold["feat_cols"]
    main = fold["main_cat"].predict(X[feat_cols]).astype(np.float32)
    alt = fold["alt_cat"].predict(X[fold["alt_cols"]]).astype(np.float32)
    stock = df["stock"].to_numpy(np.float32)
    md = df["md_since"].to_numpy(np.float32)
    # stage2 features (no expanding on test across wells — per-well cum would need order; use zeros/simple)
    ae = (main - hard).astype(np.float32)
    s2 = pd.DataFrame(
        {
            "main": main,
            "hard": hard,
            "stock": stock,
            "ae": ae,
            "abs_ae": np.abs(ae),
            "md": md,
            "em": ae,  # fallback
            "es": np.abs(ae),
        }
    )
    anchor = (main + fold["anchor_cat"].predict(s2[s2_cols])).astype(np.float32)

    X_stack = np.column_stack([main, alt, hard, hard, stock, md, anchor])
    acc = np.zeros(len(df), np.float64)
    for pack in fold["stack_packs"]:
        Xs = pack["scaler"].transform(X_stack)
        acc += main + pack["ridge"].predict(Xs)
    ridge_bag = (acc / max(len(fold["stack_packs"]), 1)).astype(np.float32)
    w = float(fold.get("blend_w", 0.5))
    blend = (w * anchor + (1.0 - w) * ridge_bag).astype(np.float32)
    return blend


def infer_topline(
    data_dir: Path,
    artifact_path: Path,
    n_seeds: int = 8,
    n_particles: int = 120,
    max_wells: int | None = None,
) -> pd.DataFrame:
    art = joblib.load(artifact_path)
    df = build_dataset(
        data_dir,
        n_seeds=n_seeds,
        n_particles=n_particles,
        require_label=False,
        max_wells=max_wells,
    )
    preds = []
    for fold in art["fold_models"]:
        preds.append(
            _predict_fold(df, fold, art["track_cols"], art["cat_cols"], art["s2_cols"])
        )
    delta = np.mean(np.stack(preds, 0), axis=0).astype(np.float32)
    tau = float(art.get("pp", {}).get("tau", 85.0))
    blend_pp = float(art.get("pp", {}).get("blend", 0.7))
    delta = warmup_delta(delta, df["md_since"].to_numpy(np.float32), tau=tau)

    # absolute TVT + geometric projection
    well_frames = {}
    data_dir = Path(data_dir)
    for w in df["well"].astype(str).unique():
        well_frames[w] = pd.read_csv(data_dir / f"{w}__horizontal_well.csv")
    delta = apply_pp_delta(
        delta,
        df["well"].astype(str).to_numpy(),
        df["md_since"].to_numpy(np.float32),
        well_frames,
        ids=df["id"].astype(str).to_numpy(),
        tau=tau,
        blend=blend_pp,
        do_warmup=False,  # already warmed
        do_proj=True,
    )
    tvt = df["last_known"].to_numpy(np.float64) + delta
    return pd.DataFrame({"id": df["id"].astype(str), "tvt": tvt.astype(float)})


def write_submission(pred: pd.DataFrame, sample_csv: Path | None, out_csv: Path) -> Path:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if sample_csv is not None and Path(sample_csv).exists():
        sample = pd.read_csv(sample_csv)
        merged = sample[["id"]].merge(pred, on="id", how="left")
        if merged["tvt"].isna().any():
            # fill missing ids with pred mean / 0
            merged["tvt"] = merged["tvt"].fillna(float(pred["tvt"].mean()))
        merged.to_csv(out_csv, index=False)
    else:
        pred.to_csv(out_csv, index=False)
    return out_csv
