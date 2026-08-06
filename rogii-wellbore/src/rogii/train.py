"""Train topline aligned with final private-LB recipe defaults."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .config import (
    ANCHOR_SMAX,
    CAT_ALT,
    CAT_ANCHOR,
    CAT_MAIN,
    HARD_RIDGE_ALPHA,
    N_SPLITS,
    PP_BLEND,
    PP_TAU,
    STACK_ALPHAS,
)
from .cv import iter_group_kfold
from .dataset import default_cat_cols, default_track_cols
from .features_pf import track_matrix
from .metrics import rmse
from .postprocess import warmup_delta
from .stack import fit_predict_stack_oof


@dataclass
class TrainArtifacts:
    track_cols: list[str]
    cat_cols: list[str]
    s2_cols: list[str]
    fold_models: list[dict]
    oof: dict[str, np.ndarray]
    stack_summary: dict
    pp: dict


def _cat_cfg(base: dict, iterations: int | None = None, use_gpu: bool = False) -> dict:
    cfg = dict(base)
    if iterations is not None:
        cfg["iterations"] = int(iterations)
    if use_gpu:
        cfg["task_type"] = "GPU"
        cfg["devices"] = "0"
    else:
        cfg["task_type"] = "CPU"
        cfg.pop("devices", None)
    return cfg


def _fit_cat(cfg: dict, Xtr, ytr, Xva, yva) -> CatBoostRegressor:
    try:
        m = CatBoostRegressor(**cfg)
        m.fit(Pool(Xtr, ytr), eval_set=Pool(Xva, yva), use_best_model=True)
        return m
    except Exception as e:
        if cfg.get("task_type") == "GPU":
            print("  gpu fail → CPU", type(e).__name__, flush=True)
            cfg = dict(cfg)
            cfg["task_type"] = "CPU"
            cfg.pop("devices", None)
            m = CatBoostRegressor(**cfg)
            m.fit(Pool(Xtr, ytr), eval_set=Pool(Xva, yva), use_best_model=True)
            return m
        raise


def _expanding_stats(wells: np.ndarray, v: np.ndarray):
    out_m = np.zeros(len(v), np.float32)
    out_s = np.zeros(len(v), np.float32)
    tmp = pd.DataFrame({"well": wells, "v": v})
    for _, sub in tmp.groupby("well", sort=False):
        idx = sub.index.to_numpy()
        x = sub["v"].to_numpy(np.float64)
        cs = np.cumsum(x)
        cs2 = np.cumsum(x * x)
        n = np.arange(1, len(x) + 1, dtype=np.float64)
        m = cs / n
        s = np.sqrt(np.maximum(cs2 / n - m * m, 0.0))
        out_m[idx] = m.astype(np.float32)
        out_s[idx] = s.astype(np.float32)
    return out_m, out_s


def train_topline(
    df: pd.DataFrame,
    n_splits: int = N_SPLITS,
    pp_tau: float = PP_TAU,
    pp_blend: float = PP_BLEND,
    cat_iterations: int | None = None,
    use_gpu: bool = False,
) -> TrainArtifacts:
    y = df["y"].to_numpy(np.float32)
    wells = df["well"].astype(str).to_numpy()
    md = df["md_since"].to_numpy(np.float32)
    track_cols = default_track_cols()
    cat_cols = [c for c in default_cat_cols(track_cols) if c in df.columns]
    M_all = track_matrix(df, track_cols)
    stock = df["stock"].to_numpy(np.float32)

    hard_oof = np.zeros(len(df), np.float32)
    main_oof = np.zeros(len(df), np.float32)
    alt_oof = np.zeros(len(df), np.float32)
    fold_models: list[dict] = []

    main_cfg = _cat_cfg(CAT_MAIN, cat_iterations, use_gpu=use_gpu)
    alt_cfg = _cat_cfg(CAT_ALT, cat_iterations, use_gpu=use_gpu)
    anchor_cfg = _cat_cfg(CAT_ANCHOR, cat_iterations, use_gpu=use_gpu)

    for fold, tr, va in iter_group_kfold(wells, n_splits=n_splits):
        # Hard: Ridge(positive) on PF tracks (P20H)
        hard_model = Ridge(HARD_RIDGE_ALPHA, positive=True).fit(M_all[tr], y[tr])
        hard_tr = hard_model.predict(M_all[tr]).astype(np.float32)
        hard_va = hard_model.predict(M_all[va]).astype(np.float32)
        hard_oof[va] = hard_va

        Xtr = df.iloc[tr][cat_cols].copy()
        Xva = df.iloc[va][cat_cols].copy()
        Xtr["hard"] = hard_tr
        Xva["hard"] = hard_va
        Xtr["pf_ens_delta"] = hard_tr
        Xva["pf_ens_delta"] = hard_va
        Xtr["pf_ridge_hard"] = hard_tr
        Xva["pf_ridge_hard"] = hard_va
        feat_cols = list(Xtr.columns)

        # Main: residual on hard (P20H)
        y_main_tr = (y[tr] - hard_tr).astype(np.float32)
        y_main_va = (y[va] - hard_va).astype(np.float32)
        main = _fit_cat(main_cfg, Xtr[feat_cols], y_main_tr, Xva[feat_cols], y_main_va)
        main_oof[va] = (hard_va + main.predict(Xva[feat_cols]).astype(np.float32)).astype(np.float32)

        # Alt: second DNA, drop dual tracks, residual on hard
        alt_cols = [c for c in feat_cols if not str(c).startswith("trk_d_")]
        alt = _fit_cat(alt_cfg, Xtr[alt_cols], y_main_tr, Xva[alt_cols], y_main_va)
        alt_oof[va] = (hard_va + alt.predict(Xva[alt_cols]).astype(np.float32)).astype(np.float32)

        fold_models.append(
            {
                "hard_ridge": hard_model,
                "main_cat": main,
                "alt_cat": alt,
                "alt_cols": alt_cols,
                "feat_cols": feat_cols,
            }
        )
        print(
            f"[base] fold{fold} hard={rmse(y[va], hard_va):.4f} "
            f"main={rmse(y[va], main_oof[va]):.4f} alt={rmse(y[va], alt_oof[va]):.4f}",
            flush=True,
        )

    ae = (main_oof - hard_oof).astype(np.float32)
    em, es = _expanding_stats(wells, ae)
    s2 = pd.DataFrame(
        {
            "main": main_oof,
            "hard": hard_oof,
            "stock": stock,
            "ae": ae,
            "abs_ae": np.abs(ae),
            "md": md,
            "em": em,
            "es": es,
        }
    )
    s2_cols = list(s2.columns)
    anchor_oof = np.zeros(len(df), np.float32)

    for fold, tr, va in iter_group_kfold(wells, n_splits=n_splits):
        # Anchor Stage2: predict (y-main), then shrink on train (P69-style)
        m = _fit_cat(
            anchor_cfg,
            s2.iloc[tr][s2_cols],
            (y - main_oof)[tr],
            s2.iloc[va][s2_cols],
            (y - main_oof)[va],
        )
        pr_tr = m.predict(s2.iloc[tr][s2_cols]).astype(np.float32)
        best_s, best_r = 0.0, rmse(y[tr], main_oof[tr])
        for s in np.linspace(0.0, ANCHOR_SMAX, 41):
            r = rmse(y[tr], main_oof[tr] + s * pr_tr)
            if r < best_r:
                best_r, best_s = r, float(s)
        pred = (main_oof[va] + best_s * m.predict(s2.iloc[va][s2_cols]).astype(np.float32)).astype(
            np.float32
        )
        anchor_oof[va] = pred
        fold_models[fold]["anchor_cat"] = m
        fold_models[fold]["anchor_shrink"] = best_s
        print(f"[anchor] fold{fold} shrink={best_s:.3f} rmse={rmse(y[va], pred):.4f}", flush=True)

    X_stack = np.column_stack([main_oof, alt_oof, hard_oof, hard_oof, stock, md, anchor_oof])
    stack_oof = fit_predict_stack_oof(
        X_stack, y, wells, main=main_oof, anchor=anchor_oof, n_splits=n_splits
    )

    for fold, tr, va in iter_group_kfold(wells, n_splits=n_splits):
        packs = []
        for alpha in STACK_ALPHAS:
            sc = StandardScaler()
            Xt = sc.fit_transform(X_stack[tr])
            reg = Ridge(alpha=alpha).fit(Xt, (y - main_oof)[tr])
            packs.append({"alpha": float(alpha), "scaler": sc, "ridge": reg})
        best_w, best_r = 1.0, 1e9
        for w in np.linspace(0, 1, 41):
            r = rmse(y[tr], w * anchor_oof[tr] + (1 - w) * stack_oof.ridge_bag[tr])
            if r < best_r:
                best_r, best_w = r, float(w)
        fold_models[fold]["stack_packs"] = packs
        fold_models[fold]["blend_w"] = best_w

    blend_wu = warmup_delta(stack_oof.blend, md, tau=pp_tau)
    summary = stack_oof.summary(y, main_oof, anchor_oof)
    summary["rmse_blend_warmup"] = float(rmse(y, blend_wu))
    summary["rmse_hard"] = float(rmse(y, hard_oof))
    summary["rmse_main"] = float(rmse(y, main_oof))
    summary["rmse_alt"] = float(rmse(y, alt_oof))
    summary["rmse_anchor"] = float(rmse(y, anchor_oof))
    summary["recipe"] = {
        "n_splits": n_splits,
        "hard_ridge_alpha": HARD_RIDGE_ALPHA,
        "stack_alphas": list(STACK_ALPHAS),
        "pp_tau": pp_tau,
        "pp_blend": pp_blend,
        "anchor_smax": ANCHOR_SMAX,
        "cat_main_iterations": main_cfg["iterations"],
    }
    print(
        "[topline]",
        json.dumps({k: v for k, v in summary.items() if k not in ("blend_weights",)}, indent=2),
        flush=True,
    )

    return TrainArtifacts(
        track_cols=track_cols,
        cat_cols=cat_cols,
        s2_cols=s2_cols,
        fold_models=fold_models,
        oof={
            "hard": hard_oof,
            "main": main_oof,
            "alt": alt_oof,
            "anchor": anchor_oof,
            "stock": stock,
            "ridge_bag": stack_oof.ridge_bag,
            "blend": stack_oof.blend,
            "blend_warmup": blend_wu,
            "y": y,
            "wells": wells,
        },
        stack_summary=summary,
        pp={"tau": pp_tau, "blend": pp_blend},
    )


def save_artifacts(art: TrainArtifacts, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "track_cols": art.track_cols,
            "cat_cols": art.cat_cols,
            "s2_cols": art.s2_cols,
            "fold_models": art.fold_models,
            "stack_summary": art.stack_summary,
            "pp": art.pp,
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps(art.stack_summary, indent=2, default=float), encoding="utf-8"
    )
