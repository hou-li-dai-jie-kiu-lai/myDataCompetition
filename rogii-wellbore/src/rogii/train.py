"""Train topline: hard Ridge + Cat main/alt + Stage2 anchor + stack."""
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

from .cv import iter_group_kfold
from .dataset import default_cat_cols, default_track_cols
from .features_pf import track_matrix
from .metrics import rmse
from .postprocess import warmup_delta
from .stack import DEFAULT_ALPHAS, fit_predict_stack_oof


CAT_MAIN = dict(
    loss_function="RMSE",
    iterations=400,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
)
CAT_ALT = dict(
    loss_function="RMSE",
    iterations=300,
    depth=5,
    learning_rate=0.06,
    l2_leaf_reg=5.0,
    random_seed=7,
    verbose=False,
    allow_writing_files=False,
)
CAT_ANCHOR = dict(
    loss_function="RMSE",
    iterations=200,
    depth=4,
    learning_rate=0.05,
    l2_leaf_reg=4.0,
    random_seed=11,
    verbose=False,
    allow_writing_files=False,
)


@dataclass
class TrainArtifacts:
    track_cols: list[str]
    cat_cols: list[str]
    s2_cols: list[str]
    fold_models: list[dict]
    oof: dict[str, np.ndarray]
    stack_summary: dict
    pp: dict


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
    n_splits: int = 5,
    pp_tau: float = 85.0,
    cat_iterations: int | None = None,
) -> TrainArtifacts:
    if cat_iterations is not None:
        CAT_MAIN["iterations"] = cat_iterations
        CAT_ALT["iterations"] = max(100, cat_iterations // 2)
        CAT_ANCHOR["iterations"] = max(80, cat_iterations // 3)

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

    for fold, tr, va in iter_group_kfold(wells, n_splits=n_splits):
        sc = StandardScaler()
        Mt = sc.fit_transform(M_all[tr])
        Mv = sc.transform(M_all[va])
        hard_model = Ridge(alpha=1.0, positive=True).fit(Mt, y[tr])
        hard_tr = hard_model.predict(Mt).astype(np.float32)
        hard_va = hard_model.predict(Mv).astype(np.float32)
        hard_oof[va] = hard_va

        Xtr = df.iloc[tr][cat_cols].copy()
        Xva = df.iloc[va][cat_cols].copy()
        Xtr["hard"] = hard_tr
        Xva["hard"] = hard_va
        feat_cols = cat_cols + ["hard"]

        main = CatBoostRegressor(**CAT_MAIN)
        main.fit(Pool(Xtr[feat_cols], y[tr]))
        main_oof[va] = main.predict(Xva[feat_cols]).astype(np.float32)

        alt_cols = [c for c in feat_cols if not str(c).startswith("trk_d_")]
        alt = CatBoostRegressor(**CAT_ALT)
        alt.fit(Pool(Xtr[alt_cols], y[tr]))
        alt_oof[va] = alt.predict(Xva[alt_cols]).astype(np.float32)

        fold_models.append(
            {
                "hard_scaler": sc,
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
        m = CatBoostRegressor(**CAT_ANCHOR)
        m.fit(Pool(s2.iloc[tr][s2_cols], (y - main_oof)[tr]))
        pred = (main_oof[va] + m.predict(s2.iloc[va][s2_cols])).astype(np.float32)
        anchor_oof[va] = pred
        fold_models[fold]["anchor_cat"] = m
        print(f"[anchor] fold{fold} rmse={rmse(y[va], pred):.4f}", flush=True)

    X_stack = np.column_stack([main_oof, alt_oof, hard_oof, hard_oof, stock, md, anchor_oof])
    stack_oof = fit_predict_stack_oof(
        X_stack, y, wells, main=main_oof, anchor=anchor_oof, n_splits=n_splits
    )

    for fold, tr, va in iter_group_kfold(wells, n_splits=n_splits):
        packs = []
        for alpha in DEFAULT_ALPHAS:
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
    print("[topline]", json.dumps({k: v for k, v in summary.items() if k != "blend_weights"}, indent=2), flush=True)

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
        pp={"tau": pp_tau, "blend": 0.7},
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
