"""Particle-filter TVT tracks (portable, self-contained)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DUAL_MODES, INIT_SPREAD, N_PARTICLES, N_SEEDS, SCALES

TRACK_KEYS = [f"s{sc:g}" for sc in SCALES] + ["mean", "ms_mean"]


def affine_cal(kgr, tw_at_k, min_pts=20):
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        return 1.0, float(np.nanmean(kgr) - np.nanmean(tw_at_k)) if v.any() else 0.0
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1)
    return float(a), float(b)


def run_particle_filter(
    hw,
    tw,
    n_particles=N_PARTICLES,
    seed=42,
    init_spread=INIT_SPREAD,
    modes=None,
    use_affine=False,
):
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 5:
        return None
    last = kn.iloc[-1]
    last_tvt = float(last["TVT_input"])
    last_Z = float(last["Z"])
    last_MD = float(last["MD"])
    tw_at_k = np.interp(kn["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    kgr = kn["GR"].fillna(0).to_numpy(float)
    if use_affine:
        a, b = affine_cal(kn["GR"].to_numpy(float), tw_at_k)
        resid = kgr - (a * tw_at_k + b)
        gs = float(np.clip(np.nanstd(resid), 10.0, 60.0))
    else:
        a, b = 1.0, 0.0
        gs = float(np.clip(np.nanstd(kgr - tw_at_k), 10.0, 60.0))
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].to_numpy(float))
    dz = np.diff(tail["Z"].to_numpy(float))
    dm = np.diff(tail["MD"].to_numpy(float))
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    N = n_particles
    rng = np.random.default_rng(seed)
    ls = last_tvt + last_Z
    if modes is None:
        pos = ls + init_spread * rng.standard_normal(N)
    else:
        modes = np.asarray(modes, float)
        mode_ids = rng.integers(0, len(modes), size=N)
        pos = ls + modes[mode_ids] + init_spread * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N
    MOM, VN, PN, RP, RR, RESAMP = 0.998, 0.002, 0.005, 0.1, 0.001, 0.5
    md_v = ev["MD"].to_numpy(float)
    z_v = ev["Z"].to_numpy(float)
    gr_interp = hw["GR"].interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    gr_v = gr_interp.to_numpy(float)[ev.index.to_numpy()]
    res = np.empty(len(ev))
    prev_MD = last_MD
    log_lik = 0.0
    for i in range(len(ev)):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = np.clip(pos - z_v[i], tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos = tvt_p + z_v[i]
        eg = a * np.interp(tvt_p, tw_tvt, tw_gr) + b
        d = (gr_v[i] - eg) / max(gs, 1e-6)
        lk = np.exp(-0.5 * np.minimum(d**2, 600.0))
        lk = np.maximum(lk, 1e-300)
        avg_lk = float((w * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N
        n_eff = 1.0 / (w**2).sum()
        if n_eff < RESAMP * N:
            cum = np.cumsum(w)
            u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w = np.ones(N) / N
        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]
    return res.astype(np.float32), float(log_lik)


def _ens_from_preds(preds, liks, scales=SCALES):
    pred_arr = np.stack(preds, 0)
    liks = np.asarray(liks, float)
    liks_n = liks - liks.max()
    out = {}
    for sc in scales:
        w = np.exp(liks_n / float(sc))
        w /= w.sum() + 1e-12
        out[f"s{sc:g}"] = (w[:, None] * pred_arr).sum(0).astype(np.float32)
    out["mean"] = pred_arr.mean(0).astype(np.float32)
    out["ms_mean"] = np.mean([out[f"s{sc:g}"] for sc in scales], axis=0).astype(np.float32)
    return out


def pf_track(
    hw,
    tw,
    dual=False,
    use_affine=False,
    n_seeds=N_SEEDS,
    n_particles=N_PARTICLES,
):
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 5:
        return None
    last = float(kn.iloc[-1]["TVT_input"])
    preds, liks = [], []
    modes = DUAL_MODES if dual else None
    for s in range(n_seeds):
        out = run_particle_filter(
            hw, tw, n_particles=n_particles, seed=s, modes=modes, use_affine=use_affine
        )
        if out is None:
            return None
        pred, ll = out
        preds.append(pred)
        liks.append(ll)
    ens = _ens_from_preds(preds, liks)
    return {
        "idx": ev.index.to_numpy(np.int32),
        "last": last,
        "abs": ens,
        "deltas": {k: (v - last).astype(np.float32) for k, v in ens.items()},
    }


def track_matrix(df: pd.DataFrame, track_cols: list[str]) -> np.ndarray:
    cols = []
    for c in track_cols:
        name = c if c in df.columns else f"trk_{c}"
        if name not in df.columns:
            cols.append(np.zeros(len(df), np.float32))
        else:
            cols.append(df[name].to_numpy(np.float32))
    return np.column_stack(cols)
