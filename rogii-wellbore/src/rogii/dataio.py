from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_horizontal_well(path: str | Path) -> pd.DataFrame:
    """Load one `{WELL}__horizontal_well.csv`."""
    df = pd.read_csv(path)
    need = {"MD", "Z"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def last_known_tvt(hw: pd.DataFrame) -> float:
    if "TVT_input" not in hw.columns:
        raise ValueError("TVT_input required to locate known prefix")
    kn = hw[hw["TVT_input"].notna()]
    if kn.empty:
        raise ValueError("no known TVT_input rows")
    return float(kn.iloc[-1]["TVT_input"])


def build_residual_table(
    train_dir: str | Path,
    max_wells: int | None = None,
) -> pd.DataFrame:
    """
    Build a row-level table for wells that have label TVT.

    Columns: well, row_idx, id, md, z, gr, md_since, last_known, y
    where y = TVT - last_known (residual target).
    Only rows with finite TVT are kept (typical train eval rows).
    """
    train_dir = Path(train_dir)
    files = sorted(train_dir.glob("*__horizontal_well.csv"))
    if max_wells is not None:
        files = files[:max_wells]

    rows: list[dict] = []
    for path in files:
        well = path.name.split("__", 1)[0]
        hw = load_horizontal_well(path)
        if "TVT" not in hw.columns:
            continue
        try:
            last = last_known_tvt(hw)
        except ValueError:
            continue
        kn = hw[hw["TVT_input"].notna()]
        md_last = float(kn.iloc[-1]["MD"])
        for i, r in hw.iterrows():
            if not np.isfinite(r.get("TVT", np.nan)):
                continue
            md = float(r["MD"])
            rows.append(
                {
                    "well": well,
                    "row_idx": int(i),
                    "id": f"{well}_{int(i)}",
                    "md": md,
                    "z": float(r["Z"]),
                    "gr": float(r["GR"]) if "GR" in hw.columns and np.isfinite(r.get("GR", np.nan)) else np.nan,
                    "md_since": max(md - md_last, 0.0),
                    "last_known": last,
                    "y": float(r["TVT"]) - last,
                    "tvt": float(r["TVT"]),
                }
            )
    return pd.DataFrame(rows)
