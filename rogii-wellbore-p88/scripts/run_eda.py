"""Light EDA over local competition data (set ROGII_DATA_ROOT)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def data_root() -> Path:
    env = os.environ.get("ROGII_DATA_ROOT")
    if not env:
        raise SystemExit(
            "Set ROGII_DATA_ROOT to the competition data root "
            "(folder that contains train/ and sample_submission.csv)."
        )
    p = Path(env)
    if not (p / "train").is_dir():
        raise SystemExit(f"No train/ under {p}")
    return p


def summarize_horizontal(path: Path) -> dict:
    df = pd.read_csv(path)
    kn = df["TVT_input"].notna() if "TVT_input" in df.columns else pd.Series(False, index=df.index)
    out = {
        "file": path.name,
        "rows": len(df),
        "prefix_rows": int(kn.sum()),
        "eval_rows": int((~kn).sum()) if "TVT_input" in df.columns else None,
        "gr_nan_frac": float(df["GR"].isna().mean()) if "GR" in df.columns else None,
        "md_span": float(df["MD"].max() - df["MD"].min()) if "MD" in df.columns else None,
    }
    if "TVT" in df.columns and df["TVT"].notna().any():
        out["tvt_mean"] = float(df["TVT"].mean())
        out["tvt_std"] = float(df["TVT"].std())
    return out


def main() -> None:
    root = data_root()
    train = root / "train"
    wells = sorted(train.glob("*__horizontal_well.csv"))
    print(f"train horizontal files: {len(wells)}")
    rows = [summarize_horizontal(p) for p in wells[:50]]  # cap for speed
    tab = pd.DataFrame(rows)
    print(tab.describe(include="all").transpose().to_string())

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    tab.to_csv(out_dir / "eda_train_sample_summary.csv", index=False)

    # one example plot if available
    if wells:
        df = pd.read_csv(wells[0])
        if {"MD", "Z"}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(df["MD"], df["Z"], lw=1)
            ax.set_xlabel("MD")
            ax.set_ylabel("Z")
            ax.set_title(wells[0].stem)
            fig.tight_layout()
            fig.savefig(out_dir / "eda_example_trajectory.png", dpi=120)
            plt.close(fig)
            print("wrote", out_dir / "eda_example_trajectory.png")

    sample = root / "sample_submission.csv"
    if sample.exists():
        sub = pd.read_csv(sample)
        print("sample_submission rows", len(sub), "cols", list(sub.columns))


if __name__ == "__main__":
    main()
