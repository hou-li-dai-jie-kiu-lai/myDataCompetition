"""Light EDA. Requires ROGII_DATA_ROOT pointing to competition data root."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii.dataio import build_residual_table, load_horizontal_well


def main() -> None:
    root = os.environ.get("ROGII_DATA_ROOT")
    if not root:
        raise SystemExit("Set ROGII_DATA_ROOT to the folder that contains train/")
    root_p = Path(root)
    train = root_p / "train"
    if not train.is_dir():
        raise SystemExit(f"missing {train}")

    files = sorted(train.glob("*__horizontal_well.csv"))
    print(f"horizontal wells: {len(files)}")

    # residual table on a sample of wells (full scan can be slow)
    tab = build_residual_table(train, max_wells=min(30, len(files)))
    print("residual rows (first wells):", len(tab))
    if len(tab):
        print(tab[["y", "md_since", "gr"]].describe().to_string())

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    if len(tab):
        tab.to_csv(out / "residual_sample.csv", index=False)

    if files:
        hw = load_horizontal_well(files[0])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(hw["MD"], hw["Z"], lw=1)
        ax.set_xlabel("MD")
        ax.set_ylabel("Z")
        ax.set_title(files[0].stem)
        fig.tight_layout()
        fig.savefig(out / "example_trajectory.png", dpi=120)
        plt.close(fig)
        print("wrote", out / "example_trajectory.png")

    sample = root_p / "sample_submission.csv"
    if sample.exists():
        sub = pd.read_csv(sample)
        print("sample_submission", len(sub), list(sub.columns))


if __name__ == "__main__":
    main()
