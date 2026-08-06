# ROGII Wellbore Geology Prediction — P88 Writeup & Repro Kit

Kaggle competition share focused on **method notes + reproducible evaluation / stacking code**.

| Item | Value |
|------|--------|
| Competition | [rogii-wellbore-geology-prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction) |
| Highlighted recipe | **P88** (core5 Ridge residual bag ⊕ P69 blend + PP) |
| Our private LB | **8.836** (final ~rank 422) |
| Our public LB (P20H hold) | **7.815** |

This repository does **not** include:

- competition CSV / well files
- trained model weights (`.pkl` / CatBoost / LightGBM dumps)
- Kaggle API tokens or personal paths with credentials

## What’s inside

```text
docs/                 competition brief, data how-to, P88 approach, lessons
src/rogii_share/      metrics, GroupKFold helpers, post-process, P88 ridge blend
scripts/              EDA + eval demos (synthetic) + recipe runner
```

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt

# synthetic sanity check (no competition data required)
python scripts/run_eval_demo.py

# after you download data yourself (see docs/02_data.md):
#   set ROGII_DATA_ROOT=/path/to/competition/root
python scripts/run_eda.py
```

## Docs

1. [赛题介绍](docs/01_competition.md)
2. [数据获取与结构](docs/02_data.md)
3. [P88 方法](docs/03_approach_p88.md)
4. [踩坑与经验](docs/04_lessons.md)

## Results snapshot (our team)

| Version | Offline OOF (proj) | Public LB | Private LB | Notes |
|---------|--------------------|-----------|------------|-------|
| P20H | ~8.71 | **7.815** | 9.039 | Single-DNA hold on public |
| P69 | ~8.45 | 7.853 | 8.908 | Stage2 on P50e lineage |
| **P88** | ~8.40 | 7.841 | **8.836** | Best private among our selects |
| P104 | ~8.39 | 9.542 | 9.840 | Prefix-gate reverse disaster |

Takeaway: **public and private disagreed hard**; prefix-heuristic gates that looked great offline could explode on LB.

## License

MIT for the code/docs in this repo. Competition data remains under **Kaggle competition rules** — do not redistribute.
