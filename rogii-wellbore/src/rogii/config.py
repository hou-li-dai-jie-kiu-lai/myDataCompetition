"""
Default hyperparameters for the published ROGII pipeline.

  PF: 64 seeds × 250 particles, scales (3,5,8,12), dual modes ±15
  Hard: Ridge(alpha=1, positive=True) on raw/aff/dual tracks
  Main Cat: residual-on-hard (8000 iter, depth 7, lr 0.025, OD)
  Anchor: Stage-2 Cat on (y - main), shrink grid up to 1.6
  Stack: multi-alpha Ridge bag {0.3,1,3,10,30} ⊕ blend with anchor
  PP: warm-up tau=85, U=TVT+Z projection blend=0.7
  CV: GroupKFold 5 by well
"""
from __future__ import annotations

# --- PF / features ---
N_SEEDS = 64
N_PARTICLES = 250
INIT_SPREAD = 2.0
SCALES = (3.0, 5.0, 8.0, 12.0)
DUAL_MODES = (-15.0, 0.0, 15.0)

# --- CV ---
N_SPLITS = 5

# --- Hard Ridge ---
HARD_RIDGE_ALPHA = 1.0

# --- CatBoost (main) ---
CAT_MAIN = dict(
    iterations=8000,
    depth=7,
    learning_rate=0.025,
    l2_leaf_reg=2.0,
    min_data_in_leaf=15,
    border_count=254,
    loss_function="RMSE",
    od_type="Iter",
    od_wait=300,
    verbose=0,
    random_seed=42,
    allow_writing_files=False,
)

# alt channel (shallower / different seed; no dual tracks at train time)
CAT_ALT = dict(
    iterations=8000,
    depth=6,
    learning_rate=0.025,
    l2_leaf_reg=2.0,
    min_data_in_leaf=15,
    border_count=254,
    loss_function="RMSE",
    od_type="Iter",
    od_wait=300,
    verbose=0,
    random_seed=7,
    allow_writing_files=False,
)

# Stage-2 / anchor
CAT_ANCHOR = dict(
    iterations=5000,
    depth=5,
    learning_rate=0.04,
    l2_leaf_reg=3.0,
    loss_function="RMSE",
    od_type="Iter",
    od_wait=200,
    verbose=0,
    random_seed=7,
    allow_writing_files=False,
)
ANCHOR_SMAX = 1.6

# --- Stack ---
STACK_ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0)
BLEND_GRID = 41

# --- Post-process ---
PP_TAU = 85.0
PP_BLEND = 0.7

# Leaderboard reference (not computed in-repo)
REF_PRIVATE_LB = 8.836
REF_PUBLIC_LB = 7.841
REF_OOF_PROJ = 8.398
