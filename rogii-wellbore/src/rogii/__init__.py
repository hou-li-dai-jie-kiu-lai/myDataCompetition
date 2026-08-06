"""ROGII topline share: metrics, CV, postprocess, residual stacking."""

from .metrics import rmse, well_rmse_table
from .cv import iter_group_kfold
from .postprocess import warmup_delta, project_well_absolute, apply_pp_delta
from .stack import fit_predict_stack_oof
from .pipeline import run_topline_stack, channels_from_npz
from .dataio import load_horizontal_well, build_residual_table

__all__ = [
    "rmse",
    "well_rmse_table",
    "iter_group_kfold",
    "warmup_delta",
    "project_well_absolute",
    "apply_pp_delta",
    "fit_predict_stack_oof",
    "run_topline_stack",
    "channels_from_npz",
    "load_horizontal_well",
    "build_residual_table",
]
