from __future__ import annotations

from typing import Iterator

import numpy as np
from sklearn.model_selection import GroupKFold


def iter_group_kfold(
    groups: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = False,
    random_state: int | None = None,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (fold_id, train_index, val_index) with grouping by well id."""
    n = len(groups)
    dummy = np.zeros(n)
    gkf = GroupKFold(n_splits=n_splits)
    # sklearn GroupKFold ignores shuffle; keep signature for clarity
    _ = (shuffle, random_state)
    for fold, (tr, va) in enumerate(gkf.split(dummy, dummy, groups)):
        yield fold, tr, va
