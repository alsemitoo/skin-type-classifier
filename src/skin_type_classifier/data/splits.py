"""Group-aware stratified splitting for FST classification.

All splits operate at the group level (``group_id``) so that images from the
same patient/case never leak across train, validation, and test sets.
Stratification uses ``revised_fitzpatrick`` to preserve class proportions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def _get_group_labels(df: pd.DataFrame) -> pd.Series:
    """Assign each ``group_id`` its modal ``revised_fitzpatrick``."""
    return df.groupby("group_id")["revised_fitzpatrick"].first()


def stratified_group_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/val/test with group-aware stratification.

    Guarantees:
        1. No ``group_id`` appears in more than one split.
        2. ``revised_fitzpatrick`` distribution is approximately preserved.
        3. Deterministic given a fixed ``random_state``.

    Returns:
        ``(train_df, val_df, test_df)`` — each a copy of the relevant rows.
    """
    group_labels = _get_group_labels(df)
    groups = group_labels.index.values
    labels = group_labels.values

    min_groups_per_class = pd.Series(labels).value_counts().min()
    required_minimum = max(3, int(np.ceil(1.0 / min(test_size, val_size))))
    if min_groups_per_class < required_minimum:
        raise ValueError(
            f"FST class with fewest groups has {min_groups_per_class} groups, "
            f"need at least {required_minimum} for a {test_size}/{val_size} split."
        )

    # First split: isolate test groups
    splitter_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(splitter_test.split(groups, labels))  # type: ignore

    test_groups = set(groups[test_idx])
    trainval_groups = groups[trainval_idx]
    trainval_labels = labels[trainval_idx]

    # Second split: isolate val from trainval
    adjusted_val_size = val_size / (1.0 - test_size)
    splitter_val = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=random_state)
    train_idx, val_idx = next(splitter_val.split(trainval_groups, trainval_labels))

    train_groups = set(trainval_groups[train_idx])
    val_groups = set(trainval_groups[val_idx])

    train_df = df[df["group_id"].isin(train_groups)].copy()
    val_df = df[df["group_id"].isin(val_groups)].copy()
    test_df = df[df["group_id"].isin(test_groups)].copy()

    return train_df, val_df, test_df


def subsample_training_groups(
    train_df: pd.DataFrame,
    fraction: float,
    random_state: int = 42,
) -> pd.DataFrame:
    """Subsample training data at the group level, stratified by FST.

    For use in learning curve experiments: vary the amount of training data
    while preserving class proportions and group integrity.

    Args:
        train_df: Full training split DataFrame.
        fraction: Fraction of training groups to keep (0.0 to 1.0).
        random_state: Seed for reproducibility.

    Returns:
        Subsampled DataFrame containing only images from selected groups.

    Raises:
        ValueError: If the subsample would leave any FST class with zero groups.
    """
    if fraction >= 1.0:
        return train_df.copy()

    group_labels = _get_group_labels(train_df)
    groups = group_labels.index.values
    labels = group_labels.values

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - fraction, random_state=random_state)
    keep_idx, _ = next(splitter.split(groups, labels))  # type: ignore
    keep_groups = set(groups[keep_idx])

    result = train_df[train_df["group_id"].isin(keep_groups)].copy()

    present_classes = set(result["revised_fitzpatrick"].unique())
    expected_classes = set(train_df["revised_fitzpatrick"].unique())
    missing = expected_classes - present_classes
    if missing:
        raise ValueError(f"Subsample at fraction={fraction} lost FST classes: {sorted(missing)}")

    return result
