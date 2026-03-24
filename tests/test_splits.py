"""Unit tests for group-aware stratified splitting."""

import numpy as np
import pandas as pd
import pytest

from skin_type_classifier.data.splits import stratified_group_split, subsample_training_groups


def _make_df(n_groups: int = 100, images_per_group: int = 2, n_classes: int = 6) -> pd.DataFrame:
    """Create a synthetic dataset with known structure for split testing."""
    rng = np.random.default_rng(42)
    rows = []
    for g in range(n_groups):
        fst = (g % n_classes) + 1  # cycle FST 1-6
        for i in range(images_per_group):
            rows.append(
                {
                    "image_path": f"img_g{g}_{i}.png",
                    "source": "scin",
                    "fitzpatrick_skin_type": fst,
                    "diagnosis": "test",
                    "age": str(rng.integers(20, 80)),
                    "sex": "FEMALE",
                    "group_id": f"G{g}",
                    "revised_fitzpatrick": fst,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    return _make_df(n_groups=120, images_per_group=2)


@pytest.mark.unit
class TestStratifiedGroupSplit:
    def test_no_group_overlap_train_val(self, synthetic_df: pd.DataFrame) -> None:
        train, val, _ = stratified_group_split(synthetic_df)
        assert not (set(train["group_id"]) & set(val["group_id"]))

    def test_no_group_overlap_train_test(self, synthetic_df: pd.DataFrame) -> None:
        train, _, test = stratified_group_split(synthetic_df)
        assert not (set(train["group_id"]) & set(test["group_id"]))

    def test_no_group_overlap_val_test(self, synthetic_df: pd.DataFrame) -> None:
        _, val, test = stratified_group_split(synthetic_df)
        assert not (set(val["group_id"]) & set(test["group_id"]))

    def test_all_rows_accounted_for(self, synthetic_df: pd.DataFrame) -> None:
        train, val, test = stratified_group_split(synthetic_df)
        assert len(train) + len(val) + len(test) == len(synthetic_df)

    def test_all_classes_in_each_split(self, synthetic_df: pd.DataFrame) -> None:
        for split_df in stratified_group_split(synthetic_df):
            present = set(split_df["revised_fitzpatrick"].unique())
            assert present == {1, 2, 3, 4, 5, 6}

    def test_deterministic_with_same_seed(self, synthetic_df: pd.DataFrame) -> None:
        split_a = stratified_group_split(synthetic_df, random_state=0)
        split_b = stratified_group_split(synthetic_df, random_state=0)
        for a, b in zip(split_a, split_b):
            assert set(a["group_id"]) == set(b["group_id"])

    def test_different_seeds_give_different_splits(self, synthetic_df: pd.DataFrame) -> None:
        split_a = stratified_group_split(synthetic_df, random_state=0)
        split_b = stratified_group_split(synthetic_df, random_state=99)
        # At least one split should differ
        assert set(split_a[0]["group_id"]) != set(split_b[0]["group_id"])

    def test_raises_on_insufficient_groups(self) -> None:
        """A class with too few groups should raise ValueError."""
        df = _make_df(n_groups=6, images_per_group=1, n_classes=6)
        with pytest.raises(ValueError, match="fewest groups"):
            stratified_group_split(df)


@pytest.mark.unit
class TestSubsampleTrainingGroups:
    def test_fraction_one_returns_full_copy(self, synthetic_df: pd.DataFrame) -> None:
        result = subsample_training_groups(synthetic_df, fraction=1.0)
        assert len(result) == len(synthetic_df)

    def test_fraction_reduces_size(self, synthetic_df: pd.DataFrame) -> None:
        result = subsample_training_groups(synthetic_df, fraction=0.5)
        assert len(result) < len(synthetic_df)

    def test_all_classes_preserved(self, synthetic_df: pd.DataFrame) -> None:
        result = subsample_training_groups(synthetic_df, fraction=0.3)
        assert set(result["revised_fitzpatrick"].unique()) == set(synthetic_df["revised_fitzpatrick"].unique())

    def test_group_integrity(self, synthetic_df: pd.DataFrame) -> None:
        """All images from kept groups should be present."""
        result = subsample_training_groups(synthetic_df, fraction=0.5)
        for gid in result["group_id"].unique():
            original_count = (synthetic_df["group_id"] == gid).sum()
            result_count = (result["group_id"] == gid).sum()
            assert original_count == result_count, f"Group {gid} has partial images"

    def test_deterministic(self, synthetic_df: pd.DataFrame) -> None:
        a = subsample_training_groups(synthetic_df, fraction=0.5, random_state=42)
        b = subsample_training_groups(synthetic_df, fraction=0.5, random_state=42)
        assert set(a["group_id"]) == set(b["group_id"])
