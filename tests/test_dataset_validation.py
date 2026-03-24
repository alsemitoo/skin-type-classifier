"""Data integrity and schema validation tests for the processed full_dataset.csv."""

import pandas as pd
import pytest

from skin_type_classifier.data.build_datasets import UNIFIED_COLUMNS

EXPECTED_SOURCES = {"scin", "pad_ufes"}


@pytest.mark.data
class TestDatasetSchema:
    """Validate that the dataset CSV has the correct schema."""

    def test_columns_match_unified_schema(self, full_dataset: pd.DataFrame) -> None:
        """Column names and order must match UNIFIED_COLUMNS."""
        assert list(full_dataset.columns) == UNIFIED_COLUMNS

    def test_no_empty_dataframe(self, full_dataset: pd.DataFrame) -> None:
        """Dataset must not be empty."""
        assert len(full_dataset) > 0

    def test_fitzpatrick_skin_type_dtype(self, full_dataset: pd.DataFrame) -> None:
        """fitzpatrick_skin_type must be integer."""
        assert pd.api.types.is_integer_dtype(full_dataset["fitzpatrick_skin_type"])

    def test_revised_fitzpatrick_dtype(self, full_dataset: pd.DataFrame) -> None:
        """revised_fitzpatrick must be integer."""
        assert pd.api.types.is_integer_dtype(full_dataset["revised_fitzpatrick"])


@pytest.mark.data
class TestDatasetIntegrity:
    """Validate data integrity constraints."""

    def test_image_path_uniqueness(self, full_dataset: pd.DataFrame) -> None:
        """Each image_path must be unique across the entire dataset."""
        assert full_dataset["image_path"].is_unique, "Duplicate image_path entries found"

    def test_source_values(self, full_dataset: pd.DataFrame) -> None:
        """Source column must contain exactly the expected dataset identifiers."""
        actual_sources = set(full_dataset["source"].unique())
        assert actual_sources == EXPECTED_SOURCES

    def test_fitzpatrick_skin_type_in_valid_range(self, full_dataset: pd.DataFrame) -> None:
        """All fitzpatrick_skin_type values must be in [1, 6]."""
        fst = full_dataset["fitzpatrick_skin_type"]
        assert fst.between(1, 6).all(), f"FST values outside 1-6: {fst[~fst.between(1, 6)].unique()}"

    def test_revised_fitzpatrick_in_valid_range(self, full_dataset: pd.DataFrame) -> None:
        """All revised_fitzpatrick values must be in [1, 6]."""
        rfst = full_dataset["revised_fitzpatrick"]
        assert rfst.between(1, 6).all(), f"Revised FST values outside 1-6: {rfst[~rfst.between(1, 6)].unique()}"

    def test_no_null_image_path(self, full_dataset: pd.DataFrame) -> None:
        """image_path must not contain any null values."""
        assert full_dataset["image_path"].notna().all()

    def test_no_null_source(self, full_dataset: pd.DataFrame) -> None:
        """source must not contain any null values."""
        assert full_dataset["source"].notna().all()

    def test_no_null_fitzpatrick_skin_type(self, full_dataset: pd.DataFrame) -> None:
        """fitzpatrick_skin_type must not contain any null values."""
        assert full_dataset["fitzpatrick_skin_type"].notna().all()

    def test_no_null_group_id(self, full_dataset: pd.DataFrame) -> None:
        """group_id must not contain any null values."""
        assert full_dataset["group_id"].notna().all()

    def test_group_id_non_empty(self, full_dataset: pd.DataFrame) -> None:
        """group_id must be non-empty (critical for stratified splitting)."""
        assert (full_dataset["group_id"].astype(str).str.strip() != "").all()


@pytest.mark.data
class TestDatasetDistribution:
    """Track data distributions and minimum representation."""

    def test_minimum_row_count(self, full_dataset: pd.DataFrame) -> None:
        """Catch unexpected data loss during pipeline reruns."""
        assert len(full_dataset) >= 6000, f"Dataset suspiciously small: {len(full_dataset)} rows"

    def test_both_sources_have_substantial_data(self, full_dataset: pd.DataFrame) -> None:
        """Each source dataset must contribute a meaningful number of rows."""
        source_counts = full_dataset["source"].value_counts()
        assert source_counts["scin"] >= 4000, f"SCIN has only {source_counts['scin']} rows"
        assert source_counts["pad_ufes"] >= 1000, f"PAD-UFES has only {source_counts['pad_ufes']} rows"

    def test_all_fst_types_represented(self, full_dataset: pd.DataFrame) -> None:
        """Every FST 1-6 should have at least some samples."""
        present_types = set(full_dataset["fitzpatrick_skin_type"].unique())
        assert present_types == {1, 2, 3, 4, 5, 6}, f"Missing FST types: {set(range(1, 7)) - present_types}"


@pytest.mark.data
class TestDataLeakagePrevention:
    """Placeholder tests for train/test split leakage detection.

    These tests will become meaningful once train/val/test splitting is implemented.
    They document the critical invariant: no group_id should appear in both train
    and test sets (to prevent patient/case-level leakage).
    """

    @pytest.mark.skip(reason="Train/test split not yet implemented")
    def test_no_group_id_overlap_between_train_and_test(self) -> None:
        """No group_id should appear in both train and test sets."""

    @pytest.mark.skip(reason="Train/test split not yet implemented")
    def test_no_group_id_overlap_between_train_and_val(self) -> None:
        """No group_id should appear in both train and validation sets."""

    @pytest.mark.skip(reason="Fairness evaluation not yet implemented")
    def test_fst_distribution_similar_across_splits(self) -> None:
        """FST distribution should be roughly similar in train/val/test (stratification check)."""
