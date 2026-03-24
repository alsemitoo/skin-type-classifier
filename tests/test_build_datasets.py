"""Unit tests for skin_type_classifier.data.build_datasets transformation functions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skin_type_classifier.data.build_datasets import (
    UNIFIED_COLUMNS,
    _compute_revised_fst_scin,
    _process_pad_ufes,
    _process_scin,
)


class TestComputeRevisedFstScin:
    """Tests for _compute_revised_fst_scin() majority voting logic."""

    @pytest.mark.unit
    def test_single_image_per_case(self) -> None:
        """Each case has one image — revised FST equals the original."""
        df = pd.DataFrame({"case_id": ["A", "B"], "fitzpatrick_skin_type": [3, 5]})
        result = _compute_revised_fst_scin(df)
        assert list(result) == [3, 5]

    @pytest.mark.unit
    def test_clear_majority(self, sample_scin_cleaned_df: pd.DataFrame) -> None:
        """Case C1 has FST [2, 2, 3] -> majority is 2."""
        result = _compute_revised_fst_scin(sample_scin_cleaned_df)
        c1_values = result[sample_scin_cleaned_df["case_id"] == "C1"]
        assert all(c1_values == 2)

    @pytest.mark.unit
    def test_tie_uses_median_floor(self, sample_scin_cleaned_df: pd.DataFrame) -> None:
        """Case C2 has FST [4, 5] -> tie -> floor(median(4,5)) = 4."""
        result = _compute_revised_fst_scin(sample_scin_cleaned_df)
        c2_values = result[sample_scin_cleaned_df["case_id"] == "C2"]
        assert all(c2_values == 4)

    @pytest.mark.unit
    def test_all_same_label(self) -> None:
        """All images in a case have the same FST — no voting needed."""
        df = pd.DataFrame({"case_id": ["X", "X", "X"], "fitzpatrick_skin_type": [3, 3, 3]})
        result = _compute_revised_fst_scin(df)
        assert all(result == 3)

    @pytest.mark.unit
    def test_three_way_tie(self) -> None:
        """Three-way tie [1, 3, 5] -> floor(median(1,3,5)) = 3."""
        df = pd.DataFrame({"case_id": ["T", "T", "T"], "fitzpatrick_skin_type": [1, 3, 5]})
        result = _compute_revised_fst_scin(df)
        assert all(result == 3)

    @pytest.mark.unit
    def test_returns_int_dtype(self, sample_scin_cleaned_df: pd.DataFrame) -> None:
        """Result should be integer dtype."""
        result = _compute_revised_fst_scin(sample_scin_cleaned_df)
        assert np.issubdtype(result.dtype, np.integer)  # type: ignore


class TestProcessPadUfes:
    """Tests for _process_pad_ufes() schema transformation."""

    @pytest.mark.unit
    def test_output_has_unified_columns(self, sample_pad_ufes_cleaned_df: pd.DataFrame) -> None:
        """Output DataFrame columns must match UNIFIED_COLUMNS exactly."""
        result = _process_pad_ufes(sample_pad_ufes_cleaned_df)
        assert list(result.columns) == UNIFIED_COLUMNS

    @pytest.mark.unit
    def test_source_is_pad_ufes(self, sample_pad_ufes_cleaned_df: pd.DataFrame) -> None:
        """All rows should have source='pad_ufes'."""
        result = _process_pad_ufes(sample_pad_ufes_cleaned_df)
        assert (result["source"] == "pad_ufes").all()

    @pytest.mark.unit
    def test_image_path_prefix(self, sample_pad_ufes_cleaned_df: pd.DataFrame) -> None:
        """Image paths should be prefixed with 'pad_ufes/images/'."""
        result = _process_pad_ufes(sample_pad_ufes_cleaned_df)
        assert all(result["image_path"].str.startswith("pad_ufes/images/"))

    @pytest.mark.unit
    def test_group_id_from_patient_id(self, sample_pad_ufes_cleaned_df: pd.DataFrame) -> None:
        """group_id should be mapped from patient_id."""
        result = _process_pad_ufes(sample_pad_ufes_cleaned_df)
        assert list(result["group_id"]) == ["PAT_1", "PAT_1", "PAT_2"]

    @pytest.mark.unit
    def test_revised_fst_equals_original(self, sample_pad_ufes_cleaned_df: pd.DataFrame) -> None:
        """PAD-UFES has no multi-rater FST, so revised == original."""
        result = _process_pad_ufes(sample_pad_ufes_cleaned_df)
        assert (result["revised_fitzpatrick"] == result["fitzpatrick_skin_type"]).all()

    @pytest.mark.unit
    def test_row_count_preserved(self, sample_pad_ufes_cleaned_df: pd.DataFrame) -> None:
        """Row count should be unchanged after transformation."""
        result = _process_pad_ufes(sample_pad_ufes_cleaned_df)
        assert len(result) == len(sample_pad_ufes_cleaned_df)


class TestProcessScin:
    """Tests for _process_scin() schema transformation.

    _process_scin() checks file existence on disk via PROCESSED_DIR. We use
    monkeypatch + tmp_path to create dummy files so the transformation runs
    without needing real data.
    """

    @pytest.fixture()
    def scin_with_mock_files(
        self, sample_scin_cleaned_df: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> pd.DataFrame:
        """Set up dummy image files and patch PROCESSED_DIR, then return the input df."""
        img_dir = tmp_path / "scin" / "images"
        img_dir.mkdir(parents=True)
        for fname in sample_scin_cleaned_df["image_filename"]:
            (img_dir / fname).touch()
        monkeypatch.setattr("skin_type_classifier.data.build_datasets.PROCESSED_DIR", tmp_path)
        return sample_scin_cleaned_df

    @pytest.mark.unit
    def test_output_has_unified_columns(self, scin_with_mock_files: pd.DataFrame) -> None:
        """Output DataFrame columns must match UNIFIED_COLUMNS exactly."""
        result = _process_scin(scin_with_mock_files)
        assert list(result.columns) == UNIFIED_COLUMNS

    @pytest.mark.unit
    def test_source_is_scin(self, scin_with_mock_files: pd.DataFrame) -> None:
        """All rows should have source='scin'."""
        result = _process_scin(scin_with_mock_files)
        assert (result["source"] == "scin").all()

    @pytest.mark.unit
    def test_image_path_prefix(self, scin_with_mock_files: pd.DataFrame) -> None:
        """Image paths should be prefixed with 'scin/images/'."""
        result = _process_scin(scin_with_mock_files)
        assert all(result["image_path"].str.startswith("scin/images/"))

    @pytest.mark.unit
    def test_revised_fst_matches_voting(self, scin_with_mock_files: pd.DataFrame) -> None:
        """Revised FST should reflect majority voting: C1=2, C2=4, C3=1."""
        result = _process_scin(scin_with_mock_files)
        expected_map = {"img1.png": 2, "img2.png": 2, "img3.png": 2, "img4.png": 4, "img5.png": 4, "img6.png": 1}
        for _, row in result.iterrows():
            fname = row["image_path"].split("/")[-1]
            assert row["revised_fitzpatrick"] == expected_map[fname], f"Mismatch for {fname}"
