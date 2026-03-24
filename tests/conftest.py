"""Shared fixtures and pytest configuration for the skin-type-classifier test suite."""

from pathlib import Path

import pandas as pd
import pytest

FULL_DATASET_PATH = Path("data/processed/full_dataset.csv")


@pytest.fixture()
def sample_scin_cleaned_df() -> pd.DataFrame:
    """Synthetic SCIN cleaned metadata with known majority-voting properties.

    Cases:
        C1: 3 images, FST [2, 2, 3] → clear majority = 2
        C2: 2 images, FST [4, 5] → tie → floor(median(4,5)) = 4
        C3: 1 image, FST [1] → trivial = 1
    """
    return pd.DataFrame(
        {
            "case_id": ["C1", "C1", "C1", "C2", "C2", "C3"],
            "image_filename": ["img1.png", "img2.png", "img3.png", "img4.png", "img5.png", "img6.png"],
            "fitzpatrick_skin_type": [2, 2, 3, 4, 5, 1],
            "diagnosis": ["eczema", "eczema", "eczema", "acne", "acne", "mole"],
            "age": ["AGE_18_TO_29", "AGE_18_TO_29", "AGE_18_TO_29", "AGE_30_TO_39", "AGE_30_TO_39", "AGE_40_TO_49"],
            "sex": ["FEMALE", "FEMALE", "FEMALE", "MALE", "MALE", "FEMALE"],
            "self_reported_fitzpatrick": [2, 2, 2, 4, 4, 1],
            "gradable_for_fitzpatrick": [True, True, True, True, True, True],
        }
    )


@pytest.fixture()
def sample_pad_ufes_cleaned_df() -> pd.DataFrame:
    """Synthetic PAD-UFES cleaned metadata with known properties."""
    return pd.DataFrame(
        {
            "img_id": ["PAT_1_100_1.png", "PAT_1_100_2.png", "PAT_2_200_1.png"],
            "fitspatrick": [2, 2, 5],
            "diagnostic": ["BCC", "BCC", "MEL"],
            "age": [55, 55, 70],
            "gender": ["MALE", "MALE", "FEMALE"],
            "patient_id": ["PAT_1", "PAT_1", "PAT_2"],
        }
    )


@pytest.fixture()
def full_dataset() -> pd.DataFrame:
    """Load the full processed dataset CSV.

    Skips automatically if the CSV is not present (e.g., in CI without DVC pull).
    """
    if not FULL_DATASET_PATH.exists():
        pytest.skip(f"Dataset not found at {FULL_DATASET_PATH} (run 'dvc pull' first)")
    return pd.read_csv(FULL_DATASET_PATH)
