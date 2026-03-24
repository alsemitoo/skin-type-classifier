"""Build consolidated dataset CSVs from the three cleaned source datasets."""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")

SCIN_CSV = PROCESSED_DIR / "scin" / "cleaned_scin_metadata.csv"
PAD_UFES_CSV = PROCESSED_DIR / "pad_ufes" / "cleaned_pad_ufes_metadata.csv"

FULL_DATASET_CSV = PROCESSED_DIR / "full_dataset.csv"

UNIFIED_COLUMNS = [
    "image_path",
    "source",
    "fitzpatrick_skin_type",
    "diagnosis",
    "age",
    "sex",
]


def _process_scin(df: pd.DataFrame) -> pd.DataFrame:
    """Transform SCIN per-image data into the unified schema.

    Args:
        df: SCIN cleaned metadata DataFrame (already one row per image).

    Returns:
        DataFrame with unified columns.
    """
    result = pd.DataFrame(
        {
            "image_path": "scin/images/" + df["image_filename"],
            "source": "scin",
            "fitzpatrick_skin_type": df["fitzpatrick_skin_type"].astype(int),
            "diagnosis": df["diagnosis"],
            "age": df["age"].astype(str),
            "sex": df["sex"],
        },
        columns=UNIFIED_COLUMNS,
    )
    exists_mask = result["image_path"].apply(lambda p: (PROCESSED_DIR / p).exists())
    dropped = (~exists_mask).sum()
    if dropped > 0:
        print(f"WARNING: dropped {dropped} SCIN rows with missing image files")
    return result[exists_mask].drop_duplicates(subset=["image_path"], keep="first")


def _process_pad_ufes(df: pd.DataFrame) -> pd.DataFrame:
    """Transform PAD-UFES data into the unified schema.

    Args:
        df: Raw PAD-UFES cleaned metadata DataFrame.

    Returns:
        DataFrame with unified columns.
    """
    result = pd.DataFrame(
        {
            "image_path": "pad_ufes/images/" + df["img_id"],
            "source": "pad_ufes",
            "fitzpatrick_skin_type": df["fitspatrick"].astype(int),
            "diagnosis": df["diagnostic"],
            "age": df["age"].astype(str),
            "sex": df["gender"],
        },
        columns=UNIFIED_COLUMNS,
    )
    return result


def build_datasets() -> None:
    """Build consolidated dataset CSVs from cleaned source datasets."""
    scin_df = pd.read_csv(SCIN_CSV)
    pad_ufes_df = pd.read_csv(PAD_UFES_CSV)

    scin = _process_scin(scin_df)
    pad_ufes = _process_pad_ufes(pad_ufes_df)

    print(f"SCIN: {len(scin)} rows")
    print(f"PAD-UFES: {len(pad_ufes)} rows")

    res_dataset = pd.concat([scin, pad_ufes], ignore_index=True)
    res_dataset.to_csv(FULL_DATASET_CSV, index=False)
    print(f"Wrote dataset without Fitzpatrick: {len(res_dataset)} rows -> {FULL_DATASET_CSV}")


if __name__ == "__main__":
    build_datasets()
