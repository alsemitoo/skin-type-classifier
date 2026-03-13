"""Build consolidated dataset CSVs from the three cleaned source datasets."""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")

SCIN_CSV = PROCESSED_DIR / "scin" / "cleaned_scin_metadata.csv"
PAD_UFES_CSV = PROCESSED_DIR / "pad_ufes" / "cleaned_pad_ufes_metadata.csv"
FITZPATRICK_CSV = PROCESSED_DIR / "fitzpatrick_17k" / "cleaned_fitzpatrick_17k_metadata.csv"

FULL_DATASET_CSV = PROCESSED_DIR / "full_dataset.csv"
NO_FITZ_DATASET_CSV = PROCESSED_DIR / "dataset_without_fitzpatrick.csv"

SCIN_IMAGE_COLS = ["image_1_path", "image_2_path", "image_3_path"]
UNIFIED_COLUMNS = ["image_path", "source", "fitzpatrick_skin_type", "diagnosis", "age", "sex"]


def _scin_image_path(csv_path: str) -> str:
    """Convert SCIN CSV image path (dataset/images/<id>.png) to processed-relative path.

    Args:
        csv_path: Raw path from the SCIN CSV, e.g. 'dataset/images/-123.png'.

    Returns:
        Path relative to data/processed/, e.g. 'scin/scin_images/-123.png'.
    """
    filename = Path(str(csv_path).strip()).name
    return f"scin/scin_images/{filename}"


def _parse_scin_fst(label: str) -> int | None:
    """Parse a SCIN Fitzpatrick label like 'FST2' into an integer.

    Args:
        label: Fitzpatrick label string, e.g. 'FST1' through 'FST6'.

    Returns:
        Integer skin type (1-6), or None if unparseable.
    """
    if pd.isna(label) or not str(label).startswith("FST"):
        return None
    try:
        return int(str(label).removeprefix("FST"))
    except ValueError:
        return None


def _process_scin(df: pd.DataFrame) -> pd.DataFrame:
    """Transform SCIN data into the unified schema, exploding multi-image rows.

    Args:
        df: Raw SCIN cleaned metadata DataFrame.

    Returns:
        DataFrame with unified columns, one row per image.
    """
    rows = []
    for _, row in df.iterrows():
        fst = _parse_scin_fst(row["dermatologist_fitzpatrick_skin_type_label_1"])
        if fst is None:
            continue
        for col in SCIN_IMAGE_COLS:
            val = row[col]
            if pd.isna(val) or not str(val).strip():
                continue
            rows.append(
                {
                    "image_path": _scin_image_path(val),
                    "source": "scin",
                    "fitzpatrick_skin_type": fst,
                    "diagnosis": row["weighted_skin_condition_label"],
                    "age": row["age_group"],
                    "sex": row["sex_at_birth"],
                }
            )
    return pd.DataFrame(rows, columns=UNIFIED_COLUMNS)


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


def _process_fitzpatrick(df: pd.DataFrame) -> pd.DataFrame:
    """Transform Fitzpatrick 17k data into the unified schema.

    Args:
        df: Raw Fitzpatrick 17k cleaned metadata DataFrame.

    Returns:
        DataFrame with unified columns.
    """
    result = pd.DataFrame(
        {
            "image_path": "fitzpatrick_17k/images/" + df["new_img_name"],
            "source": "fitzpatrick_17k",
            "fitzpatrick_skin_type": df["fitzpatrick"].astype(int),
            "diagnosis": df["label"],
            "age": pd.NA,
            "sex": pd.NA,
        },
        columns=UNIFIED_COLUMNS,
    )
    return result


def build_datasets() -> None:
    """Build consolidated dataset CSVs from cleaned source datasets."""
    scin_df = pd.read_csv(SCIN_CSV)
    pad_ufes_df = pd.read_csv(PAD_UFES_CSV)
    fitzpatrick_df = pd.read_csv(FITZPATRICK_CSV)

    scin = _process_scin(scin_df)
    pad_ufes = _process_pad_ufes(pad_ufes_df)
    fitzpatrick = _process_fitzpatrick(fitzpatrick_df)

    print(f"SCIN: {len(scin)} rows (exploded from {len(scin_df)} cases)")
    print(f"PAD-UFES: {len(pad_ufes)} rows")
    print(f"Fitzpatrick 17k: {len(fitzpatrick)} rows")

    full = pd.concat([scin, pad_ufes, fitzpatrick], ignore_index=True)
    full.to_csv(FULL_DATASET_CSV, index=False)
    print(f"Wrote full dataset: {len(full)} rows -> {FULL_DATASET_CSV}")

    no_fitz = pd.concat([scin, pad_ufes], ignore_index=True)
    no_fitz.to_csv(NO_FITZ_DATASET_CSV, index=False)
    print(f"Wrote dataset without Fitzpatrick: {len(no_fitz)} rows -> {NO_FITZ_DATASET_CSV}")


if __name__ == "__main__":
    build_datasets()
