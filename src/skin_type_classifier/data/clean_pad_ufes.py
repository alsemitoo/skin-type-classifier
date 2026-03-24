"""Clean the PAD-UFES-20 dataset by filtering cases with valid Fitzpatrick skin type labels."""

import shutil
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/pad_ufes")  # Non-existant anymore for storage reasons
PROCESSED_DIR = Path("data/processed/pad_ufes")
OUTPUT_IMAGE_DIR = PROCESSED_DIR / "images"

METADATA_CSV = RAW_DIR / "metadata.csv"
FITZ_COL = "fitspatrick"
IMAGE_COL = "img_id"


def clean_pad_ufes() -> None:
    """Run the full PAD-UFES-20 cleaning pipeline."""
    df = pd.read_csv(METADATA_CSV)

    has_label = df[FITZ_COL].notna()
    valid = df[has_label].copy()
    invalid = df[~has_label].copy()

    print(f"Total cases: {len(df)}")
    print(f"Valid cases (has Fitzpatrick label): {len(valid)}")
    print(f"Invalid cases (no label): {len(invalid)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    missing_path = PROCESSED_DIR / "missing_fitzpatrick_labels_pad_ufes.csv"
    invalid[["patient_id", IMAGE_COL]].to_csv(missing_path, index=False)
    print(f"Wrote {len(invalid)} invalid cases to {missing_path}")

    cleaned_path = PROCESSED_DIR / "cleaned_pad_ufes_metadata.csv"
    valid.to_csv(cleaned_path, index=False)
    print(f"Wrote {len(valid)} valid cases to {cleaned_path}")

    copied, missing = 0, 0
    for _, row in valid.iterrows():
        src = RAW_DIR / "images" / row[IMAGE_COL]
        if not src.exists():
            print(f"WARNING: image not found: {src}")
            missing += 1
            continue
        shutil.copy2(src, OUTPUT_IMAGE_DIR / src.name)
        copied += 1

    print(f"Copied {copied} images, {missing} missing files")


if __name__ == "__main__":
    clean_pad_ufes()
