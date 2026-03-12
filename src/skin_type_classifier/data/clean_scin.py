"""Clean the SCIN dataset by filtering cases with valid Fitzpatrick skin type labels."""

import shutil
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/scin")
PROCESSED_DIR = Path("data/processed")
OUTPUT_IMAGE_DIR = PROCESSED_DIR / "scin"

CASES_CSV = RAW_DIR / "scin_cases.csv"
LABELS_CSV = RAW_DIR / "scin_labels.csv"

FITZ_COLS = [
    "dermatologist_fitzpatrick_skin_type_label_1",
    "dermatologist_fitzpatrick_skin_type_label_2",
    "dermatologist_fitzpatrick_skin_type_label_3",
]
IMAGE_COLS = ["image_1_path", "image_2_path", "image_3_path"]


def remap_image_path(csv_path: str) -> Path | None:
    """Convert CSV image path (dataset/images/<id>.png) to actual filesystem path."""
    if pd.isna(csv_path) or not str(csv_path).strip():
        return None
    filename = Path(str(csv_path).strip()).name
    return RAW_DIR / "scin_images" / filename


def clean_scin() -> None:
    """Run the full SCIN cleaning pipeline."""
    cases = pd.read_csv(CASES_CSV)
    labels = pd.read_csv(LABELS_CSV)
    merged = cases.merge(labels, on="case_id", how="inner")

    has_label = merged[FITZ_COLS].notna().any(axis=1) & (
        merged[FITZ_COLS].astype(str).replace("", pd.NA).notna().any(axis=1)
    )
    valid = merged[has_label].copy()
    invalid = merged[~has_label].copy()

    print(f"Total cases: {len(merged)}")
    print(f"Valid cases (≥1 Fitzpatrick label): {len(valid)}")
    print(f"Invalid cases (no labels): {len(invalid)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    missing_path = PROCESSED_DIR / "missing_fitzpatrick_labels_scin.csv"
    invalid[["case_id"] + IMAGE_COLS].to_csv(missing_path, index=False)
    print(f"Wrote {len(invalid)} invalid cases to {missing_path}")

    cleaned_path = PROCESSED_DIR / "cleaned_scin_metadata.csv"
    valid.to_csv(cleaned_path, index=False)
    print(f"Wrote {len(valid)} valid cases to {cleaned_path}")

    copied, skipped, missing = 0, 0, 0
    for _, row in valid.iterrows():
        for col in IMAGE_COLS:
            src = remap_image_path(row[col])
            if src is None:
                skipped += 1
                continue
            if not src.exists():
                print(f"WARNING: image not found: {src}")
                missing += 1
                continue
            shutil.copy2(src, OUTPUT_IMAGE_DIR / src.name)
            copied += 1

    print(f"Copied {copied} images, skipped {skipped} empty refs, {missing} missing files")


if __name__ == "__main__":
    clean_scin()
