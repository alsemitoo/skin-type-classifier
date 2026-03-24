"""Clean the SCIN dataset by producing per-image rows with dermatologist Fitzpatrick labels."""

import shutil
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/scin")
PROCESSED_DIR = Path("data/processed/scin")
OUTPUT_IMAGE_DIR = PROCESSED_DIR / "images"

CASES_CSV = RAW_DIR / "dataset_scin_cases.csv"
LABELS_CSV = RAW_DIR / "dataset_scin_labels.csv"

IMAGE_SLOTS = [
    {
        "image_col": "image_1_path",
        "fst_col": "dermatologist_fitzpatrick_skin_type_label_1",
        "gradable_col": "dermatologist_gradable_for_fitzpatrick_skin_type_1",
    },
    {
        "image_col": "image_2_path",
        "fst_col": "dermatologist_fitzpatrick_skin_type_label_2",
        "gradable_col": "dermatologist_gradable_for_fitzpatrick_skin_type_2",
    },
    {
        "image_col": "image_3_path",
        "fst_col": "dermatologist_fitzpatrick_skin_type_label_3",
        "gradable_col": "dermatologist_gradable_for_fitzpatrick_skin_type_3",
    },
]


def extract_image_filename(csv_path: str) -> str | None:
    """Extract the image filename from a SCIN CSV image path.

    Args:
        csv_path: Raw path from the SCIN CSV, e.g. 'dataset/images/-123.png'.

    Returns:
        Just the filename, e.g. '-123.png', or None if the path is empty/NaN.
    """
    if pd.isna(csv_path) or not str(csv_path).strip():
        return None
    return Path(str(csv_path).strip()).name


def parse_fst_label(label: str) -> int | None:
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


def _report_incongruent_cases(valid: pd.DataFrame) -> None:
    """Log cases where different images received different dermatologist FST labels.

    Args:
        valid: DataFrame of per-image rows that have a non-null fitzpatrick_skin_type.
    """
    cases_with_multiple = valid.groupby("case_id")["fitzpatrick_skin_type"].nunique()
    incongruent = cases_with_multiple[cases_with_multiple > 1]

    if len(incongruent) == 0:
        print("No incongruent FST labels found across images within the same case.")
        return

    print(f"\nWARNING: {len(incongruent)} cases have images with different derm FST labels:")
    for case_id in incongruent.index:
        case_rows = valid[valid["case_id"] == case_id][["image_filename", "fitzpatrick_skin_type"]]
        labels = ", ".join(f"{r['image_filename']}=FST{r['fitzpatrick_skin_type']}" for _, r in case_rows.iterrows())
        print(f"  case_id={case_id}: {labels}")
    print()


def clean_scin() -> None:
    """Run the full SCIN cleaning pipeline.

    Reads the raw SCIN cases and labels CSVs, merges them, and produces per-image
    rows with dermatologist-assigned Fitzpatrick skin type labels. Images without a
    dermatologist FST label are written to a separate CSV. Cases where different
    images received different FST labels are logged to stdout.
    """
    cases = pd.read_csv(CASES_CSV)
    labels = pd.read_csv(LABELS_CSV)
    merged = cases.merge(labels, on="case_id", how="inner")

    print(f"Total cases after merge: {len(merged)}")

    rows: list[dict] = []
    for _, case_row in merged.iterrows():
        for slot in IMAGE_SLOTS:
            filename = extract_image_filename(case_row[slot["image_col"]])
            if filename is None:
                continue

            fst_raw = case_row[slot["fst_col"]]
            fst_int = parse_fst_label(fst_raw)
            gradable = case_row[slot["gradable_col"]]

            rows.append(
                {
                    "case_id": case_row["case_id"],
                    "image_filename": filename,
                    "fitzpatrick_skin_type": fst_int,
                    "gradable_for_fitzpatrick": gradable if pd.notna(gradable) else None,
                    "diagnosis": case_row["weighted_skin_condition_label"],
                    "age": case_row["age_group"],
                    "sex": case_row["sex_at_birth"],
                    "self_reported_fitzpatrick": case_row["fitzpatrick_skin_type"],
                }
            )

    all_images = pd.DataFrame(rows)
    print(f"Total image rows (all slots expanded): {len(all_images)}")

    has_label = all_images["fitzpatrick_skin_type"].notna()
    valid = all_images[has_label].copy()
    missing = all_images[~has_label].copy()

    valid["fitzpatrick_skin_type"] = valid["fitzpatrick_skin_type"].astype(int)

    print(f"Valid images (have derm FST label): {len(valid)}")
    print(f"Missing images (no derm FST label): {len(missing)}")

    _report_incongruent_cases(valid)

    image_exists = valid["image_filename"].apply(lambda f: (RAW_DIR / "images" / f).exists())
    not_found = (~image_exists).sum()
    if not_found > 0:
        for fname in valid.loc[~image_exists, "image_filename"]:
            print(f"WARNING: image not found, dropping row: {RAW_DIR / 'images' / fname}")
    valid = valid[image_exists].copy()

    dup_mask = valid.duplicated(subset=["image_filename"], keep=False)
    if dup_mask.any():
        dupes = valid[dup_mask]
        for fname, group in dupes.groupby("image_filename"):
            fst_vals = group["fitzpatrick_skin_type"].unique()
            cases = list(group["case_id"])
            if len(fst_vals) > 1:
                print(f"WARNING: duplicate image {fname} with conflicting FST labels {list(fst_vals)}, cases={cases}")
            else:
                print(f"NOTE: duplicate image {fname} (same label FST{fst_vals[0]}), cases={cases}")
    valid = valid.drop_duplicates(subset=["image_filename"], keep="first")

    print(f"After filtering: {len(valid)} valid rows ({not_found} missing files, {dup_mask.sum()} duplicate rows)")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    missing_path = PROCESSED_DIR / "missing_fitzpatrick_labels_scin.csv"
    missing[["case_id", "image_filename"]].to_csv(missing_path, index=False)
    print(f"Wrote {len(missing)} missing-label images to {missing_path}")

    cleaned_path = PROCESSED_DIR / "cleaned_scin_metadata.csv"
    valid.to_csv(cleaned_path, index=False)
    print(f"Wrote {len(valid)} valid image rows to {cleaned_path}")

    copied = 0
    for _, row in valid.iterrows():
        src = RAW_DIR / "images" / row["image_filename"]
        shutil.copy2(src, OUTPUT_IMAGE_DIR / row["image_filename"])
        copied += 1

    print(f"Copied {copied} images")


if __name__ == "__main__":
    clean_scin()
