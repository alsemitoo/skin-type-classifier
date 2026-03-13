"""Validate that the processed data directory is intact after DVC operations.

Checks:
- Each CSV file exists, has the expected number of rows and columns.
- Each image folder exists and contains the expected number of image files.
- Every image path referenced in full_dataset.csv resolves to an actual file.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


@dataclass
class CSVCheck:
    """Expected properties of a CSV file.

    Attributes:
        path: Path relative to the project root.
        expected_rows: Expected number of data rows (excluding the header).
        expected_cols: Expected number of columns.
        expected_col_names: If provided, the exact list of column names.
    """

    path: Path
    expected_rows: int
    expected_cols: int
    expected_col_names: list[str] = field(default_factory=list)


@dataclass
class ImageFolderCheck:
    """Expected properties of an image folder.

    Attributes:
        path: Path relative to the project root.
        expected_count: Expected number of image files in the folder.
        extensions: Allowed image file extensions (lowercase, with dot).
    """

    path: Path
    expected_count: int
    extensions: list[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png"])


UNIFIED_COLS = ["image_path", "source", "fitzpatrick_skin_type", "diagnosis", "age", "sex"]

CSV_CHECKS: list[CSVCheck] = [
    CSVCheck(
        path=PROCESSED_DIR / "full_dataset.csv",
        expected_rows=19127,
        expected_cols=6,
        expected_col_names=UNIFIED_COLS,
    ),
    CSVCheck(
        path=PROCESSED_DIR / "dataset_without_fitzpatrick.csv",
        expected_rows=10387,
        expected_cols=6,
        expected_col_names=UNIFIED_COLS,
    ),
    CSVCheck(
        path=PROCESSED_DIR / "fitzpatrick_17k" / "cleaned_fitzpatrick_17k_metadata.csv",
        expected_rows=8740,
        expected_cols=15,
    ),
    CSVCheck(
        path=PROCESSED_DIR / "fitzpatrick_17k" / "wrong_entries_fitzpatrick_17k.csv",
        expected_rows=2654,
        expected_cols=15,
    ),
    CSVCheck(
        path=PROCESSED_DIR / "fitzpatrick_17k" / "missing_fitzpatrick_labels_fitzpatrick_17k.csv",
        expected_rows=296,
        expected_cols=3,
        expected_col_names=["new_img_name", "fitzpatrick", "url"],
    ),
    CSVCheck(
        path=PROCESSED_DIR / "pad_ufes" / "cleaned_pad_ufes_metadata.csv",
        expected_rows=1494,
        expected_cols=26,
    ),
    CSVCheck(
        path=PROCESSED_DIR / "pad_ufes" / "missing_fitzpatrick_labels_pad_ufes.csv",
        expected_rows=804,
        expected_cols=2,
        expected_col_names=["patient_id", "img_id"],
    ),
    CSVCheck(
        path=PROCESSED_DIR / "scin" / "cleaned_scin_metadata.csv",
        expected_rows=4369,
        expected_cols=73,
    ),
    CSVCheck(
        path=PROCESSED_DIR / "scin" / "missing_fitzpatrick_labels_scin.csv",
        expected_rows=663,
        expected_cols=4,
        expected_col_names=["case_id", "image_1_path", "image_2_path", "image_3_path"],
    ),
]

IMAGE_FOLDER_CHECKS: list[ImageFolderCheck] = [
    ImageFolderCheck(
        path=PROCESSED_DIR / "fitzpatrick_17k" / "images",
        expected_count=8740,
        extensions=[".jpg", ".jpeg"],
    ),
    ImageFolderCheck(
        path=PROCESSED_DIR / "pad_ufes" / "images",
        expected_count=1494,
        extensions=[".png"],
    ),
    ImageFolderCheck(
        path=PROCESSED_DIR / "scin" / "scin_images",
        expected_count=9061,
        extensions=[".png"],
    ),
]


def check_csv(check: CSVCheck) -> list[str]:
    """Run all validations for a single CSV file.

    Args:
        check: The CSVCheck specification to validate.

    Returns:
        A list of failure messages; empty if all checks pass.
    """
    failures: list[str] = []

    if not check.path.exists():
        failures.append(f"{check.path}: file not found")
        return failures

    df = pd.read_csv(check.path)
    actual_rows = len(df)
    actual_cols = len(df.columns)

    if actual_rows != check.expected_rows:
        failures.append(f"{check.path}: expected {check.expected_rows} rows, got {actual_rows}")

    if actual_cols != check.expected_cols:
        failures.append(f"{check.path}: expected {check.expected_cols} columns, got {actual_cols}")

    if check.expected_col_names and list(df.columns) != check.expected_col_names:
        failures.append(f"{check.path}: expected columns {check.expected_col_names}, got {list(df.columns)}")

    return failures


def check_image_folder(check: ImageFolderCheck) -> list[str]:
    """Run all validations for a single image folder.

    Args:
        check: The ImageFolderCheck specification to validate.

    Returns:
        A list of failure messages; empty if all checks pass.
    """
    failures: list[str] = []

    if not check.path.exists():
        failures.append(f"{check.path}: directory not found")
        return failures

    images = [f for f in check.path.iterdir() if f.suffix.lower() in check.extensions]
    actual_count = len(images)

    if actual_count != check.expected_count:
        failures.append(f"{check.path}: expected {check.expected_count} images, got {actual_count}")

    return failures


def check_image_paths_exist(csv_path: Path) -> list[str]:
    """Verify that every image_path entry in a unified CSV resolves to an actual file.

    Args:
        csv_path: Path to the CSV containing an 'image_path' column with paths
            relative to data/processed/.

    Returns:
        A list of failure messages listing any missing files.
    """
    failures: list[str] = []

    if not csv_path.exists():
        return failures

    df = pd.read_csv(csv_path)
    missing = [
        str(PROCESSED_DIR / row["image_path"])
        for _, row in df.iterrows()
        if not (PROCESSED_DIR / row["image_path"]).exists()
    ]

    if missing:
        sample = missing[:5]
        failures.append(f"{csv_path}: {len(missing)} image(s) missing on disk. First {len(sample)}: {sample}")

    return failures


def _print_result(label: str, failures: list[str]) -> bool:
    """Print a formatted check result line.

    Args:
        label: Human-readable label for the check.
        failures: List of failure messages; empty means the check passed.

    Returns:
        True if the check passed, False otherwise.
    """
    status = PASS if not failures else FAIL
    print(f"  [{status}] {label}")
    for msg in failures:
        print(f"         {msg}")
    return not failures


def main() -> None:
    """Run all data validation checks and exit with a non-zero code on any failure."""
    all_passed = True

    print("\nCSV checks")
    print("----------")
    for check in CSV_CHECKS:
        failures = check_csv(check)
        passed = _print_result(str(check.path), failures)
        all_passed = all_passed and passed

    print("\nImage folder checks")
    print("-------------------")
    for check in IMAGE_FOLDER_CHECKS:
        failures = check_image_folder(check)
        passed = _print_result(str(check.path), failures)
        all_passed = all_passed and passed

    print("\nImage path integrity checks")
    print("---------------------------")
    full_csv = PROCESSED_DIR / "full_dataset.csv"
    failures = check_image_paths_exist(full_csv)
    passed = _print_result(f"all image paths in {full_csv} exist on disk", failures)
    all_passed = all_passed and passed

    print()
    if all_passed:
        print("All checks passed.")
    else:
        print("Some checks FAILED. See details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
