"""Clean the Fitzpatrick 17k dataset by filtering cases with valid Fitzpatrick skin type labels."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path("data/raw/fitzpatrick_17k")
PROCESSED_DIR = Path("data/processed/fitzpatrick_17k")
OUTPUT_IMAGE_DIR = PROCESSED_DIR / "images"

METADATA_CSV = RAW_DIR / "Fitzpatrick17k-C.csv"
FITZ_COL = "fitzpatrick"
URL_COL = "url"
FILENAME_COL = "new_img_name"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; skin-type-classifier/0.1)"}
DOWNLOAD_TIMEOUT = 25
MAX_WORKERS = min(32, (os.cpu_count() or 4) * 4)


def has_valid_fitzpatrick(value: int) -> bool:
    """Check if the Fitzpatrick label is valid (1-6)."""
    return value >= 1


def download_image(url: str, dest: Path) -> bool:
    """Download an image from a URL and save it to the destination path.

    Args:
        url: The URL to download the image from.
        dest: The destination file path.

    Returns:
        True if the download was successful, False otherwise.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=DOWNLOAD_TIMEOUT, stream=True)
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except (requests.RequestException, OSError) as e:
        print(f"  FAILED to download {url}: {e}")
        return False


def clean_fitzpatrick_17k() -> None:
    """Run the full Fitzpatrick 17k cleaning pipeline."""
    df = pd.read_csv(METADATA_CSV)

    has_label = df[FITZ_COL].apply(has_valid_fitzpatrick)
    valid = df[has_label].copy()
    invalid = df[~has_label].copy()

    print(f"Total cases: {len(df)}")
    print(f"Valid cases (Fitzpatrick 1-6): {len(valid)}")
    print(f"Invalid cases (no valid label): {len(invalid)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    missing_path = PROCESSED_DIR / "missing_fitzpatrick_labels_fitzpatrick_17k.csv"
    invalid[[FILENAME_COL, FITZ_COL, URL_COL]].to_csv(missing_path, index=False)
    print(f"Wrote {len(invalid)} invalid cases to {missing_path}")

    has_url = valid[URL_COL].notna() & (valid[URL_COL].astype(str).str.strip() != "")
    downloadable = valid[has_url].copy()
    no_url = valid[~has_url].copy()

    if len(no_url) > 0:
        print(f"WARNING: {len(no_url)} valid cases have no URL and cannot be downloaded")

    tasks = []
    skipped = 0
    for idx, row in downloadable.iterrows():
        dest = OUTPUT_IMAGE_DIR / row[FILENAME_COL]
        if dest.exists():
            skipped += 1
            continue
        url = str(row[URL_COL]).strip()
        tasks.append((idx, url, dest))

    print(f"Skipping {skipped} already-downloaded images, {len(tasks)} to download")
    print(f"Using {MAX_WORKERS} concurrent workers")

    downloaded, failed = 0, 0
    failed_rows = []
    total = len(tasks)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_image, url, dest): idx for idx, url, dest in tasks}
        for future in as_completed(futures):
            idx = futures[future]
            if future.result():
                downloaded += 1
            else:
                failed += 1
                failed_rows.append(idx)
                dest = OUTPUT_IMAGE_DIR / downloadable.loc[idx, FILENAME_COL]
                if dest.exists():
                    dest.unlink()

            done = downloaded + failed
            if done % 100 == 0:
                print(f"Progress: {done}/{total} — downloaded {downloaded}, failed {failed}")

    print(f"Done: {downloaded} downloaded, {skipped} already present, {failed} failed")

    all_excluded_indices = invalid.index.tolist() + no_url.index.tolist() + failed_rows
    cleaned = df.drop(index=all_excluded_indices)

    cleaned_path = PROCESSED_DIR / "cleaned_fitzpatrick_17k_metadata.csv"
    cleaned.to_csv(cleaned_path, index=False)
    print(f"Wrote {len(cleaned)} valid+downloaded cases to {cleaned_path}")

    if failed_rows or len(no_url) > 0:
        wrong_entries = pd.concat([invalid, no_url, df.loc[failed_rows]])
        wrong_path = PROCESSED_DIR / "wrong_entries_fitzpatrick_17k.csv"
        wrong_entries.to_csv(wrong_path, index=False)
        print(f"Wrote {len(wrong_entries)} wrong/excluded entries to {wrong_path}")


if __name__ == "__main__":
    clean_fitzpatrick_17k()
