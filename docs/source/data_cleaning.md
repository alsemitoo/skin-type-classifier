# Data Cleaning and Validation Report

This document summarizes the data cleaning and validation process performed on the skin type classification datasets on **2026-03-17**.

## Datasets Overview

The project uses three source datasets:

| Dataset | Description | Original Source |
|---------|-------------|-----------------|
| SCIN | Skin Condition Image Network dataset with dermatologist-annotated Fitzpatrick labels | Google Research |
| PAD-UFES | PAD-UFES-20 dataset from Brazil | Federal University of Espirito Santo |
| Fitzpatrick 17k | Fitzpatrick17k dataset with skin condition images | DDI/Groh et al. |

## Validation Process

A comprehensive validation was performed to ensure data integrity:

1. **Image-CSV Consistency**: Verified all images referenced in CSV files exist on disk
2. **Orphan Detection**: Checked for images on disk without corresponding CSV entries
3. **FST Label Validation**: Ensured all Fitzpatrick Skin Type labels are valid (FST1-FST6 or 1-6)
4. **Duplicate Detection**: Identified images referenced by multiple cases
5. **Conflict Detection**: Found images with conflicting FST labels across cases

## Issues Found

### PAD-UFES Dataset
**Status: Clean** - No issues found.

- 1,494 rows with matching images
- All FST labels valid (1-6)

### Fitzpatrick 17k Dataset
**Status: Clean** - No issues found.

- 8,740 rows with matching images
- All FST labels valid (1-6)

### SCIN Dataset
**Status: Issues Found and Resolved**

#### Issue 1: Orphan Images (2 images)

Two images existed on disk without any corresponding entry in the cleaned CSV:

| Filename | Resolution |
|----------|------------|
| `-5441447915979425216.png` | Deleted from disk |
| `-5611615930522319766.png` | Deleted from disk |

These images likely originated from cases that were filtered out during initial cleaning but whose images were incorrectly copied.

#### Issue 2: Duplicate Image References (14 images, 20 references)

The SCIN dataset contained 14 unique images that were referenced by multiple case IDs. This suggests the same image was submitted for multiple consultations.

#### Issue 3: Conflicting FST Labels (2 images, 5 cases)

Two images had conflicting Fitzpatrick Skin Type labels assigned by different dermatologists across different cases:

**Image 1:** `-9111307368692396870.png`

| Case ID | FST Label |
|---------|-----------|
| -1712619350051039848 | FST4 |
| -9099005241992076854 | FST4 |
| 2311974363421871372 | **FST3** (conflict) |

**Image 2:** `4166674568332648445.png`

| Case ID | FST Label |
|---------|-----------|
| -2257423882813548775 | FST2 |
| 4789108431219423909 | **FST1** (conflict) |

**Resolution:** All 5 cases referencing these conflicting images were removed from the dataset, and the 2 conflicting images were deleted from disk.

## Changes Applied

| Action | Count | Details |
|--------|-------|---------|
| Orphan images deleted | 2 | Images with no CSV entry |
| Cases removed | 5 | Cases with conflicting FST labels |
| Conflict images deleted | 2 | Images with inconsistent labels |
| **Total images removed** | **4** | |
| **Total cases removed** | **5** | |

## Final Dataset Statistics

### Source Datasets

| Dataset | Cases/Rows | Images | Status |
|---------|------------|--------|--------|
| SCIN | 4,364 cases | 9,057 images | Validated |
| PAD-UFES | 1,494 rows | 1,494 images | Validated |
| Fitzpatrick 17k | 8,740 rows | 8,740 images | Validated |

### Consolidated Datasets

| File | Total Rows | Sources |
|------|------------|---------|
| `full_dataset.csv` | 19,291 | SCIN + PAD-UFES + Fitzpatrick 17k |
| `dataset_without_fitzpatrick.csv` | 10,551 | SCIN + PAD-UFES |

### FST Distribution (full_dataset.csv)

| Fitzpatrick Type | Count | Percentage |
|------------------|-------|------------|
| 1 | 3,213 | 16.7% |
| 2 | 7,155 | 37.1% |
| 3 | 4,716 | 24.4% |
| 4 | 2,606 | 13.5% |
| 5 | 1,242 | 6.4% |
| 6 | 359 | 1.9% |

## Validation Checklist

After cleaning, the following validations all pass:

- [x] No orphan images on disk (images without CSV entries)
- [x] No missing images (CSV entries without corresponding images)
- [x] No duplicate `image_path` values in `full_dataset.csv`
- [x] No conflicting FST labels for the same image
- [x] All FST values are valid integers (1-6)
- [x] All image files exist and are accessible

## Data Quality Notes

### SCIN Dataset Considerations

The SCIN dataset uses dermatologist-annotated Fitzpatrick labels from up to 3 annotators per case:
- `dermatologist_fitzpatrick_skin_type_label_1`
- `dermatologist_fitzpatrick_skin_type_label_2`
- `dermatologist_fitzpatrick_skin_type_label_3`

The processing pipeline (`_process_scin` in `build_datasets.py`) uses the **first available** dermatologist label. This is appropriate since:
1. Not all cases have multiple annotations
2. The first annotator's assessment is typically the primary evaluation

### Handling of Multi-Image Cases

SCIN cases can have up to 3 images per case. The pipeline "explodes" these into separate rows, each inheriting the case's FST label. Deduplication ensures each unique image appears only once in the final dataset.

## Reproducibility

To re-run the data cleaning and validation:

```bash
# Clean individual datasets (if raw data available)
uv run python src/skin_type_classifier/data/clean_scin.py
uv run python src/skin_type_classifier/data/clean_pad_ufes.py
uv run python src/skin_type_classifier/data/clean_fitzpatrick_17k.py

# Build consolidated datasets
uv run python src/skin_type_classifier/data/build_datasets.py
```

## File Locations

```
data/processed/
├── full_dataset.csv                          # Main consolidated dataset
├── dataset_without_fitzpatrick.csv           # Without Fitzpatrick 17k
├── scin/
│   ├── cleaned_scin_metadata.csv             # Cleaned SCIN metadata
│   ├── missing_fitzpatrick_labels_scin.csv   # Cases without FST labels
│   └── scin_images/                          # 9,057 images
├── pad_ufes/
│   ├── cleaned_pad_ufes_metadata.csv
│   ├── missing_fitzpatrick_labels_pad_ufes.csv
│   └── images/                               # 1,494 images
└── fitzpatrick_17k/
    ├── cleaned_fitzpatrick_17k_metadata.csv
    ├── missing_fitzpatrick_labels_fitzpatrick_17k.csv
    ├── wrong_entries_fitzpatrick_17k.csv
    └── images/                               # 8,740 images
```
