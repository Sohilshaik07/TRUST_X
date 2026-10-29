# HECM Hindi Dataset Preprocessing & Audit Report

**Project:** TRUST-X: An Integrated Trustworthiness Evaluation Framework for Transformer-Based Multilingual Code-Mixed Hate and Offensive Speech Detection  
**Date:** 2026-09-17  
**Auditor/Processor:** Antigravity IDE  
**Status:** Preprocessing Complete — Ready for Verification (No Models Trained)

---

## 1. Source Files & Integrity Preservation

| File | Path | Status | SHA-256 / Checksum State |
|---|---|---|---|
| Text File | `Data/raw/hindi/HECM.txt` | **Unmodified (Read-Only)** | 1,453,936 bytes (9,357 lines) |
| Label File | `Data/raw/hindi/HECM_Class.txt` | **Unmodified (Read-Only)** | 18,714 bytes (9,357 lines) |

The raw author-provided source files remain byte-for-byte identical to their initial state. No cleaning, in-place editing, or renaming was performed on raw files.

---

## 2. Author-Confirmed Label Semantics

| Raw Label Code | Official Semantic Meaning | Source of Confirmation | TRUST-X Internal Label | Human-Readable Label (`original_label`) |
|---|---|---|---|---|
| `0` | **Hateful** | Direct confirmation from HECM author Prashant K. | `0` | `Hateful` |
| `1` | **Non-hateful** | Direct confirmation from HECM author Prashant K. | `1` | `Non-hateful` |

> [!IMPORTANT]
> The label mapping was **officially confirmed directly by dataset author Prashant K.** It was not inferred heuristically or from sample inspection. The numeric labels `0` and `1` are preserved as primary numeric labels.

---

## 3. Original Dataset Size & Duplicate Analysis

| Metric | Count | Details / Methodology |
|---|---|---|
| Total Raw Aligned Lines | 9,357 | Exactly aligned line-by-line between HECM.txt and HECM_Class.txt |
| Unique Raw Texts | 9,239 | Whitespace-normalized text values |
| Total Duplicate Text Groups | 118 | Texts appearing more than once in raw files (all appear exactly 2 times) |
| Total Duplicate Rows | 236 | 118 groups × 2 occurrences |
| Duplicate Excess Rows | 118 | Redundant rows across all duplicate groups |
| Conflicting-Label Groups | **25** | Same exact text associated with both `0` and `1` |
| Rows in Conflicting Groups | **50** | 25 rows with label `0`, 25 rows with label `1` |
| Non-Conflicting Duplicate Groups | 93 | Exact duplicate texts with identical labels |
| Non-Conflicting Excess Rows | 93 | Redundant copies of valid text-label pairs |

---

## 4. Conflicting-Label Handling & Modeling Dataset Size

"The 25 exact-text groups with conflicting labels were excluded from the modeling dataset because their correct class could not be established from the provided files. The original author-provided files were preserved unchanged."

### Dataset Reduction Flow:
1. **Raw pairs:** 9,357
2. **Exclusion of 25 conflicting groups (50 rows):** 9,357 - 50 = 9,307 rows
3. **Deduplication of 93 non-conflicting duplicates (93 excess rows):** 9,307 - 93 = **9,214 unique modeling samples**

---

## 5. Preprocessing Policy

In accordance with TRUST-X principles (evaluating robustness, calibration, explainability, and linguistic phenomena in multilingual code-mixed speech), **conservative, non-destructive preprocessing** was enforced:
- **Preserved:** Casing, punctuation, Hindi Devanagari script, Romanized Hindi (Hinglish), English words, emojis, @mentions, #hashtags, URLs, and repeated characters.
- **Applied:** Standard whitespace normalization only (`re.sub(r'\s+', ' ', text).strip()`).
- **Forbidden:** No lowercasing, stopword removal, stemming, lemmatization, transliteration, machine translation, or profanity masking.

---

## 6. Train / Dev / Test Split Strategy

Because the HECM dataset has no official pre-split benchmark files, a stratified split was generated:
- **Ratio:** 70% Train / 15% Validation (Dev) / 15% Test
- **Random Seed:** `42` (`scikit-learn` `train_test_split`)
- **Stratification:** Stratified by the binary HECM label (`0` = Hateful, `1` = Non-hateful) to preserve the exact class imbalance across all splits.
- **Split Timing:** Executed strictly **after** duplicate exclusion and deduplication to prevent data leakage.

### Split Size & Class Distribution

| Split | Total Samples | Label 0 (Hateful) | % Hateful | Label 1 (Non-hateful) | % Non-hateful | Imbalance Ratio (0 / 1) |
|---|---|---|---|---|---|---|
| **Cleaned Full** | 9,214 | 2,792 | 30.30% | 6,422 | 69.70% | 0.4348 |
| **Train** | 6,449 | 1,954 | 30.30% | 4,495 | 69.70% | 0.4347 |
| **Dev** | 1,382 | 419 | 30.32% | 963 | 69.68% | 0.4351 |
| **Test** | 1,383 | 419 | 30.30% | 964 | 69.70% | 0.4346 |

---

## 7. Data Leakage Audit

| Leakage Audit Level | Train vs Dev | Train vs Test | Dev vs Test | Audit Status |
|---|---|---|---|---|
| **Exact Text Overlap** | **0** | **0** | **0** | **PASS — Zero Leakage** |
| **Normalized Text Overlap** (NFKC + lowercase + whitespace) | **1** | **0** | **0** | **Documented (Case Difference)** |

### Note on the Single Normalized Overlap:
The single normalized overlap between Train and Dev corresponds to:
- **Train sample:** `"Desh ki haalat dekho, pehle log NAUKRI maangte the. Ab CHAUKRI maangte hai. Yahi hai \"Modi effect\". #modi_girlfriend_do"`
- **Dev sample:** `"Desh ki haalat dekho, pehle log naukri maangte the. Ab chaukri maangte hai. yahi hai \"Modi effect\". #modi_girlfriend_do"`

These texts differ in casing (`NAUKRI`/`CHAUKRI` vs `naukri`/`chaukri`). Because TRUST-X preserves natural capitalization as a behavioral signal for code-mixed speech, and heuristic near-duplicate removal was not instructed, both samples were retained.

---

## 8. Text Statistics (Before vs After Preprocessing)

| Metric | Raw HECM (9,357) | Cleaned Modeling HECM (9,214) | Train (6,449) | Dev (1,382) | Test (1,383) |
|---|---|---|---|---|---|
| Min Characters | 5 | 5 | 5 | 5 | 5 |
| Max Characters | 744 | 744 | 744 | 498 | 677 |
| Mean Characters | 147.58 | 148.18 | 148.24 | 148.33 | 147.74 |
| Median Characters | 131.0 | 132.0 | 132.0 | 132.0 | 131.0 |
| Min Words | 1 | 1 | 1 | 1 | 1 |
| Max Words | 89 | 89 | 89 | 83 | 87 |
| Mean Words | 24.46 | 24.59 | 24.57 | 24.69 | 24.58 |
| Median Words | 21.0 | 21.0 | 21.0 | 21.0 | 21.0 |
| 95th Percentile Words | 54.0 | 54.0 | 54.0 | 54.0 | 54.0 |

---

## 9. Specific HECM Dataset Characteristics Preserved

- **Predominantly Latin-script / Hinglish:** ~83.5% Latin-only, ~16.2% mixed Devanagari-Latin, ~0.2% pure Devanagari.
- **Social Media Structure:** ~72.1% contain @mentions, ~33.4% contain hashtags, ~15.6% contain URLs.
- **Rich Expression:** ~19.5% contain emojis, ~10.9% contain repeated character sequences.

---

## 10. Processed Output Dataset Schema

Files created under `Data/processed/hindi/`:
- `train.csv` (6,449 rows)
- `dev.csv` (1,382 rows)
- `test.csv` (1,383 rows)

**Columns:**
1. `id`: `hi_train_00000`, `hi_dev_00000`, `hi_test_00000`
2. `text`: Cleaned text with original linguistic characteristics preserved
3. `label`: Integer `0` (Hateful) or `1` (Non-hateful)
4. `original_label`: Human-readable string (`Hateful` or `Non-hateful`)
5. `language`: `hindi`
6. `task`: `hate_speech`
7. `split`: `train`, `dev`, `test`

---

## 11. Reproducibility & Artifact Manifest

- **Preprocessing Script:** `scripts/prepare_hecm_hindi.py` (Deterministic, Seed 42)
- **Supporting Tables Created:**
  - `tables/hecm_hindi_split_distribution.csv`
  - `tables/hecm_hindi_duplicate_audit.csv`
  - `tables/hecm_hindi_preprocessing_stats.csv`
- **Report Created:** `reports/hecm_hindi_preprocessing_report.md`

---

## 12. Pipeline Safeguards & Limitations

1. **No Training Executed:** Zero model training has occurred.
2. **Existing Data Intact:** Telugu, Tamil, and Malayalam processed datasets and experimental results remain untouched.
3. **Limitation:** Latin-script tokens represent Romanized Hindi/Hinglish without token-level LID annotation.
