# TRUST-X Review 2 Presentation Audit Report

**Date:** 2026-10-29  
**Audited File:** `reports/TRUST_X_Review_2_Presentation.pptx` (27 slides)  
**Corrected File:** `reports/TRUST_X_Review_2_Presentation_Corrected.pptx`  
**Backup of Original:** `reports/TRUST_X_Review_2_Presentation_ORIGINAL_BACKUP.pptx`  
**Source of Truth:** Project source code (`src/`), canonical results (`tables/final_results.csv`, `experiments.csv`), per-experiment JSON files (`checkpoints/*/results_*.json`), and `tables/processed_datasets_summary.csv`

---

## Summary

**27 evidence-based corrections** were applied across **13 slides**. The corrected PPT passes all 22 verification checks with 0 residual old-text issues. No new layout or overflow problems were introduced. The presentation's design, theme, typography, and visual style are fully preserved.

---

## Corrections Applied

### Slide 5 — Dataset Overview & Linguistic Characteristics

| # | Issue | Old Value | Corrected Value | Source |
|---|-------|-----------|-----------------|--------|
| 1 | Telugu class distribution | ~65% non-hate, ~35% hate | ~52% non-hate, ~48% hate (~1.07:1) | `processed_datasets_summary.csv`: harmful_pct = 48.44% |
| 2 | Tamil class distribution | ~81% non-offensive, ~19% offensive | ~75% non-offensive, ~25% offensive | `processed_datasets_summary.csv`: harmful_pct = 24.6% |

### Slide 6 — Raw Dataset Structures & Ingestion Protocols

| # | Issue | Old Claim | Corrected Statement | Source |
|---|-------|-----------|---------------------|--------|
| 3 | Hindi label inversion | "Our loader performed explicit inversion to enforce global canonical harmony: 0 = Non-Toxic, 1 = Toxic" | "The processed Hindi data preserves this original mapping. Macro-averaged metrics are unaffected by label direction." | `scripts/prepare_hecm_hindi.py` line 120: `df_clean["label"] = df_clean["raw_label"].astype(int)` — labels are NOT inverted |
| 4 | Heading tone | "Critical Label Polarity Anomaly" | "Label Polarity Note" | Factual framing correction |
| 5 | Hindi distribution | "0 = Non-Hate (53%), 1 = Hate (47%) [Inverted!]" | "0 = Hateful (30.3%), 1 = Non-Hateful (69.7%) [Original polarity preserved]" | `tables/hecm_hindi_split_distribution.csv`: hateful_label_0 = 30.3%, non_hateful_label_1 = 69.7% |

### Slide 7 — Data Preprocessing & Integrity Protocols

| # | Issue | Old Claim | Corrected Statement | Source |
|---|-------|-----------|---------------------|--------|
| 6 | URL/mention stripping | "Strips web URLs (https?://\S+) and user mentions (@username)" | "Normalizes multiple whitespace characters into a single clean space and strips leading/trailing whitespace" | `src/data/preprocessing.py` `clean_text()` function: only `re.sub(r'\s+', ' ', text).strip()` |
| 7 | HTML entity decoding | "Decodes HTML entities (&amp; → &, &lt; → <)" | "Preserves all Unicode scripts, emojis, and punctuation" | No HTML decoding exists in the codebase |

### Slide 8 — Final Processed Data Schema & Audit

| # | Issue | Old Claim | Corrected Statement | Source |
|---|-------|-----------|---------------------|--------|
| 8 | Storage format | "parquet/CSV formats with MD5 checksum verification" | "CSV format with deterministic file persistence" | All preprocessing scripts save as `.csv` only; no parquet or MD5 logic exists |

### Slide 9 — Dataset Distribution & Class Imbalance Analysis

| # | Issue | Old Value | Corrected Value | Source |
|---|-------|-----------|-----------------|--------|
| 9 | Telugu class skew | 65% Non-Hate vs 35% Hate (~1.9:1) | 52% Non-Hate vs 48% Hate (~1.07:1, near-balanced) | `processed_datasets_summary.csv` |
| 10 | Tamil split counts | Train: 29,315 / Val: 6,282 / Test: 6,282 | Train: 33,431 / Dev: 4,216 / Test: 4,232 | `processed_datasets_summary.csv` |
| 11 | Tamil class skew | 81% Non-Offensive vs 19% Offensive (~4.3:1) | 75% Non-Offensive vs 25% Offensive (~3:1) | `processed_datasets_summary.csv` |
| 12 | Malayalam split counts | Train: 10,137 / Val: 2,172 / Test: 2,173 | Train: 10,802 / Dev: 1,836 / Test: 1,844 | `processed_datasets_summary.csv` |
| 13 | Hindi class skew | 53% Non-Hate vs 47% Hate (~1.1:1 balanced) | 70% Non-Hate vs 30% Hate (~2.3:1) | `tables/hecm_hindi_split_distribution.csv` |

### Slide 10 — Input & Output Transformation Pipeline

| # | Issue | Old Value | Corrected Value | Source |
|---|-------|-----------|-----------------|--------|
| 14 | Dropout rate | p = 0.1 | p = 0.2 | `src/models/model_factory.py` line 79: `classifier_dropout=dropout` where `dropout=0.2` |

### Slide 12 — Complete Neural Network Architecture

| # | Issue | Old Value | Corrected Value | Source |
|---|-------|-----------|-----------------|--------|
| 15 | Dropout rate | p = 0.1 | p = 0.2 | Same as above |

### Slide 17 — Training, Validation, and Testing Pipeline

| # | Issue | Old Value | Corrected Value | Source |
|---|-------|-----------|-----------------|--------|
| 16 | Tamil train count | 29,315 | 33,431 | `processed_datasets_summary.csv` |
| 17 | Malayalam train count | 10,137 | 10,802 | `processed_datasets_summary.csv` |

### Slide 21 — The TRUST-X Multi-Pillar Framework

| # | Issue | Old Formula | Corrected Formula | Source |
|---|-------|-------------|-------------------|--------|
| 18 | Calibration Score | C = 1.0 - ECE | C = 0.5*(1-ECE) + 0.5*max(0, 1-2*Brier) | `src/twi/twi.py` `normalize_calibration_score()` lines 35-44 |

### Slide 23 — Pillar 3: Linguistic Robustness

| # | Issue | Old Claim | Corrected Statement | Source |
|---|-------|-----------|---------------------|--------|
| 19 | Composite perturbation | "chains all 5 operators simultaneously" | "chains 3 operators (char_repetition, leet_substitution, char_deletion)" | `src/robustness/perturbations.py` `apply_composite_perturbation()` lines 102-111 |

### Slide 24 — Pillar 4: Model Confidence Calibration

| # | Issue | Old Content | Corrected Content | Source |
|---|-------|-------------|-------------------|--------|
| 20 | Calibration Score formula | "TRUST-X Calibration Pillar Score (C) = 1.0 - ECE" | Full composite formula including Brier | `src/twi/twi.py` |
| 21 | Editorial language | "exceptional calibration" | "the lowest ECE among all 12 experiments" | Factual: Hindi XLM-R ECE=0.0140 is numerically lowest |
| 22 | Brier Score framing | "Brier Score Verification: Evaluated mean squared probability error to validate probabilistic sharpness." | Brier is integrated into the Calibration Score formula alongside ECE, not just verification | `src/twi/twi.py` line 43 |
| 23 | Temperature scaling | "T* in [1.05, 1.35]" with implication it was applied | Implemented but all reported metrics use raw (unscaled) softmax probabilities; specific T* range unverified | `src/calibration/calibration.py` has `ModelWithTemperature` class; no evidence T* was applied to final test metrics |

### Slide 25 — Composite TWI & Sensitivity

| # | Issue | Old Weights | Corrected Weights | Source |
|---|-------|-------------|-------------------|--------|
| 24 | Performance-Heavy profile | (0.50, 0.20, 0.15, 0.15) | (0.50, 0.15, 0.20, 0.15) | `src/twi/twi.py` line 23 |
| 25 | Robustness-Heavy profile | (0.20, 0.15, 0.50, 0.15) | (0.20, 0.15, 0.45, 0.20) | `src/twi/twi.py` line 27 |
| 26 | Profile name | "Accountability-Focused" | "Calibration-Heavy" | `src/twi/twi.py` line 29: key is `"calibration_heavy"` |
| 27 | Calibration-Heavy weights | (0.20, 0.40, 0.15, 0.25) | (0.20, 0.15, 0.20, 0.45) | `src/twi/twi.py` line 31 |

---

## Claims Verified as Correct

The following elements were audited and confirmed accurate against source code and canonical data:

- **Slide 1:** Title, team members, supervisor, institution
- **Slide 2:** Problem statement and research question framing
- **Slide 3:** 7 research objectives (totaling ~69,500 samples confirmed: 3,941+41,879+14,482+9,214 = 69,516)
- **Slide 4:** Four-tier architecture pipeline description
- **Slide 10-13:** Attention formula, transformer layer descriptions, multi-head attention mechanics
- **Slide 14:** Model registry names and checkpoints (mBERT = `bert-base-multilingual-cased`, IndicBERT = `ai4bharat/indic-bert`, XLM-R = `xlm-roberta-base`)
- **Slide 15:** Training hyperparameters (LR=2e-5, AdamW, weight_decay=0.01, warmup=10%, batch=16, max_length=128, epochs=3, seed=42) — all match config YAML files
- **Slide 15:** Class weight formula `w_c = N / (2 * N_c)` — matches `compute_class_weights()` in `trainer.py`
- **Slide 16:** Training loop description (gradient clipping at 1.0 confirmed in `trainer.py` line 166)
- **Slide 18:** 12 factorial experiments (4 languages x 3 models)
- **Slide 19:** All 12 accuracy and Macro-F1 values match `final_results.csv` exactly
- **Slide 20:** Malayalam mBERT/IndicBERT accuracy-F1 paradox (95.72% acc, 0.4891 F1) confirmed
- **Slide 22:** SHAP compactness formula (top 20% attribution concentration) matches `shap_analysis.py`
- **Slide 22:** All explainability scores match `final_results.csv`
- **Slide 23:** Robustness Score formula `R = 0.6*Retention + 0.4*(1-FlipRate)` matches `twi.py`
- **Slide 23:** All robustness score values match `final_results.csv`
- **Slide 25:** Balanced profile (0.25, 0.25, 0.25, 0.25) is correct
- **Slide 25:** TWI formula `TWI = w_P*P + w_E*E + w_R*R + w_C*C` matches `twi.py`
- **Slide 25:** Cross-model TWI averages (XLM-R 0.7634, mBERT 0.7378, IndicBERT 0.7156) match `model_averages.csv`
- **Slide 26:** All 12 experiment results in the master table match `final_results.csv` exactly
- **Slide 27:** Streamlit application description aligns with `app/` directory structure

---

## Remaining Unverified Claims (Flagged)

These claims could not be definitively confirmed or denied from the available source code and data:

1. **Model parameter counts (Slide 14):**
   - mBERT: 178M — approximately correct for `bert-base-multilingual-cased` (actual ~179M due to large vocab)
   - IndicBERT: 87M — flagged as potentially inaccurate. `ai4bharat/indic-bert` uses ALBERT architecture with parameter sharing; actual unique parameter count may differ
   - XLM-R: 250M — approximately correct for `xlm-roberta-base` (actual ~278M)

2. **Total GPU time (Slide 18):** "~14.5 GPU hours" — cannot verify without re-running all experiments; individual `train_duration_sec` values exist in per-experiment JSONs

3. **Software versions (Slide 18):** "PyTorch 2.1+, HuggingFace Transformers 4.36+" — cannot verify the exact versions used at training time

4. **Hindi label semantics impact:** While the label inversion claim is corrected, the fact that Hindi uses 0=Hateful/1=Non-hateful while other languages use 0=Non-toxic/1=Toxic means that SHAP attributions and Brier Scores are computed for different semantic classes across languages. This is a methodology note, not an error in the computed values.

---

## Pre-Existing Layout Notes

Two images on Slides 4 and 12 (architecture diagrams) extend below the visible slide boundary. This is a pre-existing condition in the original PPT and does not affect the presentation display — PowerPoint crops images to slide boundaries during presentation mode.

---

## Verification Summary

| Check | Result |
|-------|--------|
| Total slides | 27 (unchanged) |
| Corrections applied | 27 |
| Verification checks passed | 22/22 |
| Residual old text found | 0 |
| New layout issues introduced | 0 |
| Corrected PPT opens successfully | Yes |
