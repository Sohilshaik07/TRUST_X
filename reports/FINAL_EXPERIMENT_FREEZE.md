# TRUST-X Final Experiment Freeze & Checkpoint Report

**Status:** FROZEN  
**Date:** 2026-09-18  
**Scope:** Complete experimental results, tables, and visualization artifacts for all 12 TRUST-X experiments across 4 Indic languages and 3 multilingual architectures.

---

> [!IMPORTANT]
> **FREEZE NOTICE:**
> The experimental results, metric values, TWI scores, and artifact outputs documented herein are formally **FROZEN**. No model retraining, configuration changes, dataset modifications, metric recalculations, or experimental code alterations should be performed unless a reproducible methodological error is discovered and explicitly verified.

---

## 1. Verification Checklist

| # | Check Item | Requirement | Observed Status | Verdict |
|---|---|---|---|:---:|
| 1 | Experiment Completion Status | All 12 experiment rows marked `COMPLETED` | 12 of 12 `COMPLETED` | **PASS** |
| 2 | Unique Experiments in `experiments.csv` | 12 unique `experiment_id` entries | 12 unique IDs, zero duplicates | **PASS** |
| 3 | Factorial Coverage ($4 \times 3$) | 4 Languages $\times$ 3 Models | Hindi, Malayalam, Tamil, Telugu $\times$ IndicBERT, mBERT, XLM-R | **PASS** |
| 4 | Metric Completeness | Zero null/NaN values across all critical metrics | 0 missing in `test_accuracy`, `test_macro_f1`, `ece`, `brier`, `robustness_score`, `explainability_score`, `twi` | **PASS** |
| 5 | Summary Tables Generated | Presence of 3 canonical CSV tables in `tables/` | `final_results.csv`, `model_averages.csv`, `language_averages.csv` confirmed | **PASS** |
| 6 | Publication Figures Generated | Presence of 5 canonical figures in `figures/final/` | `macro_f1_comparison.png`, `twi_comparison.png`, `trust_pillars_comparison.png`, `accuracy_vs_macro_f1.png`, `macro_f1_vs_twi.png` confirmed | **PASS** |

---

## 2. Master Experiment Freeze Table (`experiments.csv`)

| Experiment ID | Language | Model | Seed | Test Acc | Test Macro-F1 | ECE | Brier | Robustness | Explainability | TWI | Status |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `telugu_mbert_seed42` | Telugu | mBERT | 42 | 0.7399 | 0.7399 | 0.0539 | 0.1734 | 0.9023 | 0.5127 | 0.7386 | `COMPLETED` |
| `telugu_indicbert_seed42` | Telugu | IndicBERT | 42 | 0.6520 | 0.6476 | 0.0196 | 0.2244 | 0.9404 | 0.4171 | 0.6927 | `COMPLETED` |
| `tamil_mbert_seed42` | Tamil | mBERT | 42 | 0.8370 | 0.7815 | 0.0884 | 0.1263 | 0.8680 | 0.5000 | 0.7448 | `COMPLETED` |
| `tamil_indicbert_seed42` | Tamil | IndicBERT | 42 | 0.8166 | 0.7672 | 0.0707 | 0.1359 | 0.8387 | 0.5041 | 0.7347 | `COMPLETED` |
| `tamil_xlmr_seed42` | Tamil | XLM-R | 42 | 0.8055 | 0.7619 | 0.0886 | 0.1406 | 0.8790 | 0.5231 | 0.7448 | `COMPLETED` |
| `malayalam_indicbert_seed42` | Malayalam | IndicBERT | 42 | 0.9572 | 0.4891 | 0.0553 | 0.0441 | 1.0000 | 0.4261 | 0.7108 | `COMPLETED` |
| `telugu_xlmr_seed42` | Telugu | XLM-R | 42 | 0.7432 | 0.7432 | 0.0563 | 0.1763 | 0.8412 | 0.5625 | 0.7356 | `COMPLETED` |
| `malayalam_mbert_seed42` | Malayalam | mBERT | 42 | 0.9572 | 0.4891 | 0.0627 | 0.0449 | 1.0000 | 0.4398 | 0.7131 | `COMPLETED` |
| `malayalam_xlmr_seed42` | Malayalam | XLM-R | 42 | 0.9528 | 0.6655 | 0.0315 | 0.0447 | 0.9054 | 0.6887 | 0.7998 | `COMPLETED` |
| `hindi_mbert_seed42` | Hindi | mBERT | 42 | 0.8228 | 0.7797 | 0.0478 | 0.1313 | 0.8608 | 0.5336 | 0.7547 | `COMPLETED` |
| `hindi_indicbert_seed42` | Hindi | IndicBERT | 42 | 0.7773 | 0.7342 | 0.0272 | 0.1570 | 0.8313 | 0.5017 | 0.7241 | `COMPLETED` |
| `hindi_xlmr_seed42` | Hindi | XLM-R | 42 | 0.7990 | 0.7541 | 0.0140 | 0.1435 | 0.8491 | 0.6405 | 0.7733 | `COMPLETED` |

---

## 3. Verified Artifact Registry

### A. Core Data & Experiments
* `experiments.csv` (1,871 bytes) — Canonical experiment log containing all 12 finalized evaluations.

### B. Summary Tables (`tables/`)
* `tables/final_results.csv` (1,353 bytes) — Full evaluation table across all metrics and derived calibration scores.
* `tables/model_averages.csv` (296 bytes) — Cross-lingual aggregate metrics for mBERT, IndicBERT, and XLM-R.
* `tables/language_averages.csv` (363 bytes) — Cross-model aggregate metrics for Telugu, Tamil, Malayalam, and Hindi.

### C. Visualizations (`figures/final/`)
* `figures/final/macro_f1_comparison.png` (118,449 bytes) — Grouped bar chart comparing Macro-F1 across 4 languages and 3 models.
* `figures/final/twi_comparison.png` (134,001 bytes) — Grouped bar chart of holistic Trustworthiness Index (TWI) values.
* `figures/final/trust_pillars_comparison.png` (300,597 bytes) — Multi-pillar radar chart mapping Performance, Explainability, Robustness, and Calibration.
* `figures/final/accuracy_vs_macro_f1.png` (222,400 bytes) — Publication-grade scatter plot with intelligent offset annotations and leader lines.
* `figures/final/macro_f1_vs_twi.png` (182,834 bytes) — Performance vs. Holistic Trustworthiness scatter comparison.

---

## 4. Methodological Summary
* **Languages Covered:** Hindi (HECM), Telugu (HECM), Tamil (HECM), Malayalam (HECM).
* **Models Evaluated:** `bert-base-multilingual-cased` (mBERT), `ai4bharat/indic-bert` (IndicBERT), `xlm-roberta-base` (XLM-R).
* **Random Seed:** Fixed at `42` across all training, validation, testing, SHAP, and perturbation pipelines.
* **Evaluation Pillars:**
  1. Performance ($F1_{\text{macro}}$)
  2. Robustness ($1 - \Delta F1_{\text{perturb}}$)
  3. Calibration ($0.5(1 - \text{ECE}) + 0.5 \max(0, 1 - 2\text{Brier})$)
  4. Explainability (SHAP token attribution compactness)
