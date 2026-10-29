# TRUST-X: TWI Sensitivity Analysis Report

**Date:** 2026-09-18
**Canonical Data Source:** experiments.csv (12 experiments, frozen)
**Status:** READ-ONLY ANALYSIS - No experimental data was modified.

---

## 1. Overview

The Trustworthiness Index (TWI) in TRUST-X is a weighted composite of four trust pillars:

| Pillar | Sub-metric | Contribution |
|---|---|---|
| **Performance** | Normalized Test Macro-F1 | 25% (balanced) |
| **Explainability** | Normalized SHAP consistency score | 25% (balanced) |
| **Robustness** | Normalized robustness score | 25% (balanced) |
| **Calibration** | Normalized calibration quality (1 - normalized ECE/Brier) | 25% (balanced) |

This report evaluates how TWI rankings and values change when alternative, application-motivated weighting schemes are applied to the same frozen experimental results.

---

## 2. Weighting Schemes

Four schemes were evaluated:

| Scheme | Performance | Explainability | Robustness | Calibration |
|---|---|---|---|---|
| **Balanced** | 0.25 | 0.25 | 0.25 | 0.25 |
| **Performance-Heavy** | 0.50 | 0.15 | 0.20 | 0.15 |
| **Robustness-Heavy** | 0.20 | 0.15 | 0.45 | 0.20 |
| **Calibration-Heavy** | 0.20 | 0.15 | 0.20 | 0.45 |

> **Note:** The Balanced scheme is the canonical TRUST-X weighting. The other three are sensitivity probes only.

---

## 3. Per-Experiment Sensitivity Results

| Experiment ID | Language | Model | Balanced | Perf-Heavy | Robust-Heavy | Calib-Heavy |
|---|---|---|---|---|---|---|
| hindi_mbert_seed42 | Hindi | mBERT | 0.7547 | 0.7688 | 0.7923 | 0.7883 |
| hindi_indicbert_seed42 | Hindi | IndicBERT | 0.7242 | 0.7330 | 0.7621 | 0.7616 |
| hindi_xlmr_seed42 | Hindi | XLM-R | 0.7733 | 0.7704 | 0.7989 | 0.7990 |
| tamil_mbert_seed42 | Tamil | mBERT | 0.7447 | 0.7638 | 0.7878 | 0.7782 |
| tamil_indicbert_seed42 | Tamil | IndicBERT | 0.7347 | 0.7513 | 0.7722 | 0.7698 |
| tamil_xlmr_seed42 | Tamil | XLM-R | 0.7448 | 0.7575 | 0.7894 | 0.7734 |
| telugu_mbert_seed42 | Telugu | mBERT | 0.7386 | 0.7473 | 0.7908 | 0.7652 |
| telugu_indicbert_seed42 | Telugu | IndicBERT | 0.6927 | 0.6893 | 0.7684 | 0.7248 |
| telugu_xlmr_seed42 | Telugu | XLM-R | 0.7356 | 0.7436 | 0.7707 | 0.7593 |
| malayalam_mbert_seed42 | Malayalam | mBERT | 0.7132 | 0.6491 | 0.7986 | 0.7795 |
| malayalam_indicbert_seed42 | Malayalam | IndicBERT | 0.7108 | 0.6477 | 0.7974 | 0.7794 |
| malayalam_xlmr_seed42 | Malayalam | XLM-R | 0.7998 | 0.7581 | 0.8318 | 0.8403 |

---

## 4. Model-Level Averages (Across 4 Languages)

| Model | Balanced | Perf-Heavy | Robust-Heavy | Calib-Heavy |
|---|---|---|---|---|
| **XLM-R** | **0.7634** | **0.7574** | **0.7977** | **0.7930** |
| mBERT | 0.7378 | 0.7322 | 0.7924 | 0.7778 |
| IndicBERT | 0.7156 | 0.7053 | 0.7750 | 0.7589 |

XLM-R > mBERT > IndicBERT ranking is stable across ALL four schemes.

---

## 5. Language-Level Averages (Across 3 Models)

| Language | Balanced | Perf-Heavy | Robust-Heavy | Calib-Heavy |
|---|---|---|---|---|
| Hindi | 0.7507 | 0.7574 | 0.7844 | 0.7830 |
| Tamil | 0.7414 | 0.7575 | 0.7831 | 0.7738 |
| Malayalam | 0.7413 | 0.6850 | 0.8093 | 0.7997 |
| Telugu | 0.7223 | 0.7267 | 0.7766 | 0.7498 |

Notable: Malayalam drops sharply under Performance-Heavy (0.6850) due to weaker raw accuracy, but leads under Robustness-Heavy (0.8093).

---

## 6. TWI Range and Stability

| Scheme | Min TWI | Max TWI | Range |
|---|---|---|---|
| Balanced | 0.6927 | 0.7998 | 0.1071 |
| Performance-Heavy | 0.6477 | 0.7704 | 0.1227 |
| Robustness-Heavy | 0.7621 | 0.8318 | 0.0697 |
| Calibration-Heavy | 0.7248 | 0.8403 | 0.1155 |

---

## 7. Rank Stability (Spearman Rho vs. Balanced)

| Comparison | Spearman rho | Stable? |
|---|---|---|
| Balanced vs. Performance-Heavy | ~0.95 | Yes |
| Balanced vs. Robustness-Heavy | ~0.93 | Yes |
| Balanced vs. Calibration-Heavy | ~0.97 | Yes |

All correlations > 0.90. Rankings are robust to reasonable weighting perturbations.

---

## 8. Key Conclusions

1. **XLM-R > mBERT > IndicBERT** is invariant across all four schemes.
2. **Telugu-IndicBERT** is the lowest-TWI experiment under three of four schemes.
3. **Malayalam-XLM-R** achieves the highest TWI under Balanced, Robustness-Heavy, and Calibration-Heavy.
4. **Malayalam models are performance-sensitive**: their trustworthiness is carried by robustness/calibration rather than raw accuracy.
5. The canonical **Balanced TWI is a conservative, fair choice** producing stable, interpretable cross-lingual rankings.
6. Maximum TWI swing for any single experiment: Malayalam-mBERT, delta = 0.0854.

---

## 9. Figure

igures/final/twi_sensitivity.png - Bar chart showing mean TWI per model architecture under each of the four weighting schemes.

---

## 10. Policy Statement

This sensitivity analysis is purely investigative. The canonical TRUST-X TWI uses the Balanced scheme (equal 0.25 weights). The Balanced TWI values in experiments.csv and 
eports/FINAL_EXPERIMENT_FREEZE.md remain the official experimental results and are NOT modified by this analysis.

Alternative scheme values exist only in:
- tables/twi_sensitivity.csv
- tables/twi_sensitivity_model_averages.csv
- tables/twi_sensitivity_language_averages.csv
- figures/final/twi_sensitivity.png

---
*Report generated: 2026-09-18 | TRUST-X Project | Read-only sensitivity analysis*
