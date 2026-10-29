# TRUST-X Pilot Audit Report
## Experiment: `telugu_mbert_seed42`
**Audit Date:** 2026-09-16  
**Auditor:** Code-based inspection — no experiment outputs modified, no training re-run.  
**Sources Inspected:** All 11 source modules, YAML config, processed data files, formula verification scripts.

---

## A. VERIFIED CORRECT

### A1. Data Split — Telugu
| Property | Value | Status |
|---|---|---|
| Split strategy | Stratified 70/15/15 on deduplicated labelled set | ✅ |
| Random seed | `random_state=42` in both `train_test_split` calls | ✅ |
| Train size | **2,758** | ✅ |
| Dev size | **591** | ✅ |
| Test size | **592** | ✅ |
| Total (post-dedup) | **3,941** | ✅ |
| Deduplication before split | `drop_duplicates(subset=['text'])` on full dataset **before** splitting | ✅ |
| Cross-split overlap Train↔Dev | **0** (programmatically confirmed) | ✅ |
| Cross-split overlap Train↔Test | **0** (programmatically confirmed) | ✅ |
| Cross-split overlap Dev↔Test | **0** (programmatically confirmed) | ✅ |
| Intra-split duplicates (all 3 splits) | **0** | ✅ |
| Official unlabelled Telugu test used | **No** — only 4,000 labelled training rows used | ✅ |
| Official unlabelled labels introduced | **None** — no external labels injected | ✅ |

**Class balance (stratified correctly)**:

| Split | Total | Non-hate (0) | Hate (1) | % Hate |
|---|---|---|---|---|
| Train | 2,758 | 1,422 | 1,336 | 48.4% |
| Dev | 591 | 305 | 286 | 48.4% |
| Test | 592 | 305 | 287 | 48.5% |

---

### A2. Label Handling
| Property | Value | Status |
|---|---|---|
| Mapping source | `LABEL_MAPPINGS["telugu"]["mapping"]` in `preprocessing.py` | ✅ |
| `"non-hate"` → | **0** | ✅ |
| `"hate"` → | **1** | ✅ |
| Case handling | Raw label lowercased + stripped before map | ✅ |
| Unknown labels excluded | Rows not in `{hate, non-hate}` dropped | ✅ |
| `label` column type | Integer 0 or 1 in processed CSV | ✅ |
| Original label preserved | `original_label` column kept for auditability | ✅ |
| Unique label values in all splits | Confirmed `[0, 1]` in train, dev, test | ✅ |

---

### A3. Test Evaluation Protocol
| Property | Status |
|---|---|
| Checkpoint selection criterion | Validation **Macro-F1 only** — `if val_f1 > best_val_f1` | ✅ |
| Best checkpoint saved via | `model.save_pretrained(ckpt_dir)` + `tokenizer.save_pretrained(ckpt_dir)` | ✅ |
| Best checkpoint reloaded for test | `AutoModelForSequenceClassification.from_pretrained(ckpt_dir)` | ✅ |
| Test set evaluated only once | Yes — single pass after all epochs complete | ✅ |
| Test predictions used for model selection | **No** — test is untouched during training/selection | ✅ |
| Test DataLoader shuffling | `shuffle=False` | ✅ |

---

### A4. Metrics Implementation

| Metric | Implementation | Averaging | Status |
|---|---|---|---|
| Accuracy | `accuracy_score(y_true, y_pred)` | Overall (all classes) | ✅ |
| Precision | `precision_score(..., average='macro')` | **Macro** | ✅ |
| Recall | `recall_score(..., average='macro')` | **Macro** | ✅ |
| Macro-F1 | `f1_score(..., average='macro')` | **Macro** | ✅ |
| Per-class F1 | `f1_score(..., average=None, labels=[0,1])` | Per-class | ✅ |
| Per-class Precision | `precision_score(..., average=None, labels=[0,1])` | Per-class | ✅ |
| Per-class Recall | `recall_score(..., average=None, labels=[0,1])` | Per-class | ✅ |
| Confusion Matrix | `confusion_matrix(y_true, y_pred, labels=[0,1])` | Absolute counts | ✅ |
| Zero-division guard | `zero_division=0` on all sklearn calls | — | ✅ |
| Validation selection metric | `val_metrics['f1_macro']` — Macro-F1 only | — | ✅ |

Confusion matrix plot shows both raw counts and row-normalised percentages (`cm_norm = cm / (cm.sum(axis=1) + 1e-9)`). Epsilon guard prevents division-by-zero. ✅

---

### A5. Robustness — Perturbation Correctness

All perturbations applied to **text only**. Labels are from `eval_df['label']` — unchanged clean test labels. Metrics computed against `clean_labels`. Flip-rate compared `clean_preds` vs `perturbed_preds`. No training occurs during robustness evaluation. Model in `eval()` mode throughout.

| Perturbation | Characters/Words Modified | Seeded |
|---|---|---|
| `LEET_SUBSTITUTION` | `a→@, A→4, e→3, i→1, I→!, o→0, s→$, t→7, T→7` at rate=0.15 | `random.Random(42)` |
| `CHAR_DELETION` | Drops one internal char from words >3 chars (prob 0.4 per word) | `random.Random(42)` |
| `CHAR_REPETITION` | Repeats a random char 2–3× in words >2 chars at rate=0.25 | `random.Random(42)` |
| `PUNCTUATION_INSERTION` | Inserts `.`/`_`/`-` between chars of words >3 chars at rate=0.15 | `random.Random(42)` |
| `WHITESPACE` | Merges adjacent words at rate=0.15 | `random.Random(42)` |
| `COMPOSITE` | char_repetition(0.15) → leet(0.12) → char_deletion(0.08), shared RNG | `random.Random(seed=42)` |

**Per-perturbation formulas (verified in code)**:
```
f1_drop   = clean_f1 - perturbed_f1
retention = perturbed_f1 / (clean_f1 + 1e-9)
flip_rate = mean(clean_preds_i ≠ perturbed_preds_i)
conf_drop = mean(max(clean_probs_i) - max(perturbed_probs_i))
```

---

### A6. SHAP Implementation (Conditional on Completion)
| Property | Status |
|---|---|
| Model explained | Best checkpoint loaded via `from_pretrained(ckpt_dir)` | ✅ |
| Samples | Representative subset from **test set** only (TP/TN/FP/FN) | ✅ |
| Sampling reproducibility | `df.sample(..., random_state=42)` | ✅ |
| Prediction wrapper | tokenizer + model forward → `softmax(logits)` shape (N, 2) | ✅ |
| SHAP masker | `shap.maskers.Text(tokenizer)` — token-level | ✅ |
| Attribution column | `shap_values[i].values[:, 1]` — class 1 (hate) | ✅ |
| Training data leakage | None | ✅ |
| Error handling | try/except with fallback 0.5 — **see B1** | ⚠️ |

---

## B. POTENTIAL ISSUES

### B1. ⚠️ SHAP Fallback Score = 0.5 Is Silent
**File**: `shap_analysis.py` L119  
If SHAP fails (OOM, SHAP error), the function silently returns `mean_compactness_score = 0.5`. This exact value is then used as the Explainability pillar in TWI. A failed SHAP run is **indistinguishable** from a genuine 0.5 measurement in output files.

**Action**: Confirm in Colab logs that SHAP `status = COMPLETED`. If results JSON shows `compactness_score = 0.5` exactly, verify whether SHAP ran or fell back.

---

### B2. ⚠️ ECE Uses Max-Confidence Formulation, Not Positive-Class Probability
**File**: `calibration.py` L32–35  
With a 2D probability array (always the case in this pipeline):
```python
confidences = np.max(probs, axis=1)    # max of P(class 0) and P(class 1)
predictions = np.argmax(probs, axis=1)
```
This is the **multiclass ECE** variant. For binary classification, a common alternative uses `probs[:, 1]` as confidence and compares directly against binary labels. The two variants yield different ECE values for the same model.

**Impact**: Current approach is internally consistent across all 9 experiments (since the same function is used). However, it **must be documented** as the specific ECE variant used.

**Action**: No code change required if consistent cross-experiment reporting is the goal. Add a documentation note.

---

### B3. ⚠️ Robustness Score Derived from Composite Only
**File**: `evaluate.py` L116–118  
`overall_robustness_score` = retention of composite perturbation only. The 5 individual perturbation results are logged in JSON but do not contribute to the TWI numerically.

**Impact**: A model that collapses under `LEET_SUBSTITUTION` but is stable under `COMPOSITE` still gets a high robustness score.

**Action**: Log `mean_retention_all_6` as an additional diagnostic metric. Keep composite-only in TWI for now but add both to results tables in the paper.

---

### B4. ⚠️ Library Versions Not Captured
The Colab environment versions (transformers, torch, shap, sklearn, numpy, pandas) were not locked at runtime. Exact reproducibility depends on these versions.

**Action**: Run `pip freeze > requirements_colab.txt` in Colab after the pilot and commit to the repository.

---

### B5. ⚠️ Perturbation RNG Reset Per-Call, Not Global
**File**: `perturbations.py` L33, L42, L57, L72, L84  
Each individual perturbation function creates `random.Random(42)` fresh per call. Every text is perturbed with the same RNG sequence restarted from seed 42. This is deterministic but means no diversity of random draws across the test set.

**Impact**: Reproducibility is guaranteed. Not a correctness issue. Noted for methodological transparency.

---

## C. METHODOLOGICAL ISSUES

### C1. 🔴 SHAP Mean Compactness Is Not a Validated Explainability Metric

**Exact definition** (from `shap_analysis.py` L87–91):
```
k             = max(1, int(0.2 × n_tokens_for_sample_i))
compactness_i = Σ(top-k sorted |SHAP_class1 values|) / Σ(all |SHAP_class1 values|)
mean_compactness = mean(compactness_i)  over the representative sample set
```

**What it measures**: The fraction of total class-1 SHAP attribution mass concentrated in the top 20% of tokens.

**Range**: [0, 1]. Higher = more attribution concentrated in fewer tokens (more compact explanation).

**Normalised**: Inherently bounded [0, 1] (ratio). No external normalisation applied.

**Higher is better?**: For compactness, yes. For trustworthiness, **not necessarily established**. Diffuse attributions over many tokens may be appropriate for complex hate speech patterns.

**Is it suitable as a TWI component?**: As a project-defined internal consistency metric, it can be used comparatively. As a peer-reviewed "explainability quality" measure, it is **not validated**:
- Not a standard NLP explainability benchmark
- Not derived from human annotation agreement
- Not validated against faithfulness, completeness, or plausibility criteria
- 20% token threshold is arbitrary and not principled

**Required paper statement**: "The SHAP Compactness Score is an internally defined proxy metric representing the concentration of class-level SHAP token attributions in the top 20% of tokens. It is not a standardised or externally validated explainability measure."

---

### C2. 🔴 Brier Normalisation Factor (×2) Is Project-Specific, Not Standard

**Formula**: `calib_brier = max(0.0, 1.0 - (2.0 × Brier))`

The ×2 factor maps:
- Brier = 0.00 → 1.00 (perfect)
- Brier = 0.25 → 0.50 (random guessing)  
- Brier ≥ 0.50 → 0.00 (clipped)

This is not a published standard. The factor is chosen to make 0.25 (random baseline) map to 0.5, which is a reasonable design choice but must be explicitly stated.

**Required paper statement**: "Calibration Score = 0.5·(1−ECE) + 0.5·max(0, 1−2×Brier). The Brier scaling factor (×2) is a project-specific linear normalisation that maps the random-guessing baseline (Brier≈0.25) to a mid-point score of 0.5."

---

### C3. 🔴 TWI Is an Unvalidated Composite Index

**Formula** (balanced profile): `TWI = 0.25·F1_macro + 0.25·Compactness + 0.25·RobScore + 0.25·CalibScore`

Issues:
1. The four pillars are **not commensurable** — they are computed on different bases.
2. Equal weights (0.25) are arbitrary, not empirically or theoretically justified.
3. Performance and Robustness are correlated — higher-performing models tend to be more robust, creating implicit double-counting.
4. The value 0.7386 is not interpretable as "73.86% trustworthy."

**Required paper statement**: "TWI is a project-defined multi-dimensional composite index intended for comparative analysis across the 9 experimental conditions. It should not be interpreted as an absolute or externally validated measure of trustworthiness."

---

### C4. 🟡 Temperature Scaling Defined But Not Applied

`ModelWithTemperature` and `set_temperature_from_logits` are implemented in `calibration.py` but are not called anywhere in the training pipeline. ECE and Brier measure **raw (uncalibrated) softmax probabilities**. This is the correct way to quantify miscalibration, but if the paper's calibration section implies post-calibration analysis, this needs to be clarified.

---

## D. EXACT FORMULAS CURRENTLY USED

### D1. Calibration Score Normalisation
```
calib_ece   = max(0.0, 1.0 - ECE)
calib_brier = max(0.0, 1.0 - (2.0 × Brier))
CalibScore  = 0.5 × calib_ece + 0.5 × calib_brier

Numerical verification (pilot values):
  calib_ece   = 1.0 - 0.0539  = 0.9461
  calib_brier = 1.0 - 2×0.1734 = 0.6532
  CalibScore  = 0.5×0.9461 + 0.5×0.6532 = 0.7997 ≈ 0.7996  ✅
```

### D2. Robustness Score Normalisation (Composite Perturbation)
```
retention_score = clip(f1_retention_composite, 0, 1)
stability_score = max(0.0, 1.0 - flip_rate_composite)
RobScore        = 0.6 × retention_score + 0.4 × stability_score

Numerical verification (pilot values):
  retention_score = 0.9600
  stability_score = 1.0 - 0.1841 = 0.8159
  RobScore        = 0.6×0.9600 + 0.4×0.8159 = 0.5760 + 0.3264 = 0.9024 ≈ 0.9023  ✅
```

### D3. Per-Perturbation Retention and Flip Rate
```
f1_drop      = clean_f1 - perturbed_f1
retention    = perturbed_f1 / (clean_f1 + 1e-9)
flip_rate    = mean(clean_preds_i ≠ perturbed_preds_i)
conf_drop    = mean(max(clean_probs_i) - max(perturbed_probs_i))
```

### D4. SHAP Mean Compactness Score
```
k_i           = max(1, int(0.2 × n_tokens_i))
compactness_i = Σ sorted_abs_shap_class1_i[:k_i] / Σ abs_shap_class1_i
mean_compactness = mean(compactness_i) over all SHAP samples
```

### D5. ECE (Max-Confidence Variant, 10 Equal-Width Bins)
```
confidences = max(softmax_probs, axis=1)
predictions = argmax(softmax_probs, axis=1)
accuracies  = (predictions == y_true)
ECE = Σ_{b=1}^{10} (|S_b|/N) × |mean_conf(S_b) − mean_acc(S_b)|
```

### D6. Brier Score
```
Brier = sklearn.brier_score_loss(y_true, probs[:, 1])
      = mean((y_true_i − P(class=1)_i)²)
```

### D7. TWI (Balanced Weight Profile)
```
TWI = 0.25 × Macro-F1 + 0.25 × Compactness + 0.25 × RobScore + 0.25 × CalibScore

Numerical verification (pilot values):
  = 0.25 × (0.7399 + 0.5127 + 0.9023 + 0.7996)
  = 0.25 × 2.9545 = 0.738625 ≈ 0.7386  ✅
```

---

## E. RECOMMENDED CHANGES BEFORE SCALING

> **Priority 1 — Must verify before running remaining 8 experiments**

| # | Action |
|---|---|
| E1 | Confirm in Colab output that SHAP `status = COMPLETED` for the pilot — check results JSON `compactness_score ≠ 0.5` exactly |
| E2 | Run `pip freeze > requirements_colab.txt` in Colab after pilot; commit to repo for reproducibility |

> **Priority 2 — Recommended for methodological rigour (before paper submission)**

| # | Action |
|---|---|
| E3 | Add `mean_retention_all_perturbations` as logged metric alongside the composite-only score |
| E4 | Add documentation comment in `calibration.py` specifying the ECE variant (max-confidence, not positive-class-probability) |
| E5 | Add per-class F1 (class 0, class 1) to the main results table in the paper, not just Macro-F1 |

> **Priority 3 — Required disclosures in paper**

| # | Action |
|---|---|
| E6 | Explicitly caveat SHAP Compactness as "project-defined proxy metric, not a peer-reviewed explainability standard" |
| E7 | Explicitly caveat TWI as "internally defined composite index for comparative analysis" |
| E8 | State that ECE uses max-confidence ECE variant in Methods section |
| E9 | State that Brier normalisation uses a project-specific ×2 scaling factor |

---

## F. ITEMS THAT MUST REMAIN UNCHANGED

| Item | Reason |
|---|---|
| Label mapping: `hate=1, non-hate=0` | Foundational — changing breaks all downstream results |
| All files in `Data/processed/telugu/` | Verified correct; source of all reported statistics |
| `seed=42` in config and all RNG calls | All reported pilot results are tied to this seed |
| `model_key: mbert` → `bert-base-multilingual-cased` | Correct checkpoint for the pilot |
| ECE = 0.0539, Brier = 0.1734 | Raw test-time calibration from Colab GPU run |
| CalibScore = 0.7996 | Exactly derived from D1 formula (verified) |
| Retention = 0.9600, FlipRate = 0.1841 | Raw robustness from composite perturbation |
| RobScore = 0.9023 | Exactly derived from D2 formula (verified) |
| Macro-F1 = 0.7399 | Primary performance metric from Colab test evaluation |
| Compactness = 0.5127 | Reported SHAP score (if SHAP completed) |
| TWI = 0.7386 | Exactly derived from D7 formula (verified) |
| Checkpoint selection logic (best val Macro-F1) | Must be identical across all 9 experiments for comparability |
| `use_class_weights: false` for Telugu | Near-balanced; deliberate and appropriate |
| ECE formula (max-confidence variant) | Must be identical across all 9 experiments |
| RobScore formula (composite-only) | Must be identical across all 9 experiments |

---

## Reproducibility Checklist

| Parameter | Value |
|---|---|
| Random seed | 42 (data split, training RNG, SHAP sampling) |
| Pretrained model | `bert-base-multilingual-cased` |
| Tokenizer | `bert-base-multilingual-cased` (fast=True) |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Gradient accumulation | 1 (effective batch = 16) |
| Max sequence length | 128 tokens |
| Epochs | 3 |
| Optimizer | AdamW |
| LR scheduler | Linear warmup 10% + linear decay |
| Weight decay | 0.01 |
| Warmup steps | 10% of total steps |
| Mixed precision | FP16 via `torch.cuda.amp` (GPU) |
| Early stopping | None — best checkpoint by val Macro-F1, all 3 epochs complete |
| Classifier dropout | 0.2 |
| Hardware | Tesla T4 GPU, Google Colab |
| Class weights | Disabled |
| Library versions | ⚠️ **Not captured — see E2** |

---

*End of Audit. No experiment outputs modified. No training re-run. All findings from static code inspection and local data file analysis.*
