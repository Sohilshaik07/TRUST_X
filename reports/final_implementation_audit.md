# TRUST-X: Final Implementation Audit Report

**Project:** TRUST-X — An Integrated Trustworthiness Evaluation Framework for  
Transformer-Based Multilingual Code-Mixed Hate and Offensive Speech Detection  
**Audit Date:** 2026-09-18  
**Auditor:** Antigravity IDE (read-only source inspection)

---

## 1. Dataset Summary

| Language  | Train  | Dev   | Test  | Total  | Label 0 (Hateful) | Label 1 (Non-Hateful) |
|-----------|--------|-------|-------|--------|-------------------|-----------------------|
| Telugu    | —      | —     | —     | ✓ processed | ✓ | ✓ |
| Tamil     | —      | —     | —     | ✓ processed | ✓ | ✓ |
| Malayalam | —      | —     | —     | ✓ processed | ✓ | ✓ |
| **Hindi (HECM)** | **6,449** | **1,382** | **1,383** | **9,214** | **0 = Hateful** | **1 = Non-Hateful** |

**Hindi specifics:**
- Source: HECM.txt + HECM_Class.txt (author-provided, untouched)
- Label confirmed by author Prashant K.: `0 = Hateful`, `1 = Non-Hateful`
- 25 exact-text groups with conflicting labels excluded from modelling dataset
- Split: 70 / 15 / 15 stratified by label, seed=42
- Zero cross-split exact-text overlap (verified)
- Schema identical to Telugu/Tamil/Malayalam (`text`, `label`)

---

## 2. Models

| Key        | HuggingFace Checkpoint               |
|------------|--------------------------------------|
| `mbert`    | `bert-base-multilingual-cased`       |
| `indicbert`| `ai4bharat/indic-bert`               |
| `xlmr`     | `xlm-roberta-base`                   |

All three models support 2-class sequence classification (`num_labels=2`).

---

## 3. Experiment Configuration

All 12 experiments use identical hyper-parameters (from their YAML configs):

| Parameter                  | Value   |
|----------------------------|---------|
| Epochs                     | 3       |
| Batch size                 | 16      |
| Gradient accumulation      | 2       |
| Effective batch size       | 32      |
| Learning rate              | 2e-5    |
| Weight decay               | 0.01    |
| Max sequence length        | 128     |
| Warmup fraction            | 10 %    |
| Seed                       | 42      |
| Class weights              | ✓ (balanced) |
| Run SHAP                   | ✓       |
| Run Calibration            | ✓       |
| Run Robustness             | ✓       |
| Run TWI                    | ✓       |

---

## 4. Training Pipeline (src/training/trainer.py)

### 4.1 Class Weights
Computed **once per experiment** on the training split:

```
weight[c] = N / (n_classes × count[c])
```

Implemented in `compute_class_weights()` → passed to `nn.CrossEntropyLoss(weight=...)`.

### 4.2 Optimiser & Scheduler
- **Optimiser:** AdamW (`lr=2e-5`, `weight_decay=0.01`)  
- **Scheduler:** Linear warmup → linear decay  
  - `num_warmup_steps = 0.1 × total_steps`
- **Mixed precision:** FP16 via `torch.cuda.amp` (GPU only)
- **Gradient clipping:** `max_norm=1.0`

### 4.3 Best Checkpoint Selection
Model is saved **only when validation Macro-F1 improves** (not accuracy, not loss).

### 4.4 Test Evaluation
Best checkpoint is **reloaded from disk** via `AutoModelForSequenceClassification.from_pretrained()` before test inference — ensuring the test set was never seen during checkpoint selection.

---

## 5. TRUST-X Pillars

### Pillar 1 — Performance (src/evaluation/metrics.py)

**Primary metric:** Test Macro-F1  
Computed by `sklearn.metrics.f1_score(..., average='macro')`.

Also stored: accuracy, per-class precision/recall/F1, confusion matrix.

**Column in experiments.csv:** `test_macro_f1`

---

### Pillar 2 — Explainability (src/explainability/shap_analysis.py)

**Method:** SHAP Text Explainer (`shap.Explainer`) with `shap.maskers.Text(tokenizer)`.

**Representative subset:** 40 samples (balanced TP/TN/FP/FN) selected with `seed=42`.

**Metric — Mean Compactness Score:**  
For each sample, the fraction of SHAP attribution energy concentrated in the **top 20 % of tokens**:

```
k = max(1, int(0.2 × |tokens|))
compactness_i = sum(|shap|[:k]) / sum(|shap|)
explainability_score = mean(compactness_i)   over all samples
```

Range: [0, 1]. Higher = more compact (fewer decisive tokens).  
If SHAP fails, a neutral fallback of 0.5 is used.

**Column in experiments.csv:** `explainability_score`

---

### Pillar 3 — Robustness (src/robustness/evaluate.py)

**Method:** 6-perturbation suite evaluated on the full test set:

| Perturbation           | Description                                |
|------------------------|--------------------------------------------|
| `leet_substitution`    | Replace chars with leet equivalents        |
| `char_deletion`        | Random character deletions                 |
| `char_repetition`      | Random character repetitions               |
| `punctuation_insertion`| Insert random punctuation                  |
| `whitespace`           | Insert/remove whitespace                   |
| `composite`            | Apply all perturbations in sequence        |

**Reported score** is derived from **composite** perturbation only:

```
F1_retention = perturbed_macro_f1 / (clean_macro_f1 + 1e-9)
flip_rate    = mean(clean_pred ≠ perturbed_pred)
robustness_score = 0.6 × min(1, F1_retention) + 0.4 × max(0, 1 − flip_rate)
```

Implemented in `src/twi/twi.py:normalize_robustness_score()`.

**Column in experiments.csv:** `robustness_score`

---

### Pillar 4 — Calibration (src/calibration/calibration.py)

**Metrics computed:** ECE (10 bins) and Brier Score (sklearn).

**Temperature Scaling:** Fitted via L-BFGS on **validation logits only** (not test set).

**Normalisation formula** (from `src/twi/twi.py:normalize_calibration_score()`):

```
calib_ECE   = max(0, 1 − ECE)
calib_Brier = max(0, 1 − 2 × Brier)
calibration_score = 0.5 × calib_ECE + 0.5 × calib_Brier
```

> Note: `calibration_score` is **not** stored in `experiments.csv` directly.  
> `generate_final_analysis.py` re-derives it from the stored `ece` and `brier` columns.

**Columns in experiments.csv:** `ece`, `brier`

---

### Pillar Aggregation — TWI (src/twi/twi.py)

**Four weight profiles:**

| Profile             | w_perf | w_expl | w_rob | w_calib |
|---------------------|--------|--------|-------|---------|
| `balanced`          | 0.25   | 0.25   | 0.25  | 0.25    |
| `performance_heavy` | 0.50   | 0.15   | 0.20  | 0.15    |
| `robustness_heavy`  | 0.20   | 0.15   | 0.45  | 0.20    |
| `calibration_heavy` | 0.20   | 0.15   | 0.20  | 0.45    |

**TWI formula (balanced profile — stored in experiments.csv):**

```
TWI = 0.25 × Macro-F1 + 0.25 × Explainability + 0.25 × Robustness + 0.25 × Calibration
```

All component scores are in [0, 1]. TWI ∈ [0, 1].

**Column in experiments.csv:** `twi`

---

## 6. Orchestration

### 6.1 run_all_experiments.py

- **Languages:** `["telugu", "tamil", "malayalam", "hindi"]` ✓ (Hindi added)
- **Models:** `["mbert", "indicbert", "xlmr"]`
- **Skip logic:** Reads `results_<experiment_id>.json`; skips if `status == "COMPLETED"`
- **Post-run:** `gc.collect()` + `torch.cuda.empty_cache()` after every run
- **CLI:** `--language`, `--model`, `--epochs`, `--batch_size`, `--no_shap`, `--force_rerun`

### 6.2 Config files (configs/)

| File                      | Experiment ID               |
|---------------------------|-----------------------------|
| `telugu_mbert.yaml`       | `telugu_mbert_seed42`       |
| `telugu_indicbert.yaml`   | `telugu_indicbert_seed42`   |
| `telugu_xlmr.yaml`        | `telugu_xlmr_seed42`        |
| `tamil_mbert.yaml`        | `tamil_mbert_seed42`        |
| `tamil_indicbert.yaml`    | `tamil_indicbert_seed42`    |
| `tamil_xlmr.yaml`         | `tamil_xlmr_seed42`         |
| `malayalam_mbert.yaml`    | `malayalam_mbert_seed42`    |
| `malayalam_indicbert.yaml`| `malayalam_indicbert_seed42`|
| `malayalam_xlmr.yaml`     | `malayalam_xlmr_seed42`     |
| `hindi_mbert.yaml`        | `hindi_mbert_seed42`        |
| `hindi_indicbert.yaml`    | `hindi_indicbert_seed42`    |
| `hindi_xlmr.yaml`         | `hindi_xlmr_seed42`         |

---

## 7. Output Artefacts Per Experiment

Each experiment writes to `checkpoints/<experiment_id>/`:

| File                                  | Contents                                      |
|---------------------------------------|-----------------------------------------------|
| `best_checkpoint/`                    | HuggingFace model + tokenizer (Transformers format) |
| `predictions_<exp_id>.csv`            | text, true label, predicted label, probabilities |
| `results_<exp_id>.json`              | Full metrics JSON (all 4 pillars + TWI)       |
| `confusion_matrix_<exp_id>.png`       | Publication-quality confusion matrix          |
| `reliability_diagram_<exp_id>.png`    | Calibration reliability diagram               |
| `shap_explanations/shap_global_bar.png` | SHAP global attribution bar plot            |

Global log:
- `experiments.csv` (project root) — one row per experiment, all key metrics

---

## 8. Final Analysis Script

**File:** `scripts/generate_final_analysis.py`

Run **after syncing `experiments.csv` from Colab/Drive:**

```bash
python scripts/generate_final_analysis.py
# or, if experiments.csv is elsewhere:
python scripts/generate_final_analysis.py --csv /path/to/experiments.csv
```

**Outputs:**

| Type    | Path                              | Contents                            |
|---------|-----------------------------------|-------------------------------------|
| Table   | `tables/final_results.csv`        | 12 rows × all metrics               |
| Table   | `tables/model_averages.csv`       | 3 rows — per-model averages         |
| Table   | `tables/language_averages.csv`    | 4 rows — per-language averages      |
| Figure  | `figures/final/macro_f1_comparison.png`      | Grouped bar (language × model) |
| Figure  | `figures/final/twi_comparison.png`           | Grouped bar (language × model) |
| Figure  | `figures/final/trust_pillars_comparison.png` | Radar chart (per model)        |
| Figure  | `figures/final/accuracy_vs_macro_f1.png`     | Scatter (accuracy vs macro-F1) |
| Figure  | `figures/final/macro_f1_vs_twi.png`          | Scatter (macro-F1 vs TWI)      |

---

## 9. Sync Instructions (Action Required)

> [!IMPORTANT]
> The following files were produced in Google Colab (Google Drive) and must be
> synced to the local project before the final analysis script can run.

### Required files to copy from Drive → local project:

```
experiments.csv                              (project root)
checkpoints/hindi_mbert_seed42/
checkpoints/hindi_indicbert_seed42/
checkpoints/hindi_xlmr_seed42/
```

Optionally (for completeness):
```
checkpoints/telugu_mbert_seed42/results_telugu_mbert_seed42.json
checkpoints/telugu_indicbert_seed42/results_...json
... (one JSON per experiment)
```

### Then run:
```bash
python scripts/generate_final_analysis.py
```

---

## 10. Audit Verdicts

| Check                                      | Status  |
|--------------------------------------------|---------|
| HECM raw files untouched                   | ✅ PASS |
| Hindi preprocessed data (9,214 samples)    | ✅ PASS |
| Zero cross-split text overlap              | ✅ PASS |
| Label semantics confirmed by author        | ✅ PASS |
| Hindi YAML configs created (3 files)       | ✅ PASS |
| `run_all_experiments.py` includes `hindi`  | ✅ PASS |
| All 12 configs map to correct exp IDs      | ✅ PASS |
| Class-weight formula verified              | ✅ PASS |
| Best checkpoint selection (Val Macro-F1)   | ✅ PASS |
| Test set isolated until final eval         | ✅ PASS |
| ECE: 10-bin equal-width, max-confidence    | ✅ PASS |
| Brier score via sklearn                    | ✅ PASS |
| Temperature scaling on val set only        | ✅ PASS |
| SHAP compactness formula verified          | ✅ PASS |
| Robustness composite formula verified      | ✅ PASS |
| TWI balanced formula verified              | ✅ PASS |
| `experiments.csv` available locally        | ⏳ PENDING SYNC |
| Final figures generated                    | ⏳ PENDING SYNC |
| Final tables generated                     | ⏳ PENDING SYNC |

---

*Report generated by read-only source inspection. No model weights, training data, or existing results were modified.*
