# `run_all_experiments.py` Audit Report
**Audit Date:** 2026-09-16  
**Purpose:** Pre-Step-6 safety check before running all remaining experiments on Google Colab GPU.  
**Status of telugu_mbert_seed42:** ALREADY COMPLETED on Colab (T4 GPU).  
**No training run. No experiment outputs modified.**

---

## 🔴 CRITICAL FINDING — READ FIRST

> **`run_all_experiments.py` contains NO skip/resume logic.**  
> If run without modification, it will **unconditionally rerun `telugu_mbert_seed42`** as experiment #1.  
> This will **overwrite** the completed pilot checkpoint, predictions, SHAP outputs, and results JSON.  
> It will **replace** the `telugu_mbert_seed42` row in `experiments.csv` with freshly computed values.  
> Due to GPU non-determinism (even with seed=42), the rerun results will likely **differ slightly** from the pilot.

**Action required before running Step 6** — see Section 10.

---

## 1. The Exact 9 Experiments (Execution Order)

The orchestrator iterates `LANGUAGES × MODELS` in this fixed nested order:

| # | Config File | `experiment_id` |
|---|---|---|
| 1 | `telugu_mbert.yaml` | `telugu_mbert_seed42` ← **ALREADY DONE** |
| 2 | `telugu_indicbert.yaml` | `telugu_indicbert_seed42` |
| 3 | `telugu_xlmr.yaml` | `telugu_xlmr_seed42` |
| 4 | `tamil_mbert.yaml` | `tamil_mbert_seed42` |
| 5 | `tamil_indicbert.yaml` | `tamil_indicbert_seed42` |
| 6 | `tamil_xlmr.yaml` | `tamil_xlmr_seed42` |
| 7 | `malayalam_mbert.yaml` | `malayalam_mbert_seed42` |
| 8 | `malayalam_indicbert.yaml` | `malayalam_indicbert_seed42` |
| 9 | `malayalam_xlmr.yaml` | `malayalam_xlmr_seed42` |

All 9 experiment IDs are **unique** (programmatically confirmed).  
All 9 `output_dir` paths are **unique** — no checkpoint collision between experiments.

---

## 2. Does It Include `telugu_mbert_seed42`?

**Yes.** It is experiment #1 in the run sequence (`configs/telugu_mbert.yaml` → `experiment_id: telugu_mbert_seed42`).

---

## 3. Skip/Resume Logic: Does It Detect Completed Experiments?

**No. There is zero skip or resume logic anywhere in the codebase.**

- `run_all_experiments.py` (L48–77): Iterates all languages × models unconditionally. No check for existing checkpoints, JSONs, or CSV rows.
- `trainer.py` (L107–109): Calls `os.makedirs(output_dir, exist_ok=True)` — silently creates or reuses the directory. No existence check on prior outputs.
- `trainer.py` (L388–395): `record_to_experiments_csv` **deduplicates by `experiment_id`** — it drops the old row and inserts the new one. This means a rerun **replaces** the existing CSV row, not appends.

**Conclusion:** Running `python scripts/run_all_experiments.py` without modification will unconditionally rerun all 9 experiments from scratch, beginning with the completed pilot.

---

## 4. Inspect Resume Logic Based on Each Artifact

| Artifact | Check in code? | Behaviour on rerun |
|---|---|---|
| `checkpoints/telugu/mbert/best_checkpoint/` | **No** | Silently **overwritten** by `save_pretrained()` |
| `results_telugu_mbert_seed42.json` | **No** | Silently **overwritten** (L349–351 `open(..., "w")`) |
| `predictions_telugu_mbert_seed42.csv` | **No** | Silently **overwritten** (L313–314 `to_csv(...)`) |
| `confusion_matrix_telugu_mbert_seed42.png` | **No** | Silently **overwritten** (L247–248) |
| `reliability_diagram_telugu_mbert_seed42.png` | **No** | Silently **overwritten** (L254–255) |
| `shap_explanations/shap_global_bar.png` | **No** | Silently **overwritten** |
| `experiments.csv` row for pilot | Deduplicated by `experiment_id` | **Row replaced**, not duplicated |

---

## 5. What Happens If the Script Is Run Twice?

| Output | On Second Run |
|---|---|
| `experiments.csv` | **No duplicate row** — old row for each `experiment_id` is removed before new one inserted (L391). Safe from duplicates, but original values are lost. |
| Checkpoints | **Overwritten** — both weight files and tokenizer config |
| Predictions CSV | **Overwritten** — original pilot predictions lost |
| Results JSON | **Overwritten** — original ECE/Brier/TWI/SHAP values lost |
| SHAP figures | **Overwritten** |
| Confusion matrix | **Overwritten** |
| Reliability diagram | **Overwritten** |

**Summary:** Running twice produces no duplicate CSV rows, but **all original pilot outputs are silently destroyed**. The second run's values replace the first.

---

## 6. Pipeline Consistency Across All 9 Experiments

Verified by inspecting all 9 YAML files and `trainer.py`:

| Parameter | All 9 experiments | Consistent? |
|---|---|---|
| Seed | 42 | ✅ |
| Evaluation protocol | Val Macro-F1 checkpoint selection, single test pass | ✅ |
| Metric implementation | Same `compute_all_metrics()` call | ✅ |
| Calibration | Same `compute_ece()` + `compute_brier_score()`, 10 bins, max-confidence ECE | ✅ |
| Robustness | Same `run_robustness_evaluation()`, composite-only score | ✅ |
| SHAP | Same `run_shap_analysis()`, 40 representative samples, seed 42 | ✅ |
| TWI formula | Same `calculate_twi(..., profile="balanced")` | ✅ |
| Epochs | 3 | ✅ |
| Batch size | 16 | ✅ |
| Learning rate | 2e-5 | ✅ |
| Max sequence length | 128 | ✅ |
| Optimizer | AdamW | ✅ |
| Scheduler | Linear warmup 10% + linear decay | ✅ |
| Weight decay | 0.01 | ✅ |
| Gradient accumulation | 1 (effective batch = 16) | ✅ |
| Mixed precision | FP16 on CUDA | ✅ |

---

## 7. Language × Model Coverage

| Language | mBERT | IndicBERT | XLM-R | Data Dir |
|---|---|---|---|---|
| Telugu | `telugu_mbert_seed42` ✅ | `telugu_indicbert_seed42` ✅ | `telugu_xlmr_seed42` ✅ | `Data/processed/telugu` |
| Tamil | `tamil_mbert_seed42` ✅ | `tamil_indicbert_seed42` ✅ | `tamil_xlmr_seed42` ✅ | `Data/processed/tamil` |
| Malayalam | `malayalam_mbert_seed42` ✅ | `malayalam_indicbert_seed42` ✅ | `malayalam_xlmr_seed42` ✅ | `Data/processed/malayalam` |

All 9 cells populated. No Hindi experiments present. ✅

---

## 8. Accidental Use of Official Unlabelled Telugu Test Data

**Not present in any config.** All Telugu configs point to `Data/processed/telugu`, which contains:
- `train.csv` — 2,758 samples from the **labelled** HOLD training set
- `dev.csv` — 591 samples (stratified split)
- `test.csv` — 592 samples (stratified split)

The official unlabelled Telugu competition test file is in `Data/raw/telugu/` and is never referenced by any config or loader. ✅

---

## 9. Tamil and Malayalam — Correct Processed Splits?

### Tamil
- **Config `data_dir`**: `Data/processed/tamil/` ✅
- **Official splits preserved**: DravidianCodeMix-2020 train/dev/test files used as-is (not re-split)
- **Sizes (programmatically confirmed)**:
  - Train: 33,431 | Dev: 4,216 | Test: 4,232
- **Label mapping**: `Not_offensive → 0`, all `Offensive_*` subtypes → 1, `not-Tamil` excluded ✅
- **Deduplication**: Train split deduplicated only; official dev/test preserved intact ✅

### Malayalam
- **Config `data_dir`**: `Data/processed/malayalam/` ✅
- **Official splits preserved**: DravidianCodeMix-2020 train/dev/test files used as-is
- **Sizes (programmatically confirmed)**:
  - Train: 10,802 | Dev: 1,836 | Test: 1,844
- **Label mapping**: `Not_offensive → 0`, all `Offensive_*` subtypes → 1, `not-malayalam` excluded ✅
- **Deduplication**: Train split deduplicated only; official dev/test preserved intact ✅

---

## 10. Class Weights — Correct Per Language?

| Language | Class Imbalance | `use_class_weights` | Correct? |
|---|---|---|---|
| Telugu | 51.6% / 48.4% (near balanced) | `false` | ✅ Appropriate — near-balanced |
| Tamil | 75.4% / 24.6% (moderate imbalance) | `true` | ✅ Appropriate |
| Malayalam | 96.0% / 4.0% (severe imbalance) | `true` | ✅ Essential |

Malayalam has only **430 hate samples out of 10,802 training samples** (4.0%). Class weights are critical — without them, the model would learn to predict "Not offensive" for everything.

**Class weight formula** (from `trainer.py` L56):
```
weight[c] = N / (n_classes × count[c])
```
For Malayalam: weight[0] ≈ 0.52, weight[1] ≈ 12.56 (hate class weighted ~24× more).

---

## 11. Sequential vs. Parallel Execution

**Fully sequential.** The orchestrator's double `for` loop (L48–77) runs each experiment to completion before starting the next. Each call to `train_and_evaluate(config)` is **blocking** — it loads a model, trains, evaluates, runs SHAP/robustness/calibration, saves outputs, then returns. Only then does the next iteration begin.

- One model is in GPU memory at a time ✅  
- No multiprocessing, no threading, no simultaneous loads ✅  
- After each experiment, the `model` and `best_model` objects go out of scope; Python GC and CUDA memory should free them. However, there is **no explicit `torch.cuda.empty_cache()` call** between experiments — see Section 12.

---

## 12. Runtime Estimation

**Pilot benchmark**: Telugu + mBERT, seed 42, T4 GPU — training completed (exact duration logged in `results_telugu_mbert_seed42.json`, reported by user as successful).

**Estimated per-experiment runtime** based on dataset sizes and model complexity:

| Language | Train Size | Relative Scale vs. Telugu | Estimated Runtime |
|---|---|---|---|
| Telugu | 2,758 | 1× (baseline ~20–35 min with SHAP) | ~25–35 min |
| Tamil | 33,431 | ~12× larger | ~90–150 min |
| Malayalam | 10,802 | ~4× larger | ~40–60 min |

**Model overhead** (relative to mBERT, similar parameter counts):
- mBERT: baseline
- XLM-R: slightly slower per step (different architecture)
- IndicBERT: potentially slower (gated — extra HF auth overhead at load; same architecture once loaded)

**Total estimated suite runtime** (8 remaining experiments):

| Experiment | Estimated Time |
|---|---|
| telugu_indicbert + telugu_xlmr | ~50–70 min total |
| tamil_mbert + tamil_indicbert + tamil_xlmr | ~270–450 min total |
| malayalam_mbert + malayalam_indicbert + malayalam_xlmr | ~120–180 min total |
| **Total (8 remaining)** | **~440–700 min (~7–12 hours)** |

> [!IMPORTANT]
> SHAP analysis is the dominant cost for larger datasets. Tamil's 33K training set means the SHAP representative subset (40 samples) evaluation still runs quickly, but the robustness evaluation runs all 4,232 test samples through 6 perturbation passes — that alone is ~25K model forward passes per Tamil experiment.

---

## 13. No-CUDA Behaviour

All 9 configs lack `require_cuda: true`. This means if Colab drops GPU (disconnects), experiments will **silently fall back to CPU** and run extremely slowly without any error. Consider adding `require_cuda: true` to all remaining configs.

---

## Required Actions Before Running Step 6

### 🔴 ACTION 1 — Skip the Completed Pilot (MANDATORY)

Run only the **8 remaining experiments**, not all 9. Use the `--language` or direct config filtering. The safest approach:

**Option A (Recommended): Filter out Telugu mBERT with a flag in Colab Step 6:**
```python
# In Colab Step 6, replace the default command with:
!python scripts/run_all_experiments.py --language telugu --model indicbert
!python scripts/run_all_experiments.py --language telugu --model xlmr
!python scripts/run_all_experiments.py --language tamil
!python scripts/run_all_experiments.py --language malayalam
```

**Option B: Run full suite but back up pilot outputs first:**
```python
!cp -r checkpoints/telugu/mbert checkpoints/telugu/mbert_PILOT_BACKUP
!cp checkpoints/telugu/mbert/results_telugu_mbert_seed42.json ./results_pilot_backup.json
# Then run the full suite — pilot will be rerun but backup is safe
```

**Option C: Add skip logic to `run_all_experiments.py`** — see recommended code change below.

---

### 🟡 ACTION 2 — Add `torch.cuda.empty_cache()` Between Experiments (Recommended)

Insert after each experiment completes to free VRAM before the next model loads. This prevents OOM on later experiments, especially Tamil (33K samples, larger DataLoaders).

---

### 🟡 ACTION 3 — Add `require_cuda: true` to All 8 Remaining Configs

Prevents silent CPU fallback if Colab disconnects from GPU mid-run.

---

### 🟡 ACTION 4 — Capture `pip freeze` Before Running Step 6

Run this once in Colab after the pilot, before Step 6:
```bash
!pip freeze > requirements_colab.txt
```
Commit to the repository for reproducibility.

---

## Recommended Minimal Code Fix for Skip Logic

Add the following at the top of the `for lang in active_langs:` loop in `run_all_experiments.py` (after L69):

```python
# Check if experiment already completed (based on results JSON)
results_json_path = os.path.join(
    root_dir, config["output_dir"], f"results_{config['experiment_id']}.json"
)
if os.path.exists(results_json_path):
    with open(results_json_path) as f:
        existing = json.load(f)
    if existing.get("status") == "COMPLETED":
        print(f"[SKIP] {config['experiment_id']} already COMPLETED. Skipping.")
        successful_runs.append(config['experiment_id'] + " (pre-existing)")
        continue
```

This checks the results JSON for `status: COMPLETED` before launching each experiment. The pilot will be detected and skipped automatically.

---

## Summary Table

| Audit Item | Finding |
|---|---|
| Total experiments | 9 (3 languages × 3 models) |
| Coverage | Telugu × {mBERT, IndicBERT, XLM-R}, Tamil × {mBERT, IndicBERT, XLM-R}, Malayalam × {mBERT, IndicBERT, XLM-R} ✅ |
| Pilot (`telugu_mbert_seed42`) in run | **YES** — position #1 |
| Skip/resume logic | **NONE** — will rerun pilot unconditionally |
| `experiments.csv` on rerun | No duplicates (deduplication by ID), but pilot values replaced |
| Checkpoint overwrite | YES — `save_pretrained()` overwrites silently |
| All other output overwrite | YES — all files opened with `"w"` mode |
| All 9 experiment_ids unique | ✅ |
| All 9 output_dirs unique | ✅ |
| Official unlabelled Telugu test used | **No** ✅ |
| Tamil/Malayalam correct processed splits | ✅ |
| Class weights correct per language | ✅ (Telugu: off, Tamil: on, Malayalam: on) |
| Sequential execution | ✅ |
| Shared pipeline across all 9 | ✅ (seed, metrics, robustness, SHAP, TWI) |
| No explicit CUDA cache clear between runs | ⚠️ OOM risk on Tamil experiments |
| Estimated runtime (8 remaining) | ~7–12 hours on T4 |

---

*End of audit. No training run. No experiment outputs modified.*
