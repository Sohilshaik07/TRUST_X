# TRUST-X User-Facing Application — Design Document

**Version:** 1.0  
**Date:** 2026-09-18  
**Status:** Design Phase — No code written yet  
**Constraint:** Read-only inspection complete. No experimental data, metrics, or checkpoints were modified.

---

## 1. Inspection Findings

### 1.1 Project Directory Structure

```
DL_PROJECT/
├── experiments.csv                # Canonical frozen results (12 rows)
├── requirements.txt               # Python dependencies
├── configs/
│   ├── label_mappings.json        # Language-level task/label definitions
│   └── {lang}_{model}.yaml x12   # Per-experiment configs
├── src/
│   ├── models/model_factory.py    # Model + tokenizer loading, MODEL_REGISTRY
│   ├── data/
│   │   ├── loaders.py             # CodeMixedDataset, get_dataloaders()
│   │   └── preprocessing.py      # clean_text(), language preprocessors
│   ├── calibration/calibration.py # compute_ece, compute_brier_score
│   ├── evaluation/metrics.py      # compute_all_metrics, plot_confusion_matrix
│   ├── explainability/shap_analysis.py  # run_shap_analysis(), predict_probs wrapper
│   ├── robustness/evaluate.py     # run_robustness_evaluation()
│   └── twi/twi.py                 # calculate_twi(), WEIGHT_PROFILES
├── checkpoints/                   # EMPTY — checkpoints saved to Colab/cloud
├── figures/final/ (6 PNGs)        # Frozen publication figures
├── tables/ (7 CSVs)               # Frozen summary tables
└── reports/                       # Freeze report + sensitivity report
```

### 1.2 Critical Finding: Checkpoints Directory Is Empty

**The local `checkpoints/` directory is EMPTY.**
Training was performed in Google Colab with GPU. Checkpoint directories are referenced in configs as
`checkpoints/{language}/{model}/best_checkpoint/` (HuggingFace `save_pretrained()` format),
but those files were never transferred back to the local machine.

**Consequence for the app:**
Live Detection (Page B) cannot load local HuggingFace fine-tuned checkpoints.
The app must load the **base pretrained models** (mBERT, IndicBERT, XLM-R from HuggingFace Hub)
as a fallback, with a clear scientific warning.
All other pages (Dashboard, Evaluation, Comparison, Sensitivity) are unaffected — they use
only the frozen CSV/PNG files.

### 1.3 Checkpoint Loading Path (from trainer.py)

```
trainer.py -> model.save_pretrained(ckpt_dir)
           -> AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
```

`ckpt_dir = output_dir + "/best_checkpoint"`,
e.g. `checkpoints/hindi/mbert/best_checkpoint/`

When checkpoints are transferred from Colab, dropping them in that path is sufficient
for the app to automatically use the fine-tuned weights.

### 1.4 Model Registry (model_factory.py)

| Key | HuggingFace Checkpoint ID |
|---|---|
| mbert | bert-base-multilingual-cased |
| indicbert | ai4bharat/indic-bert |
| xlmr | xlm-roberta-base |

### 1.5 Tokenizer / Model Loading Function

`get_model_and_tokenizer(model_key, num_labels=2, dropout=0.2)` in `src/models/model_factory.py`
Accepts model_key in {mbert, indicbert, xlmr}. Returns `(model, tokenizer, checkpoint_name)`.

### 1.6 SHAP Inference (shap_analysis.py)

`run_shap_analysis(model, tokenizer, representative_df, device, output_dir)` — designed for batch datasets.
For live single-text use, the internal `predict_probs(text_list)` wrapper is directly reusable:
tokenizes input, runs `model.eval()`, returns softmax probabilities.

### 1.7 Preprocessing (preprocessing.py)

`clean_text(text)` — conservative whitespace normalization, safe for live input reuse.
Language-specific preprocessors (preprocess_telugu/tamil/malayalam) are for dataset splits only — NOT reused.

### 1.8 Data File Locations and Schemas

| File | Key Columns |
|---|---|
| experiments.csv | experiment_id, language, model, checkpoint, test_accuracy, test_macro_f1, ece, brier, robustness_score, explainability_score, twi, status |
| tables/final_results.csv | experiment_id, language, model, test_accuracy, test_macro_f1, ece, brier, robustness_score, explainability_score, twi |
| tables/model_averages.csv | model, test_accuracy, test_macro_f1, ece, brier, robustness_score, explainability_score, twi |
| tables/language_averages.csv | language, test_accuracy, test_macro_f1, ece, brier, robustness_score, explainability_score, twi |
| tables/twi_sensitivity.csv | experiment_id, language, model, balanced_twi, performance_heavy_twi, robustness_heavy_twi, calibration_heavy_twi |
| tables/twi_sensitivity_model_averages.csv | model, balanced_twi, performance_heavy_twi, robustness_heavy_twi, calibration_heavy_twi |
| tables/twi_sensitivity_language_averages.csv | language, balanced_twi, performance_heavy_twi, robustness_heavy_twi, calibration_heavy_twi |

### 1.9 Existing Python Dependencies (Confirmed Installed)

- streamlit 1.56.0 — **INSTALLED**
- torch 2.11.0+cpu
- transformers 5.15.0
- shap 0.51.0
- scikit-learn 1.8.0
- matplotlib 3.10.8, seaborn 0.13.2
- pandas >= 2.2.0, numpy >= 1.26.0
- scipy 1.17.1, pyyaml >= 6.0.0

### 1.10 Figures Available (figures/final/)

| File | Content |
|---|---|
| macro_f1_comparison.png | Bar chart: Macro-F1 per language/model |
| twi_comparison.png | Bar chart: TWI per language/model |
| trust_pillars_comparison.png | Grouped bar: All 4 trust pillars |
| accuracy_vs_macro_f1.png | Annotated scatter: Accuracy vs Macro-F1 |
| macro_f1_vs_twi.png | Scatter: Macro-F1 vs TWI |
| twi_sensitivity.png | Bar chart: TWI under 4 weighting schemes |

---

## 2. Proposed Architecture

### 2.1 Guiding Principles

1. **Data integrity**: All experimental metrics are served from frozen CSVs. Zero recomputation.
2. **Scientific honesty**: Live input produces ONLY prediction + confidence + SHAP token attributions.
   It never produces ECE, Brier, robustness score, or TWI — those are dataset-level statistics.
3. **Modularity**: Each page is a self-contained Python module in `app/pages/`.
4. **Reuse first**: Existing `src/` utilities are imported directly — no code is duplicated.
5. **Graceful degradation**: If checkpoints are missing, Live Detection falls back to the base
   pretrained model with a clear scientific warning. All other pages remain fully functional.

### 2.2 File Layout

```
DL_PROJECT/
└── app/
    ├── main.py                     # Streamlit entry point, sidebar nav, page router
    ├── config.py                   # App-level constants (ROOT_DIR, CSV paths, figure paths)
    ├── data_loader.py              # Cached CSV loaders using @st.cache_data
    ├── model_loader.py             # Checkpoint loading with @st.cache_resource + fallback
    ├── pages/
    │   ├── 01_dashboard.py         # Page A: TRUST-X Overview Dashboard
    │   ├── 02_live_detection.py    # Page B: Live Inference + SHAP explanation
    │   ├── 03_model_evaluation.py  # Page C: Frozen metric display per experiment
    │   ├── 04_model_comparison.py  # Page D: Cross-model comparison
    │   └── 05_twi_sensitivity.py   # Page E: Sensitivity analysis display
    └── components/
        ├── metric_cards.py         # Reusable st.metric() card groups
        ├── trust_pillar_gauge.py   # Bar display of 4 trust pillar scores
        └── shap_display.py         # SHAP token HTML rendering
```

---

## 3. Page / Tab Structure

### Page A — Dashboard
**Purpose:** High-level TRUST-X orientation for a new visitor.

Sections:
- Hero banner: TRUST-X title, tagline, 4-language / 3-model scope
- What is TRUST-X? — Concise explainer: 4 pillars, TWI formula, motivation
- Key stats: 12 experiments | 4 languages | 3 models | TWI range min-max
- TWI leaderboard table (from experiments.csv, sorted by TWI descending)
- Quick-access cards linking to other pages

### Page B — Live Detection
**Purpose:** Run inference on user-supplied text. Scientific scope: prediction + confidence + SHAP only.

Sections:
- Language selector (Telugu / Tamil / Malayalam / Hindi)
- Model selector (mBERT / IndicBERT / XLM-R)
- Text input (st.text_area, placeholder with example code-mixed text)
- [Analyze] button
- Output panel:
  - Predicted class badge: Harmful (red) / Non-Harmful (green)
  - Confidence bar (softmax probability for predicted class)
  - SHAP token attribution display
  - Scientific disclaimer box:
    "This prediction uses a transformer model fine-tuned on [language] hate/offensive speech data.
     Confidence reflects raw softmax output — NOT temperature-calibrated probability.
     ECE, Brier Score, Robustness Score, and TWI are dataset-level evaluation metrics
     reported in the Model Evaluation and Model Comparison pages."
- Checkpoint status badge (Fine-tuned / Base model fallback)

### Page C — Model Evaluation
**Purpose:** Display all frozen experimental metrics for one selected experiment.

Sections:
- Selectors: Language + Model (→ maps to 1 of 12 experiments)
- 7 metric cards: Test Accuracy | Macro-F1 | ECE | Brier Score | Robustness | Explainability | TWI
- TWI pillar breakdown: horizontal bar chart of 4 normalized pillar scores
- Experiment metadata: checkpoint ID, epochs, seed, lr, batch size
- Note: "These are dataset-level evaluation metrics from the frozen TRUST-X experiment."

### Page D — Model Comparison
**Purpose:** Compare all 3 models for a selected language, using frozen data and figures.

Sections:
- Language selector
- 3-row comparison table: mBERT | IndicBERT | XLM-R, all 7 metrics
- Embedded final figures:
  - macro_f1_comparison.png
  - twi_comparison.png
  - trust_pillars_comparison.png
  - accuracy_vs_macro_f1.png
  - macro_f1_vs_twi.png
- Cross-language model averages table (from model_averages.csv)

### Page E — TWI Sensitivity
**Purpose:** Display pre-computed sensitivity analysis results. Zero recomputation.

Sections:
- Explainer: Why do weighting schemes matter?
- Four weighting profiles display (Balanced / Perf-Heavy / Rob-Heavy / Calib-Heavy)
- twi_sensitivity.png (embedded)
- Per-experiment sensitivity table (from twi_sensitivity.csv)
- Model averages under each scheme (from twi_sensitivity_model_averages.csv)
- Language averages under each scheme (from twi_sensitivity_language_averages.csv)
- Key conclusions: "XLM-R > mBERT > IndicBERT is stable under all 4 schemes (Spearman rho > 0.93)"

---

## 4. Data Flow

```
experiments.csv (frozen, canonical)
        |
        v
app/data_loader.py — @st.cache_data (TTL = session)
  load_experiments()           -> experiments.csv as DataFrame
  load_final_results()         -> tables/final_results.csv
  load_model_averages()        -> tables/model_averages.csv
  load_language_averages()     -> tables/language_averages.csv
  load_sensitivity()           -> tables/twi_sensitivity.csv
  load_sensitivity_model()     -> tables/twi_sensitivity_model_averages.csv
  load_sensitivity_lang()      -> tables/twi_sensitivity_language_averages.csv
        |
        v
Pages A, C, D, E — read DataFrames + st.image(PNG) — no model loaded
        |
        v
Page B (Live Detection) — model loading path:
  app/model_loader.py
    load_model_for_inference(language, model_key)
      1. Primary:  from_pretrained("checkpoints/{language}/{model_key}/best_checkpoint")
      2. Fallback: from_pretrained(HF_ID[model_key], num_labels=2)
         + display prominent warning banner
    @st.cache_resource keyed on (language, model_key)
        |
        v
  clean_text(user_input)      [src.data.preprocessing.clean_text]
        |
        v
  tokenizer(text, max_length=128, truncation=True, padding=True, return_tensors="pt")
        |
        v
  torch.no_grad(): model(**enc).logits -> softmax
        |
        +-> predicted_class + confidence_score -> display
        |
        v
  shap.maskers.Text(tokenizer)
  shap.Explainer(predict_probs_fn, masker)
  shap_values = explainer([text], max_evals=50)
        |
        v
  app/components/shap_display.py -> token attribution HTML -> st.components.v1.html()
```

---

## 5. Checkpoint Loading Strategy

### Primary Strategy (fine-tuned checkpoints present)
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ckpt_dir = f"checkpoints/{language}/{model_key}/best_checkpoint"
model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
```
Cache with `@st.cache_resource(hash_funcs=...)` keyed on `(language, model_key)`.

### Fallback Strategy (checkpoints absent)
```python
HF_IDS = {
    "mbert":     "bert-base-multilingual-cased",
    "indicbert": "ai4bharat/indic-bert",
    "xlmr":      "xlm-roberta-base",
}
model = AutoModelForSequenceClassification.from_pretrained(HF_IDS[model_key], num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(HF_IDS[model_key], use_fast=True)
```
Display banner:
> "⚠️ Fine-tuned checkpoint not found at checkpoints/{language}/{model_key}/best_checkpoint/.
>  Using base pretrained model. Predictions DO NOT reflect TRUST-X experimental results."

### Checkpoint Transfer Path
When Colab-trained checkpoints are available, place them at:
```
checkpoints/hindi/mbert/best_checkpoint/      (config.json, pytorch_model.bin, tokenizer files)
checkpoints/hindi/indicbert/best_checkpoint/
checkpoints/hindi/xlmr/best_checkpoint/
checkpoints/tamil/mbert/best_checkpoint/
checkpoints/tamil/indicbert/best_checkpoint/
checkpoints/tamil/xlmr/best_checkpoint/
checkpoints/telugu/mbert/best_checkpoint/
checkpoints/telugu/indicbert/best_checkpoint/
checkpoints/telugu/xlmr/best_checkpoint/
checkpoints/malayalam/mbert/best_checkpoint/
checkpoints/malayalam/indicbert/best_checkpoint/
checkpoints/malayalam/xlmr/best_checkpoint/
```
No app code changes required — the primary strategy auto-activates.

---

## 6. Component Reuse Strategy

### Directly Reused (import as-is, zero modification)

| Component | Source File | Usage in App |
|---|---|---|
| `clean_text(text)` | src/data/preprocessing.py | Pre-process live user input before tokenization |
| `get_device()` | src/models/model_factory.py | Detect CPU/GPU for model inference |
| `MODEL_REGISTRY` | src/models/model_factory.py | HF checkpoint IDs for fallback loading |
| `WEIGHT_PROFILES` | src/twi/twi.py | Weight scheme labels/values for Page E display |
| All 7 CSV files | tables/ + experiments.csv | Read-only pandas display across all pages |
| All 6 PNG figures | figures/final/ | st.image() embed in Pages D and E |

### Adapted (import + wrap, original files untouched)

| Component | Adaptation |
|---|---|
| predict_probs pattern from shap_analysis.py | Replicated as a local closure in model_loader.py for single-text SHAP |
| SHAP explainer setup from shap_analysis.py | Called with a 1-element text list instead of a representative DataFrame |

### New Files (must be created)

| File | Purpose |
|---|---|
| app/main.py | Streamlit entry point and page router |
| app/config.py | ROOT_DIR, CSV paths, figure paths, app constants |
| app/data_loader.py | @st.cache_data CSV loaders (7 loaders) |
| app/model_loader.py | Checkpoint loading + fallback strategy |
| app/components/metric_cards.py | Reusable st.metric() card group component |
| app/components/trust_pillar_gauge.py | 4-pillar horizontal bar display |
| app/components/shap_display.py | SHAP token attribution HTML renderer |
| app/pages/01_dashboard.py | Page A implementation |
| app/pages/02_live_detection.py | Page B implementation (inference + SHAP) |
| app/pages/03_model_evaluation.py | Page C implementation |
| app/pages/04_model_comparison.py | Page D implementation |
| app/pages/05_twi_sensitivity.py | Page E implementation |

**Total: 12 new files. Zero modifications to existing project files.**

---

## 7. Known Limitations

| # | Limitation | Impact | Mitigation |
|---|---|---|---|
| L1 | **All 12 fine-tuned checkpoints are absent locally.** Trained in Colab, not transferred. | Page B uses base pretrained model; predictions are not meaningful scientifically. | Architecture transparently swaps to fine-tuned weights when checkpoints are placed in correct paths. |
| L2 | **SHAP is slow on CPU.** Single-text SHAP with max_evals=50 may take 30–90 seconds. | Live Detection latency is high. | Spinner + status message; optional "Skip SHAP" toggle. Lower max_evals to 30 for speed. |
| L3 | **Hindi has no LABEL_MAPPINGS entry** in preprocessing.py (only Telugu/Tamil/Malayalam). Hindi uses the HECM dataset with a different pipeline. | Label text must be hardcoded for Hindi. | Use unified display: class 0 = Non-Harmful, class 1 = Harmful across all languages. |
| L4 | **Live predictions cannot yield dataset-level metrics.** ECE, Brier, robustness score, explainability compactness, and TWI are meaningful only over a full test set. | Risk of scientific misrepresentation. | Enforce strict separation: Page B shows only predicted class + confidence + SHAP. All evaluation metrics live only on Pages C/D/E, sourced exclusively from frozen CSVs. |
| L5 | **XLM-R does not produce token_type_ids.** Existing CodeMixedDataset handles this gracefully already. | Minor inference detail. | model_loader.py inspects tokenizer output dict and passes only available keys. |
| L6 | **CPU-only environment.** No CUDA GPU locally. | Inference and SHAP are slower than Colab. | Acceptable for a demonstration app. max_length=128 matches training. |

---

## 8. Dependency Requirements

### Already Installed (confirmed, no action required)
- streamlit 1.56.0
- torch 2.11.0+cpu
- transformers 5.15.0
- shap 0.51.0
- pandas >= 2.2.0, numpy >= 1.26.0
- matplotlib 3.10.8, seaborn 0.13.2
- scikit-learn 1.8.0, scipy 1.17.1

### Potentially Required (verify before Phase 2)
- `plotly` — for interactive bar/radar charts (pip install plotly)
- `altair` — ships with Streamlit 1.56, usually pre-installed

### Not Required
- No training frameworks (accelerate, datasets, etc.)
- No raw dataset files (app reads only from CSVs and PNGs)
- No calibration refitting (all values are pre-computed and frozen)

---

## 9. Implementation Plan

### Phase 1 — Foundation
1. `app/config.py` — paths, constants, MODEL_REGISTRY copy
2. `app/data_loader.py` — 7 @st.cache_data CSV loaders
3. `app/model_loader.py` — checkpoint + fallback loading strategy
4. `app/components/metric_cards.py` — reusable metric display
5. `app/main.py` — sidebar nav, page routing

### Phase 2 — Static Display Pages
6. `app/pages/01_dashboard.py` — hero, leaderboard, quick links
7. `app/pages/03_model_evaluation.py` — 7 metric cards + pillar chart
8. `app/pages/04_model_comparison.py` — comparison table + 5 embedded PNGs
9. `app/pages/05_twi_sensitivity.py` — scheme tables + sensitivity PNG

### Phase 3 — Live Inference Page
10. `app/components/shap_display.py` — token attribution HTML
11. `app/components/trust_pillar_gauge.py` — pillar bar chart
12. `app/pages/02_live_detection.py` — full inference + SHAP pipeline
13. End-to-end test: `streamlit run app/main.py`

**Estimated scope:** 12 new files, 0 modified files, ~800–1200 lines of new Python.

---

## 10. Component Reuse Summary

### Can Be Reused (Existing — No Modification)
| Component | Source | How |
|---|---|---|
| `clean_text()` | src/data/preprocessing.py | Live text normalization before tokenization |
| `get_device()` | src/models/model_factory.py | Device detection |
| `MODEL_REGISTRY` | src/models/model_factory.py | HuggingFace IDs for fallback |
| `WEIGHT_PROFILES` | src/twi/twi.py | Scheme labels and weights on Page E |
| 7 frozen CSV files | tables/ + experiments.csv | Read-only via @st.cache_data |
| 6 frozen PNG figures | figures/final/ | st.image() embed |

### Must Be Created (New)
| File | Why New |
|---|---|
| app/main.py | Streamlit application does not exist yet |
| app/config.py | Central path registry needed |
| app/data_loader.py | Cached CSV loading layer |
| app/model_loader.py | HF checkpoint + fallback logic |
| app/pages/01_dashboard.py | New UI page |
| app/pages/02_live_detection.py | New UI page with inference + SHAP |
| app/pages/03_model_evaluation.py | New UI page |
| app/pages/04_model_comparison.py | New UI page |
| app/pages/05_twi_sensitivity.py | New UI page |
| app/components/metric_cards.py | Reusable UI component |
| app/components/trust_pillar_gauge.py | Reusable UI component |
| app/components/shap_display.py | Reusable UI component |

---

*This document is a read-only design proposal.*
*No code has been written. No experimental data has been modified.*
*All experimental metrics, figures, and CSV files remain frozen.*
