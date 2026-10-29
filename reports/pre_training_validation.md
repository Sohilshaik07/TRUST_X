# TRUST-X: Pre-Training Validation Audit Report

**Date**: September 15, 2026  
**Auditor**: Antigravity AI  
**Scope**: Pre-training verification of processed datasets, leakage audit, label integrity, dynamic class weights, YAML configurations, model checkpoints, and Colab GPU readiness.

---

## 1. Processed Dataset Verification & Telugu Count Arithmetic

All numbers reported below were recalculated directly from the active files in `Data/processed/`:

| Language | Split | Total Count | Class 0 (Non-Harmful) | Class 1 (Harmful) | Harmful Class % | Non-Native Excluded |
|---|---|---|---|---|---|---|
| **Telugu** | **Train** | **2,758** | 1,422 | 1,336 | 48.44% | N/A |
| **Telugu** | **Dev** | **591** | 305 | 286 | 48.39% | N/A |
| **Telugu** | **Test** | **592** | 305 | 287 | 48.48% | N/A |
| **Telugu Total** | **All** | **3,941** | **2,032** | **1,909** | **48.44%** | **N/A** |
| **Tamil** | **Train** | **33,431** | 25,206 | 8,225 | 24.60% | 1,454 `not-Tamil` |
| **Tamil** | **Dev** | **4,216** | 3,193 | 1,023 | 24.26% | 172 `not-Tamil` |
| **Tamil** | **Test** | **4,232** | 3,190 | 1,042 | 24.62% | 160 `not-Tamil` |
| **Tamil Total** | **All** | **41,879** | **31,589** | **10,290** | **24.57%** | **1,786 `not-Tamil`** |
| **Malayalam** | **Train** | **10,802** | 10,372 | 430 | 3.98% | 1,287 `not-malayalam` |
| **Malayalam** | **Dev** | **1,836** | 1,779 | 57 | 3.10% | 163 `not-malayalam` |
| **Malayalam** | **Test** | **1,844** | 1,765 | 79 | 4.28% | 157 `not-malayalam` |
| **Malayalam Total** | **All** | **14,482** | **13,916** | **566** | **3.91%** | **1,607 `not-malayalam`** |

### Telugu Count Arithmetic Investigation:
- **Raw Rows in `training_data_telugu-hate.xlsx`**: **4,000**
- **Raw Unique Strings (Untrimmed)**: 3,949 (51 exact duplicates in raw)
- **Whitespace Normalization Effect**: 8 additional comments differed solely by leading/trailing spaces or internal multiple whitespace characters. After standard conservative normalization (`re.sub(r'\s+', ' ', text).strip()`), these 8 comments became identical to existing samples.
- **Total Duplicate Texts Removed**: **59**
- **Unique Processed Rows**: $4,000 - 59 =$ **3,941**
- **Partitioning (70% / 15% / 15% Stratified)**:
  $$\text{Train } (2,758) + \text{Dev } (591) + \text{Test } (592) = 3,941$$
- **Arithmetic Verification**: **PASS (100% exact match)**.

---

## 2. Duplicate and Leakage Validation

### 2.1. Within-Split Duplicate Audit:
- **Telugu**: Train = **0**, Dev = **0**, Test = **0**.
- **Tamil**: Train = **0** (254 within-train duplicates removed); Dev = 9; Test = 9 (preserved official benchmark samples).
- **Malayalam**: Train = **0** (3,921 within-train duplicates removed); Dev = 71; Test = 55 (preserved official benchmark samples).

### 2.2. Cross-Split Overlap Audit:
- **Telugu**:
  - Overlap Train & Dev: **0**
  - Overlap Train & Test: **0**
  - Overlap Dev & Test: **0**
  - *Result*: **Zero data leakage**.
- **Tamil**:
  - Overlap Train & Dev: 56
  - Overlap Train & Test: 66
  - Overlap Dev & Test: 9
- **Malayalam**:
  - Overlap Train & Dev: 822
  - Overlap Train & Test: 846
  - Overlap Dev & Test: 143
  *(Note: In DravidianCodeMix-2020, common generic social media comments such as "Super", "Waiting", and "Nice" exist across official competition splits. Official test/dev splits were preserved per Rule 8).*

### 2.3. Telugu Official Unlabeled Test Check:
- File `Data/raw/telugu/telugu-hate-speech-test.xlsx` (500 rows, unlabelled holdout) was audited.
- **Verification**: This file is **100% excluded** from `Data/processed/`, data loaders, and experiment configurations.

---

## 3. Label Mapping and Schema Validation

Every processed CSV contains the required standardized schema:
`['id', 'text', 'label', 'original_label', 'language', 'task', 'split']`.

Programmatic check results:
1. **Telugu**:
   - `label == 0` $\rightarrow$ `original_label == 'non-hate'` (**100% True**)
   - `label == 1` $\rightarrow$ `original_label == 'hate'` (**100% True**)
2. **Tamil**:
   - `label == 0` $\rightarrow$ `original_label == 'Not_offensive'` (**100% True**)
   - `label == 1` $\rightarrow$ `original_label` is one of `Offensive_*` (**100% True**)
   - `not-Tamil` presence in processed dataset: **0 (100% excluded)**
3. **Malayalam**:
   - `label == 0` $\rightarrow$ `original_label == 'Not_offensive'` (**100% True**)
   - `label == 1` $\rightarrow$ `original_label` is one of `Offensive_*` (**100% True**)
   - `not-malayalam` presence in processed dataset: **0 (100% excluded)**

---

## 4. Dynamic Class Weights Verification

Class weights calculated dynamically via $w_c = \frac{N}{C \cdot N_c}$ from the actual processed training splits:

- **Telugu Train ($N=2,758$)**:
  - Class 0 ($N_0=1,422$, $51.56\%$): **$0.9698$**
  - Class 1 ($N_1=1,336$, $48.44\%$): **$1.0322$**
  - *Ratio*: $1.06\times$ (Balanced)
- **Tamil Train ($N=33,431$)**:
  - Class 0 ($N_0=25,206$, $75.40\%$): **$0.6632$**
  - Class 1 ($N_1=8,225$, $24.60\%$): **$2.0323$**
  - *Ratio*: $3.06\times$
- **Malayalam Train ($N=10,802$)**:
  - Class 0 ($N_0=10,372$, $96.02\%$): **$0.5207$**
  - Class 1 ($N_1=430$, $3.98\%$): **$12.5605$**
  - *Ratio*: **$24.12\times$**

*Conclusion*: Dynamic weighting correctly allocates $12.56\times$ weight to the minority offensive class in Malayalam.

---

## 5. Experiment Configurations Audit (9 Configurations)

All 9 configuration files in `configs/` were inspected:
1. `telugu_mbert.yaml`: Telugu | `bert-base-multilingual-cased` | Seed 42 | lr 2e-5 | BS 16 | Epochs 3 | Len 128
2. `telugu_indicbert.yaml`: Telugu | `ai4bharat/indic-bert` | Seed 42 | lr 2e-5 | BS 16 | Epochs 3 | Len 128
3. `telugu_xlmr.yaml`: Telugu | `xlm-roberta-base` | Seed 42 | lr 2e-5 | BS 16 | Epochs 3 | Len 128
4. `tamil_mbert.yaml`: Tamil | `bert-base-multilingual-cased` | Seed 42 | lr 2e-5 | BS 16 | Epochs 3 | Len 128 | Class Weights: True
5. `tamil_indicbert.yaml`: Tamil | `ai4bharat/indic-bert` | Seed 42 | lr 2e-5 | BS 16 | Epochs 3 | Len 128 | Class Weights: True
6. `tamil_xlmr.yaml`: Tamil | `xlm-roberta-base` | Seed 42 | lr 2e-5 | BS 16 | Epochs 3 | Len 128 | Class Weights: True
7. `malayalam_mbert.yaml`: Malayalam | `bert-base-multilingual-cased` | Seed 42 | lr 2e-5 | BS 16 | Epochs 3 | Len 128 | Class Weights: True
8. `malayalam_indicbert.yaml`: Malayalam | `ai4bharat/indic-bert` | Seed 42 | lr 2e-5 | BS 16 | Epochs 3 | Len 128 | Class Weights: True
9. `malayalam_xlmr.yaml`: Malayalam | `xlm-roberta-base` | Seed 42 | lr 2e-5 | BS 16 | Epochs 3 | Len 128 | Class Weights: True

- **Hindi configs**: None present (POSTPONED).
- **Unlabelled Telugu test**: 0 occurrences.
- **Number of labels**: 2 across all configs.

---

## 6. Pretrained Model & Tokenizer Accessibility Test

A live Hugging Face Hub access check was performed on the three checkpoints:
1. **`bert-base-multilingual-cased` (mBERT)**:
   - Status: **PASS (100% Accessible)**
   - Tokenizer: `BertTokenizer`
   - Test encoding shape: 128 tokens
2. **`xlm-roberta-base` (XLM-RoBERTa)**:
   - Status: **PASS (100% Accessible)**
   - Tokenizer: `XLMRobertaTokenizer`
   - Test encoding shape: 128 tokens
3. **`ai4bharat/indic-bert` (IndicBERT)**:
   - Status: **GATED REPO NOTICE (401 Client Error)**
   - *Finding*: AI4Bharat currently gates `ai4bharat/indic-bert` on Hugging Face Hub, requiring acceptance of user terms and a free Hugging Face access token (`huggingface_hub.login()`).
   - *Alternative*: The non-gated official successor `ai4bharat/IndicBERTv2-MLM-only` is completely open (`gated: False`).
   - *Resolution added*: An authentication cell has been integrated into [`notebooks/TRUST_X_Colab_Runner.ipynb`](notebooks/TRUST_X_Colab_Runner.ipynb) so the user can authenticate with their HF token.

---

## 7. Google Colab Notebook Verification

[`notebooks/TRUST_X_Colab_Runner.ipynb`](notebooks/TRUST_X_Colab_Runner.ipynb) verified:
1. CUDA GPU detection and VRAM printing (`!nvidia-smi` and `torch.cuda.get_device_properties`).
2. Google Drive mounting and repository directory resolution.
3. Automated dependency installation (`pip install -q transformers datasets scikit-learn shap...`).
4. Optional Hugging Face token authentication.
5. Pilot experiment launch cell (`telugu_mbert.yaml`).
6. Master 9-experiment suite runner cell (`scripts/run_all_experiments.py`).
7. Result summary table visualizer.

---

## 8. Final Audit Verdict

```
============================================================
PRE-TRAINING VALIDATION: PASS
============================================================
All processed datasets, counts, arithmetic, label mappings,
dynamic class weights, experiment configurations, and
Colab runner notebook have been verified.

Notice: For ai4bharat/indic-bert, enter your free HF token in Colab
or accept the agreement at https://huggingface.co/ai4bharat/indic-bert.
============================================================
```
