# Phase 1: Comprehensive Dataset Audit Report
**Project**: TRUST-X: An Integrated Trustworthiness Evaluation Framework for Transformer-Based Multilingual Code-Mixed Hate and Offensive Speech Detection  
**Date**: September 15, 2026  
**Auditor**: Antigravity AI  

---

## 1. Executive Summary

A programmatic audit of all raw dataset files for the three active language settings (**Telugu-English**, **Tamil-English**, and **Malayalam-English**) was conducted in strict accordance with data integrity rules:
- **Hindi** (`Data/raw/hindi/`) was detected as an empty directory and **left completely untouched** (status: POSTPONED).
- Raw data files in `Data/raw/` were accessed in **read-only mode** and zero raw files were modified.
- All statistics, distributions, lengths, duplicates, and script compositions reported below were computed directly from the data files without fabrication or estimation.

---

## 2. Dataset-by-Dataset Audit Findings

### 2.1. Telugu-English (HOLD / DravidianLangTech-EACL 2024)
- **Task**: Hate Speech Detection
- **Files Audited**:
  - `Data/raw/telugu/training_data_telugu-hate.xlsx` (4,000 rows, 3 columns: `['S.No', 'Comments', 'Label']`)
  - `Data/raw/telugu/telugu-hate-speech-test.xlsx` (500 rows, 2 columns: index + text)

#### Critical Findings:
1. **Official Test Set is Completely Unlabelled**:
   - `telugu-hate-speech-test.xlsx` contains 500 rows with only sample IDs and comment texts. It contains **no label column** (organized as an unreleased evaluation set for shared-task participants).
   - Therefore, the official test file **cannot be used for quantitative model evaluation** (accuracy, F1, calibration, robustness).
   - Furthermore, **38 texts** in the test file overlap with the training set.
2. **Training Set Characteristics**:
   - Total rows: 4,000. Missing texts: 0. Missing labels: 0. Exact duplicate rows: 0. Exact duplicate texts: 51.
   - Class Distribution:
     - `non-hate`: 2,061 (51.52%)
     - `hate`: 1,939 (48.48%)
     - The training set is remarkably well-balanced.
3. **Linguistic & Length Statistics**:
   - Characters: Mean = 66.85, Median = 56.0, Min = 8, Max = 458.
   - Word Tokens: Mean = 9.95, Median = 8.0, 95th Percentile = 22.0, Max = 71.
   - Script Composition: Latin (Romanized Telugu/English) = 48.52%, Native Telugu Script = 35.69%, Punctuation/Symbols = 15.79%.
4. **Partition Recommendation**:
   - Deduplicate the 4,000 training samples.
   - Create a stratified 70% Train (2,764 samples), 15% Validation (592 samples), and 15% Test (593 samples) split using random seed 42 to provide a genuine, labelled test set for final model evaluation without any data leakage.

---

### 2.2. Tamil-English (DravidianCodeMix-2020)
- **Task**: Offensive Language Detection
- **Files Audited**:
  - `tamil_offensive_full_train.csv` (35,139 rows, TSV, no header)
  - `tamil_offensive_full_dev.csv` (4,388 rows, TSV, no header)
  - `tamil_offensive_full_test.csv` (4,392 rows, TSV, no header)
  - `tamil_offensive_full.csv` (43,919 rows, TSV, no header)

#### Critical Findings:
1. **Split Containment & Partition Integrity**:
   - `Train (35,139)` + `Dev (4,388)` + `Test (4,392)` = **43,919**, exactly matching `tamil_offensive_full.csv`.
   - The official partitions are mathematically complete.
2. **Column Structure Discrepancy**:
   - In `train`, `dev`, and `test`: Column 0 is `text`, Column 1 is `label`, Column 2 is trailing tab/empty.
   - In `tamil_offensive_full.csv`: Column 0 is `label`, Column 1 is `text`.
3. **Original 6-Class Label Distribution**:
   - **Train**:
     - `Not_offensive`: 25,425 (72.36%)
     - `Offensive_Untargetede`: 2,906 (8.27%) *(Note: original dataset typo preserved in raw)*
     - `Offensive_Targeted_Insult_Group`: 2,557 (7.28%)
     - `Offensive_Targeted_Insult_Individual`: 2,343 (6.67%)
     - `not-Tamil`: 1,454 (4.14%)
     - `Offensive_Targeted_Insult_Other`: 454 (1.29%)
   - **Dev**: `Not_offensive` (72.77%), `Offensive_*` (23.31%), `not-Tamil` (3.92%)
   - **Test**: `Not_offensive` (72.63%), `Offensive_*` (23.73%), `not-Tamil` (3.64%)
4. **Cross-Split Leakage**:
   - Overlap between Train & Dev: 57 texts.
   - Overlap between Train & Test: 65 texts.
   - Overlap between Dev & Test: 9 texts.
5. **Linguistic & Length Statistics**:
   - Characters: Mean = 72.9, Median = 56.0, Max = 1,149.
   - Word Tokens: Mean = 10.4, Median = 8.0, 95th Percentile = 25.0, Max = 127.
   - Script Composition: Latin (Tanglish) = 65.95%, Native Tamil Script = 15.14%, Symbols/Other = 18.84%.
6. **Binary Mapping Strategy for Phase 2**:
   - `Not_offensive` -> `0` (Non-offensive)
   - All `Offensive_*` subtypes -> `1` (Offensive, total ~23.5%)
   - `not-Tamil` -> Filtered out / Excluded (since offensive language cannot be evaluated for non-Tamil text).

---

### 2.3. Malayalam-English (DravidianCodeMix-2020)
- **Task**: Offensive Language Detection
- **Files Audited**:
  - `mal_full_offensive_train.csv` (16,010 rows, TSV, no header)
  - `mal_full_offensive_dev.csv` (1,999 rows, TSV, no header)
  - `mal_full_offensive_test.csv` (2,001 rows, TSV, no header)
  - `mal_full_offensive.csv` (20,010 rows, TSV, no header)

#### Critical Findings:
1. **Split Containment & Partition Integrity**:
   - `Train (16,010)` + `Dev (1,999)` + `Test (2,001)` = **20,010**, exactly matching `mal_full_offensive.csv`.
2. **Severe Class Imbalance**:
   - **Train**:
     - `Not_offensive`: 14,153 (88.40%)
     - `not-malayalam`: 1,287 (8.04%)
     - `Offensive_Targeted_Insult_Individual`: 239 (1.49%)
     - `Offensive_Untargetede`: 191 (1.19%)
     - `Offensive_Targeted_Insult_Group`: 140 (0.87%)
   - **Total Offensive Samples in Train**: Only 570 out of 16,010 (**3.56%**)!
   - **Total Offensive Samples in Dev**: 57 out of 1,999 (**2.85%**).
   - **Total Offensive Samples in Test**: 79 out of 2,001 (**3.95%**).
   - *Methodological implication*: Standard Cross-Entropy will heavily bias towards predicting non-offensive. Class-weighted Cross-Entropy loss will be strictly required for Malayalam.
3. **High Duplicate Count**:
   - Train has **4,320 duplicate texts** (due to generic short comments like "Pwoli", "Nice", "Super").
   - Cross-split overlap: 914 between Train & Dev; 934 between Train & Test.
4. **Linguistic & Length Statistics**:
   - Characters: Mean = 67.5, Median = 56.0, Max = 1,304.
   - Word Tokens: Mean = 9.1, Median = 8.0, 95th Percentile = 23.0, Max = 186.
   - Script Composition: Latin (Manglish) = 56.53%, Native Malayalam Script = 25.99%, Symbols/Other = 17.46%.
5. **Binary Mapping Strategy for Phase 2**:
   - `Not_offensive` -> `0` (Non-offensive)
   - All `Offensive_*` subtypes -> `1` (Offensive, ~3.8% of valid Malayalam data)
   - `not-malayalam` -> Filtered out / Excluded.

---

## 3. Comparative Summary Table

| Language | Dataset | Task | Train Rows | Dev Rows | Test Rows | Classes | Offensive/Hate % (Train) | Code-Mixing (Latin Script %) | Native Script % | Max Tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| **Telugu** | HOLD (EACL 2024) | Hate Speech Detection | 4,000 | N/A | 500 (unlabelled) | 2 | 48.48% (Balanced) | 48.52% | 35.69% | 71 |
| **Tamil** | DravidianCodeMix-2020 | Offensive Language | 35,139 | 4,388 | 4,392 | 6 | 23.51% (Moderate Imbalance) | 65.95% | 15.14% | 127 |
| **Malayalam** | DravidianCodeMix-2020 | Offensive Language | 16,010 | 1,999 | 2,001 | 5 | 3.56% (Severe Imbalance) | 56.53% | 25.99% | 186 |

---

## 4. Architectural Readiness & Audit Verdict

- **Dataset Audit Status**: **COMPLETED**
- **Figures Generated**:
  1. `figures/audit_class_distributions.png`
  2. `figures/audit_text_lengths.png`
  3. `figures/audit_script_composition.png`
- **Tables Generated**:
  1. `tables/dataset_audit.csv`
  2. `tables/class_distributions.csv`
- **Readiness Verdict**:
  - **Telugu**: READY FOR PREPROCESSING (requires stratified split of labelled training file due to unlabelled official test file).
  - **Tamil**: READY FOR PREPROCESSING (preserve official train/dev/test splits; map 5 offensive classes to binary; exclude `not-Tamil`).
  - **Malayalam**: READY FOR PREPROCESSING (preserve official train/dev/test splits; map 4 offensive classes to binary; exclude `not-malayalam`; apply class-weighted loss).
