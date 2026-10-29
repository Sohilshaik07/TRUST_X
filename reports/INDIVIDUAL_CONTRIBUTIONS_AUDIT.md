# TRUST-X PROJECT — INDIVIDUAL CONTRIBUTIONS AUDIT & ALLOCATION REPORT

**Course Code:** BCSE332L — Deep Learning  
**Project Title:** TRUST-X: An Integrated Trustworthiness Evaluation Framework for Transformer-Based Multilingual Code-Mixed Hate and Offensive Speech Detection  
**Institution:** School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology (VIT), Vellore  
**Course Instructor / Supervisor:** Dr. CHELLATAMILAN T, Professor Grade 1, SCOPE  
**Audit Date:** October 29, 2026  

---

## 1. Executive Summary & Deliverables Inventory

In accordance with the university regulation that *"EVERY INDIVIDUAL HAS TO SUBMIT THE PROJECT REPORT (HIGHLIGHTING INDIVIDUAL CONTRIBUTION ……)"*, this audit documents the formal, verified division of technical responsibilities among the three TRUST-X project team members:

1. **Shaik Mohammad Sohil — 23BAI0133**
2. **Jeevan G A — 23BAI0140**
3. **Shashank Pinnelli — 23BAI0152**

### Generated Artifacts Summary
The following submission-ready artifacts were generated, validated, and verified:

| Deliverable Type | File Path | Scope & Purpose | Status / Page Count |
| :--- | :--- | :--- | :---: |
| **Standalone Contribution Doc** | [`reports/Individual_Contribution_Sohil_23BAI0133.docx`](file:///c:/Users/SKSOH/OneDrive/Desktop/DL_PROJECT/reports/Individual_Contribution_Sohil_23BAI0133.docx) | Standalone individual contribution report for Shaik Mohammad Sohil | **Complete (3 pages)** |
| **Standalone Contribution Doc** | [`reports/Individual_Contribution_Jeevan_23BAI0140.docx`](file:///c:/Users/SKSOH/OneDrive/Desktop/DL_PROJECT/reports/Individual_Contribution_Jeevan_23BAI0140.docx) | Standalone individual contribution report for Jeevan G A | **Complete (3 pages)** |
| **Standalone Contribution Doc** | [`reports/Individual_Contribution_Shashank_23BAI0152.docx`](file:///c:/Users/SKSOH/OneDrive/Desktop/DL_PROJECT/reports/Individual_Contribution_Shashank_23BAI0152.docx) | Standalone individual contribution report for Shashank Pinnelli | **Complete (3 pages)** |
| **Individualized Full Report** | [`reports/TRUST_X_Report_Sohil_23BAI0133.docx`](file:///c:/Users/SKSOH/OneDrive/Desktop/DL_PROJECT/reports/TRUST_X_Report_Sohil_23BAI0133.docx) | Complete 47-page report copy personalized for Sohil (Declaration, Certificate, Table E.1 highlighted) | **Verified (47 pages)** |
| **Individualized Full Report** | [`reports/TRUST_X_Report_Jeevan_23BAI0140.docx`](file:///c:/Users/SKSOH/OneDrive/Desktop/DL_PROJECT/reports/TRUST_X_Report_Jeevan_23BAI0140.docx) | Complete 47-page report copy personalized for Jeevan (Declaration, Certificate, Table E.1 highlighted) | **Verified (47 pages)** |
| **Individualized Full Report** | [`reports/TRUST_X_Report_Shashank_23BAI0152.docx`](file:///c:/Users/SKSOH/OneDrive/Desktop/DL_PROJECT/reports/TRUST_X_Report_Shashank_23BAI0152.docx) | Complete 47-page report copy personalized for Shashank (Declaration, Certificate, Table E.1 highlighted) | **Verified (47 pages)** |
| **Master Report (Team Copy)** | [`reports/TRUST_X_Deep_Learning_Project_Report_MASTER.docx`](file:///c:/Users/SKSOH/OneDrive/Desktop/DL_PROJECT/reports/TRUST_X_Deep_Learning_Project_Report_MASTER.docx) | Master report incorporating the full 4-column team matrix in Appendix Table E.1 | **Verified (47 pages)** |

---

## 2. Document-Level Integrity and University Compliance Verification

All generated full-report copies were repaginated and evaluated directly through Microsoft Word COM Automation (`Word.Application` with `Document.Repaginate()` and `ComputeStatistics(wdStatisticPages)`):

| Compliance Requirement | Target / Spec | Sohil Report | Jeevan Report | Shashank Report | Compliance Result |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Rendered Pages** | 20 – 50 pages (Target 45–48) | **47 pages** | **47 pages** | **47 pages** | **STRICTLY COMPLIANT** |
| **Visual Media Preservation** | 14 images (100%) | 14 images | 14 images | 14 images | **ZERO ASSETS LOST (100% byte match)** |
| **Tabular Preservation** | 29 tables (100%) | 29 tables | 29 tables | 29 tables | **ZERO TABLES LOST (100% preserved)** |
| **Paper Size & Margins** | A4; Left: 3.81cm, Right/Top/Bottom: 2.54cm | Compliant | Compliant | Compliant | **PASSED** |
| **Typography & Hierarchy** | TNR 12pt body, 1.5 spacing, 16pt bold Ch, 14pt Sec | Compliant | Compliant | Compliant | **PASSED** |
| **Table Header Readability** | Bold white text on `#1F4E79` | Compliant | Compliant | Compliant | **PASSED (0 dark-on-dark issues)** |

---

## 3. Verified Technical Responsibility Allocation

The technical responsibilities were established by cross-referencing the project's source code, execution pipelines, data tables, and experiment audit logs. A clear boundary is maintained between individual module ownership and collective team collaboration:

### 3.1 Member 1: Shaik Mohammad Sohil (23BAI0133)
* **Primary Role:** Lead Integration Engineer, Experimental Orchestrator & Web Application Developer.
* **Responsibilities Verified from Project Artifacts:**
  1. **Central Pipeline Orchestration:** Authored `run_all_experiments.py` and `train.py`, configuring command-line interfaces, argument parsing, automated logging, and sequential experiment scheduling.
  2. **Compute Environment Management:** Configured GPU runtime scripts (`scripts/create_colab_notebook.py`) on Google Colab Tesla T4, implementing CUDA memory clearing (`torch.cuda.empty_cache()`) and garbage collection between training runs to prevent out-of-memory crashes.
  3. **Training Harness Engineering:** Structured `src/training/trainer.py` and `src/models/model_factory.py`, configuring the AdamW optimizer with linear warmup, weight decay (0.01), sequence truncation (128 tokens), and validation-loss checkpoint saving.
  4. **Streamlit Web Application Architecture:** Designed and implemented the complete multi-page web application (`app/main.py`), cached data layer (`app/data_loader.py`), and interactive UI panels:
     - *Benchmark Dashboard:* Interactive factorial filtering across the 12 experiments.
     - *Multi-Model Live Inference:* Parallel 3-model inference with real-time toxicity threshold gauges and SHAP highlight rendering.
     - *Model Auditing:* Multi-pillar radar plots and latency-VRAM tradeoff visualizers.
     - *Sensitivity Explorer:* Real-time weight sliders updating architectural rankings dynamically.
* **Codebase Ownership:** `train.py`, `run_all_experiments.py`, `src/models/model_factory.py`, `src/training/trainer.py`, `app/main.py`, `app/data_loader.py`, `app/components/*.py`, `scripts/create_colab_notebook.py`.

### 3.2 Member 2: Jeevan G A (23BAI0140)
* **Primary Role:** Lead Data Engineer, Preprocessing Specialist & Data Quality Auditor.
* **Responsibilities Verified from Project Artifacts:**
  1. **Corpus Ingestion & Schema Normalization:** Handled diverse raw formats (.txt, .xlsx, TSV) spanning Telugu HOLD, Tamil DravidianCodeMix, Malayalam DravidianCodeMix, and Hindi HECM, formulating the unified schema (`id, text, label, original_label, language`).
  2. **Data Sanitization & Text Normalization:** Authored `src/data/preprocessing.py`, implementing regex cleaning to strip URLs, user handles, and encoding artifacts while strictly preserving code-mixed orthography, script transliterations, and emotional punctuation.
  3. **Duplicate Detection & Conflict Resolution:** Identified and purged 51 exact duplicate comments in Telugu HOLD (reducing 4,000 raw samples to 3,941 clean samples). Programmatically audited and excluded 25 contradictory text groups (143 lines) in Hindi HECM. Filtered non-regional language instances (`not-Tamil`: 1,786 rows; `not-malayalam`: 1,607 rows).
  4. **Data Partitioning & Leakage Prevention:** Built stratified 70/15/15 train/dev/test splits across all languages (`scripts/prepare_hecm_hindi.py`, `src/data/loaders.py`). Quarantined the 500 unlabelled Telugu test rows to ensure zero evaluation leakage.
  5. **Data Imbalance & Token Profiling:** Authored `src/data/audit.py`, generating `tables/dataset_audit.csv`, `tables/class_distributions.csv`, and Chapter 4 figures (Figure 4.1, Figure 4.2, Figure 4.3). Verified the critical **1,844 test sample count** in Malayalam and documented its 24:1 class skew.
* **Codebase Ownership:** `src/data/audit.py`, `src/data/preprocessing.py`, `src/data/loaders.py`, `run_preprocessing.py`, `scripts/prepare_hecm_hindi.py`, `scripts/run_audit*.py`, `tables/dataset_audit.csv`, `tables/class_distributions.csv`, `tables/processed_datasets_summary.csv`, `tables/hecm_hindi_*.csv`, `reports/dataset_audit_report.md`.

### 3.3 Member 3: Shashank Pinnelli (23BAI0152)
* **Primary Role:** Lead Trustworthiness & Evaluation Researcher, Analytical Reporter.
* **Responsibilities Verified from Project Artifacts:**
  1. **Multi-Pillar Trustworthiness Formulation:** Formulated the mathematical foundations and normalizations of the TRUST-X framework ($P, E, R, C$) and defined the composite Trustworthiness Index ($TWI = w_{\text{perf}} P + w_{\text{expl}} E + w_{\text{rob}} R + w_{\text{calib}} C$) in `src/twi/twi.py`.
  2. **Predictive Performance Evaluation:** Developed `src/evaluation/metrics.py`, computing Accuracy, Precision, Recall, Macro-F1, Weighted-F1, and class-wise performance matrices across all 12 experiments.
  3. **Explainability & Attribution Compactness:** Authored `src/explainability/shap_analysis.py`, integrating the SHAP Text Explainer, sampling representative error subsets (TP, TN, FP, FN), and formulating the top-20% explanation compactness score.
  4. **Confidence Calibration Modeling:** Developed `src/calibration/calibration.py`, computing Expected Calibration Error (ECE across 10 equal-width bins) and Brier scores on raw test softmax probabilities. Implemented `ModelWithTemperature` for validation-set temperature scaling investigations.
  5. **Linguistic Robustness Engine:** Developed `src/robustness/perturbations.py` and `src/robustness/evaluate.py`, implementing 6 realistic social media noise operators (leet substitution, character deletion, repetition, punctuation insertion, whitespace perturbation, composite perturbation) and evaluating F1 retention and prediction flip rate ($R = 0.6 \text{ retention} + 0.4 (1 - \text{flip rate})$).
  6. **Sensitivity Auditing & Analytical Reporting:** Executed multi-profile sensitivity analysis across Balanced, Performance-Heavy, Robustness-Heavy, and Calibration-Heavy profiles, proving that the ranking XLM-R > mBERT > IndicBERT remains strictly invariant. Authored results discussion (Chapter 7), diagnostic error analysis, and master result tables.
* **Codebase Ownership:** `src/evaluation/metrics.py`, `src/calibration/calibration.py`, `src/robustness/perturbations.py`, `src/robustness/evaluate.py`, `src/explainability/shap_analysis.py`, `src/twi/twi.py`, `tables/final_results.csv`, `tables/model_averages.csv`, `tables/language_averages.csv`, `tables/twi_sensitivity*.csv`, `reports/FINAL_EXPERIMENT_FREEZE.md`, `reports/TWI_SENSITIVITY_ANALYSIS.md`.

---

## 4. Comprehensive Team Contribution Matrix (Appendix Table E.1)

The following 4-column matrix is integrated into Appendix E of the project report, explicitly distinguishing individual lead and supporting roles across the 9 core project activities:

| Project Activity / Domain | Shaik Mohammad Sohil (23BAI0133) | Jeevan G A (23BAI0140) | Shashank Pinnelli (23BAI0152) |
| :--- | :--- | :--- | :--- |
| **1. Project Integration & Orchestration** | **Lead:** CLI runner, Colab GPU orchestrator (`train.py`, `run_all_experiments.py`), memory clearing, seed 42. | Integrated preprocessing pipeline runner (`run_preprocessing.py`) and schema mapping. | Integrated evaluation metrics pipeline and experimental results serialization. |
| **2. Dataset Auditing & Preprocessing** | Validation split verification, Colab dataset caching, and data loader testing. | **Lead:** Raw corpus acquisition, regex cleaning, duplicate purging (51 Telugu, 25 Hindi groups), stratified splits. | Verification of class-weight inputs, label distribution auditing, and evaluation split integrity. |
| **3. Transformer Fine-Tuning & Training** | **Lead:** Hyperparameter tuning (lr=2e-5, AdamW, warmup), custom Trainer wrapper, CUDA memory clearing. | Class-weight tensor computation for weighted CE loss handling severe Malayalam imbalance (24:1 skew). | Checkpoint selection criteria based on validation Macro-F1; seed 42 reproducibility validation. |
| **4. Predictive Performance Evaluation** | Model inference batching, prediction latency profiling, and peak VRAM tracking. | Per-language split evaluation auditing, test sample accounting (1,844 Malayalam test verification). | **Lead:** Macro-F1, Accuracy, Precision, Recall, Confusion Matrices, and class-wise performance breakdown. |
| **5. SHAP Explainability & Compactness** | Integration of SHAP visual heatmaps into the Streamlit web application. | Extraction of representative samples (TP, TN, FP, FN) preserving code-mixed token boundaries. | **Lead:** SHAP Text Explainer integration, token-level attribution computation, and top-20% compactness scoring. |
| **6. Confidence Calibration & Robustness** | Live calibration gauge displays and adversarial perturbation testing in Streamlit. | Typographical error distribution profiling in social media code-mixed comments. | **Lead:** ECE (10 bins), Brier score, reliability diagrams, 6 synthetic perturbation operators, and F1 retention. |
| **7. Trustworthiness Index (TWI) & Sensitivity** | Interactive TWI sensitivity explorer implementation with dynamic weight sliders in Streamlit. | Cross-language trust profile characterization and linguistic difficulty correlation. | **Lead:** Mathematical formulation of TWI ($P, E, R, C$ normalizations) and 4-profile sensitivity rank invariance. |
| **8. Streamlit Web Application Development** | **Lead:** Multi-page web application architecture (`app/`), benchmark dashboard, live 3-model parallel inference. | Data loading layer caching (`app/data_loader.py`), corpus sample viewer integration. | Visualization components (radar spider plots, calibration reliability curves, error analysis panels). |
| **9. Technical Documentation & Reporting** | Co-author: System architecture (Ch 3), implementation details (Ch 6), web application documentation (Ch 8). | Co-author: Dataset provenance & preprocessing (Ch 4), class distribution analysis, data audit reports. | Co-author: Empirical results & discussion (Ch 7), false-positive diagnostic analysis, conclusions (Ch 9). |

---

## 5. Items for Student Confirmation Prior to Submission

Before submitting the individualized project reports to the faculty portal, the team members should confirm the following administrative and logistical items:

1. **Student Signatures:**
   - Print or digitally sign the Declaration on physical page 3 of each individual report copy.
   - Confirm that the student registration numbers (`23BAI0133`, `23BAI0140`, `23BAI0152`) match official university records.
2. **Supervisor Title Verification:**
   - Confirm that the guide designation (`Dr. CHELLATAMILAN T, Professor Grade 1, School of Computer Science and Engineering (SCOPE)`) matches current departmental listings.
3. **Electronic Code Submission:**
   - Confirm that the GitHub code repository link or ZIP submission archive contains the identical codebase verified in this audit (`train.py`, `src/`, `app/`, `tables/`, `figures/`).
4. **Demonstration Readiness:**
   - Ensure that the Streamlit application (`app/main.py`) runs locally via `streamlit run app/main.py` using the cached checkpoints and data tables for oral evaluation.

---

## 6. Final Sign-off

- [x] Three standalone Individual Contribution documents generated.
- [x] Three individualized, complete 47-page report copies generated with personalized declarations and certificates.
- [x] All 14 visual images and 29 tables preserved across all document copies.
- [x] Word COM automation repagination confirms strict compliance with the $\le 50$-page university limit.
- [x] Canonical experimental metrics, datasets, and source files unmodified.
- [x] Comprehensive contribution matrix verified and integrated into Appendix Table E.1.

**All deliverables are complete, verified, and ready for individual student submission.**

