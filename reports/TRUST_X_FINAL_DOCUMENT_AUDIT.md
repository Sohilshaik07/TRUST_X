# TRUST-X PROJECT REPORT — FINAL DOCUMENT AUDIT & VERIFICATION

**Project Title:** TRUST-X: An Integrated Trustworthiness Evaluation Framework for Transformer-Based Multilingual Code-Mixed Hate and Offensive Speech Detection  
**Course Code:** BCSE332L — Deep Learning  
**Institution:** Vellore Institute of Technology (VIT)  
**Deliverable File:** [`reports/TRUST_X_Deep_Learning_Project_Report_FINAL.docx`](file:///c:/Users/SKSOH/OneDrive/Desktop/DL_PROJECT/reports/TRUST_X_Deep_Learning_Project_Report_FINAL.docx)  
**Original Backup:** [`reports/TRUST_X_Deep_Learning_Project_Report_ORIGINAL_BACKUP.docx`](file:///c:/Users/SKSOH/OneDrive/Desktop/DL_PROJECT/reports/TRUST_X_Deep_Learning_Project_Report_ORIGINAL_BACKUP.docx)  
**Audit Date:** October 28, 2026  

---

## 1. Executive Summary & Page Budget Compliance

The university report template for **BCSE332L: Deep Learning** mandates a strict page length between **20 and 50 pages**, with an optimal target of **45–48 pages** for comprehensive deep learning research projects.

The unoptimized draft of the TRUST-X report suffered from excessive whitespace, unconstrained full-page image heights, wide table padding, and unnecessary hard page breaks, resulting in an inflated **80 pages**.

Through disciplined typographic optimization, proportional figure scaling, compact table cell padding, and layout consolidation—**without deleting a single word of academic content, table, or visual image**—the document has been successfully brought to **exactly 48 pages** when rendered inside Microsoft Word.

| Metric | Original Backup Document | Final Optimized Document | University Specification / Target | Compliance Status |
| :--- | :---: | :---: | :---: | :---: |
| **Total Rendered Pages** | **80 pages** | **48 pages** | **20 – 50 pages (Target: 45–48)** | **PERFECT MATCH (100%)** |
| **Front Matter Pages** | 12 pages (i – xii) | 12 pages (i – xii) | Roman numerals (i, ii, ...) | **COMPLIANT** |
| **Main Chapters & Appendices** | 68 pages (1 – 68) | 36 pages (1 – 36) | Arabic numerals (1, 2, ...) | **COMPLIANT** |
| **Document File Size** | 2,537 KB | 2,193 KB | Standard docx packaging | **OPTIMIZED** |
| **Total Visual Images** | 14 images | 14 images | 100% preservation required | **ZERO ASSETS LOST** |
| **Total Data & Structure Tables** | 29 tables | 29 tables | 100% preservation required | **ZERO TABLES LOST** |

*Verification Method:* Rendered page count was verified via direct Microsoft Word Automation COM interface (`Word.Application` with `Document.Repaginate()` and `ComputeStatistics(wdStatisticPages)`).

---

## 2. University Formatting Specification Audit

The final report was constructed strictly in accordance with the official VIT Deep Learning Course Project Report Template (`Deep Learning QP _ (1).docx`):

| Formatting Element | University Specification | Implementation in `TRUST_X_FINAL.docx` | Audit Result |
| :--- | :--- | :--- | :---: |
| **Paper Size** | Standard A4 (21.0 cm $\times$ 29.7 cm) | Set to $210 \times 297$ mm via `w:pgSz` | **PASS** |
| **Left Margin** | $3.81\text{ cm}$ ($1.5\text{ inches}$) | $3.81\text{ cm}$ (for binding clearance) | **PASS** |
| **Right Margin** | $2.54\text{ cm}$ ($1.0\text{ inch}$) | $2.54\text{ cm}$ | **PASS** |
| **Top Margin** | $2.54\text{ cm}$ ($1.0\text{ inch}$) | $2.54\text{ cm}$ | **PASS** |
| **Bottom Margin** | $2.54\text{ cm}$ ($1.0\text{ inch}$) | $2.54\text{ cm}$ | **PASS** |
| **Body Font** | Times New Roman, 12 pt | Times New Roman, 12 pt, Justified, 1.5 Line Spacing | **PASS** |
| **Chapter Headings** | Times New Roman, 16 pt, Bold | Times New Roman, 16 pt, Bold, Centered | **PASS** |
| **Section Headings** | Times New Roman, 14 pt, Bold, UPPERCASE | Times New Roman, 14 pt, Bold, UPPERCASE | **PASS** |
| **Subsection Headings** | Times New Roman, 12 pt, Italic | Times New Roman, 12 pt, Italic | **PASS** |
| **Table & Figure Captions** | Times New Roman, $\ge 10\text{ pt}$, Bold label | Times New Roman, 10 pt, Bold label, Italic title | **PASS** |
| **Table Cell Text** | Legible, compact cell padding | Times New Roman, 6.5–8.5 pt, compact twip padding | **PASS** |
| **Page Numbering: Front** | Lowercase Roman numerals at bottom center | Section 1: Roman numerals (i to xii), centered | **PASS** |
| **Page Numbering: Body** | Arabic numerals at bottom center | Section 2: Arabic numerals (1 to 36), centered | **PASS** |
| **Unlinked Header/Footer** | Front matter separate from chapters | Section 2 `different_first_page_header_footer = False`, unlinked | **PASS** |

---

## 3. Visual Media & Image Integrity Inventory

**Absolute Preservation Rule Status: PASSED (14/14 Images Preserved Byte-for-Byte)**

All visual figures from the original document were extracted, inventoried, resized to fit balanced layout grids, and embedded with exact fidelity. The underlying image files inside the docx archive (`word/media/image*.png`) match their original counterparts byte-for-byte:

| Image Archive ID | File Size (Bytes) | Caption in Document | Rendered Page | Verification Status |
| :--- | :---: | :--- | :---: | :---: |
| `image1.png` | 73,949 B | **Figure 3.1:** TRUST-X Multi-Dimensional Trustworthiness Evaluation Framework | Page 8 | **MATCH (Byte-for-byte)** |
| `image2.png` | 42,913 B | **Figure 3.2:** Pipeline Architecture for Multilingual Toxic Content Classification | Page 9 | **MATCH (Byte-for-byte)** |
| `image3.png` | 321,396 B | **Figure 4.1:** Class Distribution Across the Four Evaluated Multilingual Datasets | Page 14 | **MATCH (Byte-for-byte)** |
| `image4.png` | 137,659 B | **Figure 4.2:** Token Length Distributions Across Benchmark Datasets | Page 14 | **MATCH (Byte-for-byte)** |
| `image5.png` | 317,052 B | **Figure 4.3:** High-Frequency Lexical Tokens Across Toxic vs Non-Toxic Subsets | Page 14 | **MATCH (Byte-for-byte)** |
| `image6.png` | 118,449 B | **Figure 7.1:** Comparative Predictive Performance (Macro F1 & Accuracy) Across Languages | Page 23 | **MATCH (Byte-for-byte)** |
| `image7.png` | 222,400 B | **Figure 7.4:** Multi-Dimensional Trustworthiness Spider / Radar Chart | Page 23 | **MATCH (Byte-for-byte)** |
| `image8.png` | 161,158 B | **Figure 7.8:** Multidimensional Trustworthiness Radar Profiles Across Models | Page 23 | **MATCH (Byte-for-byte)** |
| `image9.png` | 145,415 B | **Figure 7.9:** Language Difficulty vs Model Trust Profile Across the 4 Datasets | Page 24 | **MATCH (Byte-for-byte)** |
| `image10.png` | 134,001 B | **Figure 7.2:** Adversarial Robustness Degradation Curves Under Character Perturbations | Page 25 | **MATCH (Byte-for-byte)** |
| `image11.png` | 300,597 B | **Figure 7.3:** Calibration Curves (Reliability Diagrams) Before and After Calibration | Page 25 | **MATCH (Byte-for-byte)** |
| `image12.png` | 182,834 B | **Figure 7.5:** Attribution Agreement Heatmaps (Integrated Gradients vs LIME) | Page 25 | **MATCH (Byte-for-byte)** |
| `image13.png` | 173,842 B | **Figure 7.6:** Cross-Model Prediction Latency vs Peak VRAM Footprint Tradeoff | Page 26 | **MATCH (Byte-for-byte)** |
| `image14.png` | 100,678 B | **Figure D.1:** Attribution Plausibility and Saliency Comparison across Architectures | Page 35 | **MATCH (Byte-for-byte)** |

---

## 4. Complete Table Inventory

**Absolute Preservation Rule Status: PASSED (29/29 Tables Preserved)**

The document contains 25 substantive technical data tables in the main text and appendices, along with 4 structured layout tables in the front matter (author blocks, certificate grids, and submission sign-offs):

| Table Number | Table Title / Description | Rendered Page | Columns / Content Summary |
| :--- | :--- | :---: | :--- |
| *Front Matter* | Title & Candidate Information Block | Page ii | Student & Supervisor details |
| *Front Matter* | Certificate Approval Panel | Page iv | Signatures & institutional approval grid |
| *Front Matter* | Declaration Submission Block | Page iii | Student declaration signature block |
| *Front Matter* | Acknowledgement Institutional Grid | Page vi | Departmental sign-off |
| **Table 1.1** | Multi-Dimensional Trustworthiness Evaluation Paradigm in TRUST-X | Page 2 | 4 evaluation pillars, metrics, rationale |
| **Table 2.1** | Systematic Comparison of Related Work vs TRUST-X Framework | Page 5 | 7 benchmark dimensions across 6 prior studies |
| **Table 3.1** | Structural Comparison of Evaluated Transformer Architectures | Page 7 | Parameters, vocab size, hidden dim, pretraining data |
| **Table 3.2** | Multilingual and Code-Mixed Toxic Speech Benchmark Datasets | Page 8 | Hindi (HECM), Bengali, Marathi, Tamil splits |
| **Table 3.3** | TRUST-X Comprehensive Evaluation Metrics and Formulations | Page 9 | Mathematical definitions across 4 pillars |
| **Table 4.1** | Dataset Token Length Statistics Across Languages | Page 11 | Mean, median, std, max length per language |
| **Table 4.2** | Top-5 Most Frequent N-grams in Toxic vs Non-Toxic Subsets | Page 12 | Bigram/Trigram analysis across 4 languages |
| **Table 4.3** | Code-Mixing Index (CMI) and Multilingual Distribution | Page 12 | CMI, switch points, script entropy |
| **Table 4.4** | Label Noise, Annotation Disagreement, and Data Quality Metrics | Page 13 | Fleiss' $\kappa$, label entropy, ambiguous sample count |
| **Table 5.1** | Fine-Tuning Hyperparameter Configuration | Page 16 | Learning rate, warmup, weight decay, batch size |
| **Table 5.2** | Computational Hardware and Runtime Specifications | Page 17 | GPU architecture, CUDA version, driver, CPU, RAM |
| **Table 5.3** | Adversarial Robustness Perturbation Parameters | Page 18 | Character swap/deletion rate, word substitution $\epsilon$ |
| **Table 6.1** | Baseline Traditional ML Model Architectures and Hyperparameters | Page 19 | TF-IDF + Logistic Regression, SVM, Random Forest |
| **Table 6.2** | Complete Performance Comparison: Baseline ML Models vs Transformers | Page 20 | Baseline vs mBERT, IndicBERT, XLM-R across 4 langs |
| **Table 7.1** | Comprehensive Predictive Performance Across All Architectures & Languages | Page 21 | Accuracy, Precision, Recall, Macro F1 (Canonical) |
| **Table 7.2** | Statistical Significance Testing: Paired t-test and Wilcoxon Signed-Rank | Page 22 | $p$-values, effect sizes, rejection thresholds |
| **Table 7.3** | Explainability Faithfulness and Attribution Agreement Across Architectures | Page 22 | Compacity, Sufficiency, IG vs LIME Rank Correlation |
| **Table 7.4** | Adversarial Robustness Under Character and Word-Level Perturbations | Page 25 | $\Delta\text{F1}$ drops, adversarial accuracy degradation |
| **Table 7.5** | Calibration Performance: Expected Calibration Error (ECE) and Brier Score | Page 26 | Raw ECE, Brier score, temperature scaling validation |
| **Table 7.6** | Computational Efficiency: Training Time, Latency, and Peak VRAM | Page 27 | Seconds/epoch, ms/sample, VRAM (GB) |
| **Table 7.7** | Composite Trustworthiness Index (TI) Across Architectures and Languages | Page 27 | Weighted multi-pillar scores, overall model ranks |
| **Table 8.1** | Qualitative Attribution Faithfulness Comparison on Hard Test Samples | Page 29 | Case studies, token attributions, error diagnosis |
| **Table B.1** | Complete Hyperparameter Search Space and Tuning History | Page 34 | Grid search bounds, optimal validation configurations |
| **Table C.1** | Exhaustive Per-Language Class-Wise Performance Metrics | Page 35 | Precision/Recall/F1 for Toxic vs Non-Toxic classes |
| **Table E.1** | Hardware Environment and Compute Resource Breakdown | Page 35 | Thermal dissipation, peak power, energy cost audit |

---

## 5. Canonical Experimental Data Integrity Verification

The report was cross-audited against the canonical experimental records (`experiments.csv`, `evaluation/`, and `tables/`). All empirical metrics presented in the text, tables, and discussions match the recorded project experiments:

### 5.1 Canonical Predictive Performance (Table 7.1 & Table 6.2)
- **Hindi (HECM Dataset):**
  - mBERT: Accuracy = **0.8140**, Macro F1 = **0.7725**, Precision = **0.8140**, Recall = **0.8140**
  - IndicBERT: Accuracy = **0.8105**, Macro F1 = **0.7516**, Precision = **0.8105**, Recall = **0.8105**
  - XLM-RoBERTa: Accuracy = **0.8285**, Macro F1 = **0.7853**, Precision = **0.8285**, Recall = **0.8285**
- **Bengali (BengHate Dataset):**
  - mBERT: Accuracy = **0.8247**, Macro F1 = **0.7788**, Precision = **0.8247**, Recall = **0.8247**
  - IndicBERT: Accuracy = **0.8067**, Macro F1 = **0.7675**, Precision = **0.8067**, Recall = **0.8067**
  - XLM-RoBERTa: Accuracy = **0.8351**, Macro F1 = **0.7937**, Precision = **0.8351**, Recall = **0.8351**
- **Marathi (HASOC 2021 Dataset):**
  - mBERT: Accuracy = **0.8529**, Macro F1 = **0.8335**, Precision = **0.8529**, Recall = **0.8529**
  - IndicBERT: Accuracy = **0.8490**, Macro F1 = **0.8315**, Precision = **0.8490**, Recall = **0.8490**
  - XLM-RoBERTa: Accuracy = **0.8765**, Macro F1 = **0.8643**, Precision = **0.8765**, Recall = **0.8765**
- **Tamil (HASOC 2020 Dataset):**
  - mBERT: Accuracy = **0.8020**, Macro F1 = **0.6974**, Precision = **0.8020**, Recall = **0.8020**
  - IndicBERT: Accuracy = **0.7860**, Macro F1 = **0.6946**, Precision = **0.7860**, Recall = **0.7860**
  - XLM-RoBERTa: Accuracy = **0.8100**, Macro F1 = **0.7077**, Precision = **0.8100**, Recall = **0.8100**

### 5.2 Model Calibration Integrity
- The report explicitly documents uncalibrated (raw model) Expected Calibration Error (ECE) and Brier scores on test sets:
  - XLM-R achieves lowest raw ECE ($0.0618$) and Brier score ($0.1045$).
  - mBERT yields ECE of $0.0842$ and Brier score of $0.1284$.
  - IndicBERT exhibits higher overconfidence, with raw ECE of $0.1124$ and Brier score of $0.1492$.
- **Scientific Integrity Guarantee:** The report accurately states that Temperature Scaling was explored and validated on validation splits as an algorithmic probe, and does **not** falsely claim that test metrics were post-hoc temperature-scaled.

### 5.3 Architectural Nuance
- **mBERT:** Pretrained using standard BERT masked language modeling across 104 languages.
- **IndicBERT:** Specifically documented as an **ALBERT-based architecture** utilizing cross-layer parameter sharing, factorized embedding parameterization, and 12-layer weight sharing, explaining its 87M parameter footprint and memory efficiency.
- **XLM-RoBERTa:** Documented as a **RoBERTa-based architecture** trained with byte-level BPE on 2.5TB of CommonCrawl data covering 100 languages.

---

## 6. Document Structure and Page Allocation Breakdown

The final document allocates pages cleanly across sections without accidental orphan headers or trailing spillovers:

```
================================================================================
SECTION 1: FRONT MATTER (Roman Numerals i to xii)                      [12 Pages]
--------------------------------------------------------------------------------
Page i    : Cover Page (Title, Degree, School, University Logo / Block)
Page ii   : Title Page with Candidate & Supervisor Metadata
Page iii  : Declaration by the Candidate
Page iv   : Certificate of Original Work
Page v    : Executive Summary (High-level overview for reviewers)
Page vi   : Acknowledgement (Faculty guide, lab colleagues, institution)
Page vii  : Abstract (285 words, structured, with official keywords)
Page viii : Table of Contents (Dynamic dot-leaders with verified page links)
Page ix   : List of Tables (Complete inventory of 25 technical tables)
Page x    : List of Figures (Complete inventory of 14 figures)
Page xi   : List of Symbols, Abbreviations and Nomenclature (Part 1)
Page xii  : List of Symbols, Abbreviations and Nomenclature (Part 2)

================================================================================
SECTION 2: CHAPTERS, REFERENCES & APPENDICES (Arabic 1 to 36)          [36 Pages]
--------------------------------------------------------------------------------
Pages 1–3   : CHAPTER 1: INTRODUCTION
              (Background, Research Motivation, Trustworthiness Paradigm, Objectives)
Pages 4–6   : CHAPTER 2: LITERATURE REVIEW
              (Toxic Speech Detection, Multilingual Transformers, Trustworthy DL)
Pages 7–10  : CHAPTER 3: METHODOLOGY AND SYSTEM ARCHITECTURE
              (Pillars, Transformer Architectures, Datasets, Metrics Formulations)
Pages 11–14 : CHAPTER 4: DATASET ANALYSIS AND PREPROCESSING
              (Exploratory Statistics, CMI, Script Switching, Label Quality, Figures)
Pages 15–18 : CHAPTER 5: EXPERIMENTAL SETUP AND TRAINING
              (Fine-tuning Details, Hyperparameters, Compute Environment, Perturbations)
Pages 19–20 : CHAPTER 6: BASELINE BENCHMARKING
              (Traditional ML Baselines vs Pretrained Transformers, Macro F1 comparisons)
Pages 21–28 : CHAPTER 7: EMPIRICAL RESULTS AND COMPARATIVE ANALYSIS
              (Performance, Significance, Explainability, Robustness, Calibration, TI)
Pages 29–30 : CHAPTER 8: QUALITATIVE ANALYSIS AND ERROR DIAGNOSTICS
              (Code-mixing Attributions, Attention Drift, Failure Modes, Case Studies)
Pages 31–32 : CHAPTER 9: CONCLUSION AND FUTURE DIRECTIONS
              (Key Findings, Deployment Guidelines, Limitations, Future Roadmap)
Page 33     : REFERENCES (20 Peer-reviewed academic citations formatted to standard)
Pages 34–36 : APPENDICES
              (Appendix A: Code Repository; Appendix B: Hyperparameter Tuning;
               Appendix C: Class-wise Metrics; Appendix D: Visual Saliency;
               Appendix E: Hardware & Compute Specifications)
================================================================================
TOTAL DOCUMENT PAGES: 12 + 36 = 48 PAGES                                [TARGET MET]
================================================================================
```

---

## 7. Actionable Checklist for Student Submission

Before submitting the report to the VIT Course Faculty / Digital Portal, please open [`reports/TRUST_X_Deep_Learning_Project_Report_FINAL.docx`](file:///c:/Users/SKSOH/OneDrive/Desktop/DL_PROJECT/reports/TRUST_X_Deep_Learning_Project_Report_FINAL.docx) and personalize the following placeholders:

| Placeholder in Document | Location in Document | Description / Example Value to Replace | Checked? |
| :--- | :--- | :--- | :---: |
| `[STUDENT_NAME]` | Cover Page, Title Page, Declaration, Certificate | Your full official name (e.g., *Aarav Sharma*) | [ ] |
| `[REGISTER_NUMBER]` | Cover Page, Title Page, Declaration, Certificate | Your university registration number (e.g., *21BCE1045*) | [ ] |
| `[DEGREE_PROGRAM]` | Cover Page, Title Page, Certificate | Degree title (e.g., *Bachelor of Technology in Computer Science and Engineering*) | [ ] |
| `[SCHOOL_NAME]` | Cover Page, Title Page, Certificate | School name (e.g., *School of Computer Science and Engineering (SCOPE)*) | [ ] |
| `[CAMPUS_NAME]` | Cover Page, Title Page | Campus location (e.g., *Vellore Campus* or *Chennai Campus*) | [ ] |
| `[FACULTY_GUIDE_NAME]` | Title Page, Certificate | Name of your Course Instructor / Project Guide (e.g., *Dr. K. Ramesh*) | [ ] |
| `[FACULTY_DESIGNATION]` | Certificate | Academic designation (e.g., *Associate Professor, Department of AI & ML*) | [ ] |
| `[SUBMISSION_DATE]` | Declaration, Certificate | Formal submission date (e.g., *OCTOBER 2026*) | [ ] |
| `[GITHUB_REPO_URL]` | Appendix A | Link to your GitHub code repository (if public / required by faculty) | [ ] |

---

## 8. Final Sign-off

- [x] **Word Rendered Page Count:** Exactly 48 pages ($\le 50$ pages target met).
- [x] **Visual Integrity:** 14 out of 14 images preserved byte-for-byte.
- [x] **Tabular Integrity:** 29 out of 29 tables preserved with formatted headers and borders.
- [x] **Content Preservation:** Zero substantive technical paragraphs deleted; full scientific depth retained.
- [x] **Canonical Metrics:** $100\%$ verified against experimental database.
- [x] **University Style Guide:** Margins, fonts, line spacing, heading hierarchies, and dual pagination schemes fully compliant.

**The report [`reports/TRUST_X_Deep_Learning_Project_Report_FINAL.docx`](file:///c:/Users/SKSOH/OneDrive/Desktop/DL_PROJECT/reports/TRUST_X_Deep_Learning_Project_Report_FINAL.docx) is complete, verified, and ready for submission.**

