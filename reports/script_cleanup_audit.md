# TRUST-X Project Script Cleanup Audit & Inventory

**Project Root:** `DL_PROJECT`  
**Date of Audit:** October 29, 2026  
**Auditor:** Antigravity Automated Verification Engine  
**Objective:** Identify, categorize, and safely remove scripts created exclusively for Word report generation, individual contribution reports, pagination optimization, report auditing, or Review-2 PowerPoint presentation generation, while strictly preserving all Machine Learning pipeline, dataset processing, model training, evaluation, explainability, calibration, robustness, and Streamlit application assets.

---

## 1. CATEGORIZATION METHODOLOGY

Every script residing in `scripts/` was analyzed by reading its source code, imports, external dependencies, and cross-project references. Scripts are classified into four mutually exclusive categories:

- **Category A: REPORT / PRESENTATION GENERATION**  
  Scripts exclusively created to generate, edit, format, paginate, repair, or assemble Word reports (`.docx`), individual contribution reports, or PowerPoint presentations (`.pptx`).
- **Category B: REPORT / PRESENTATION AUDITING**  
  Scripts exclusively used to inspect report/presentation formatting, page counts, XML drawings, embedded figures, table headers, or document consistency.
- **Category C: CORE ML PIPELINE (STRICTLY PRESERVED)**  
  Scripts required for data ingestion, dataset preprocessing, dataset auditing, model training, hyperparameter orchestration, evaluation, metric formula verification, checkpoint inspection, or Streamlit app support.
- **Category D: UNCERTAIN / MIXED PURPOSE (CONSERVATIVELY PRESERVED)**  
  Scripts whose purpose overlaps with research, or whose classification cannot be conclusively established as report-exclusive.

---

## 2. COMPLETE SCRIPT INVENTORY (53 SCRIPTS AUDITED)

| # | Script Filename | Actual Purpose (Based on Source Code Inspection) | Category | Safe to Delete? | Justification / Reason |
| :-: | :--- | :--- | :-: | :-: | :--- |
| **1** | `analyze_section_lengths.py` | Parses `TRUST_X_Deep_Learning_Project_Report_ORIGINAL_BACKUP.docx` using `python-docx` to count words, tables, and figures per chapter. | **B** | **YES** | Exclusively used to inspect Word report document length; no ML functionality. |
| **2** | `append_remaining_chapters.py` | Appends Chapters 7–9, References, and Appendices into `generate_optimized_report.py`. | **A** | **YES** | One-off report generator assembly helper; report is already generated and finalized. |
| **3** | `apply_final_47_pages.py` | Applies surgical paragraph and spacing modifications to `generate_optimized_report.py` to hit 47 pages. | **A** | **YES** | Word report pagination tuning only. |
| **4** | `audit_all_reports.py` | Uses `python-docx` and `zipfile` to verify paragraphs, tables, images, and Table E.1 across student report documents. | **B** | **YES** | Report document validation utility only. |
| **5** | `audit_corrected_docx.py` | Inspects XML shading, font colors, and media hashes of `TRUST_X_Deep_Learning_Project_Report_FINAL_CORRECTED.docx`. | **B** | **YES** | Document formatting and readability verification only. |
| **6** | `audit_drawings.py` | Inspects `w:drawing` and `a:blip` XML tags in `reports/TRUST_X_Deep_Learning_Project_Report_ORIGINAL_BACKUP.docx`. | **B** | **YES** | Word document drawing audit only. |
| **7** | `audit_existing_doc.py` | Audits margins, sections, table counts, inline shapes, and word counts of original backup Word document. | **B** | **YES** | Word report inspection utility only. |
| **8** | `audit_run_all.py` | Verifies processed dataset sample sizes across splits, YAML configuration uniqueness, and checkpoint directories. | **C** | **NO (KEEP)** | **Core ML Pipeline / Pre-training audit:** Verifies dataset integrity and prevents checkpoint collisions. |
| **9** | `build_all_contribution_reports.py` | Reads master Word report and generates three individualized full reports for Sohil, Jeevan, and Shashank. | **A** | **YES** | Report generation script; reports already generated in `reports/`. |
| **10** | `check_individual_page_counts.ps1` | PowerShell script invoking Word COM automation to query page counts of individual contribution `.docx` files. | **B** | **YES** | Word COM pagination check script only. |
| **11** | `compact_spillover.py` | Modifies chapter text in `generate_optimized_report.py` to eliminate multi-page spillovers. | **A** | **YES** | Word document pagination tuning only. |
| **12** | `condense_for_47.py` | Trims redundant phrasing in `generate_optimized_report.py` chapters to hit exact 47-page target. | **A** | **YES** | Word report text condensation helper only. |
| **13** | `create_colab_notebook.py` | Generates the `notebooks/TRUST_X_Colab_Runner.ipynb` Jupyter Notebook used for running experiments on Google Colab GPUs. | **C** | **NO (KEEP)** | **Core ML Research Tool:** Rebuilds or maintains the Colab training environment. |
| **14** | `extract_all_media.py` | Extracts all embedded media from `TRUST_X_Deep_Learning_Project_Report_ORIGINAL_BACKUP.docx` into `reports/extracted_images/`. | **A** | **YES** | Extraction complete; extracted images already safely saved in `reports/extracted_images/`. |
| **15** | `finalize_47_pages.py` | Sets font sizes, margins, and table padding in `generate_optimized_report.py` to land on 47 pages. | **A** | **YES** | Word document pagination tuning only. |
| **16** | `find_add_figures.py` | Regex search scanning `scripts/generate_project_report.py` for `add_figure` function calls. | **B** | **YES** | Internal helper for analyzing report generator code. |
| **17** | `find_fig3.py` | Line search scanning `scripts/generate_project_report.py` for Figure 3 references. | **B** | **YES** | Internal helper for analyzing report generator code. |
| **18** | `format_toc_2tuple.py` | Reformats Table of Contents data structures in `generate_optimized_report.py`. | **A** | **YES** | Word report formatting helper only. |
| **19** | `generate_final_analysis.py` | Ingests `experiments.csv` and produces `tables/final_results.csv`, `model_averages.csv`, `language_averages.csv`, and all canonical plots in `figures/final/`. | **C** | **NO (KEEP)** | **Core ML Analysis Engine:** Generates official evaluation tables and benchmark figures from raw experiment outputs. |
| **20** | `generate_final_corrected_report.py` | Large standalone script (182 KB) generating the 47-page academic Word report `TRUST_X_Deep_Learning_Project_Report_FINAL_CORRECTED.docx`. | **A** | **YES** | Report generator; final report already created and verified in `reports/`. |
| **21** | `generate_individual_contribution_docs.py` | Generates standalone Individual Contribution Word documents (`Individual_Contribution_*.docx`) for all three team members. | **A** | **YES** | Contribution document generator; documents already finalized in `reports/`. |
| **22** | `generate_optimized_report.py` | Generates `TRUST_X_Deep_Learning_Project_Report_FINAL.docx`. | **A** | **YES** | Intermediate report generator; final corrected report supersedes it. |
| **23** | `generate_project_report.py` | Original report generator script (212 KB) for initial Word report. | **A** | **YES** | Superseded report generator; final report already created. |
| **24** | `generate_review2_presentation.py` | Generates the 27-slide PowerPoint presentation `reports/TRUST_X_Review_2_Presentation.pptx`. | **A** | **YES** | Presentation generator; final presentation already generated and verified in `reports/`. |
| **25** | `get_corrected_exact_item_pages.ps1` | PowerShell Word COM script querying exact page locations of headings in corrected docx. | **B** | **YES** | Word COM inspection script only. |
| **26** | `get_corrected_page_count.ps1` | PowerShell Word COM script measuring page count of `TRUST_X_Deep_Learning_Project_Report_FINAL_CORRECTED.docx`. | **B** | **YES** | Word COM page measurement script only. |
| **27** | `get_exact_item_pages.ps1` | PowerShell Word COM script querying exact page locations of figures and tables in final docx. | **B** | **YES** | Word COM inspection script only. |
| **28** | `get_page_count.ps1` | PowerShell Word COM script querying total page count of backup docx. | **B** | **YES** | Word COM page measurement script only. |
| **29** | `hit_47_pages.py` | Spacing and padding optimizer for `generate_optimized_report.py` targeting 47 pages. | **A** | **YES** | Word document pagination tuning only. |
| **30** | `inspect_current_state.py` | Verifies existence, file sizes, timestamps, and contents of `experiments.csv`, `checkpoints/`, and `results*.json`. | **C** | **NO (KEEP)** | **ML Experiment Diagnostic:** Checks checkpoint integrity and experiment output state. |
| **31** | `inspect_images_detail.py` | Inspects XML relationship targets of images in backup Word document. | **B** | **YES** | Word document XML inspection helper only. |
| **32** | `inspect_structure.py` | Inspects paragraph breaks and section headings in backup Word document. | **B** | **YES** | Word document structural inspection helper only. |
| **33** | `inspect_zip_media.py` | Inspects dimensions and image formats of media files inside backup docx zip archive. | **B** | **YES** | Word document image inspection helper only. |
| **34** | `kill_word.ps1` | PowerShell script executing `Stop-Process -Force` on orphaned `WINWORD.EXE` processes. | **B** | **YES** | Word process cleanup utility used during docx editing. |
| **35** | `list_all_tables.py` | Inspects and lists all table dimensions and headers in backup docx. | **B** | **YES** | Word document table inspection helper only. |
| **36** | `lock_47_pages.py` | Locks spacing and pagination parameters in `generate_optimized_report.py`. | **A** | **YES** | Word document pagination tuning only. |
| **37** | `make_exactly_47_pages.py` | Spacing adjustments in `generate_optimized_report.py` targeting exactly 47 pages. | **A** | **YES** | Word document pagination tuning only. |
| **38** | `match_extracted_images.py` | Computes MD5 hashes of docx extracted images and matches them against `figures/`. | **B** | **YES** | One-off image matching helper; matches already verified. |
| **39** | `optimize_to_target_pages.py` | Paginator script adjusting line spacing and margins to reach 47 pages. | **A** | **YES** | Word document pagination tuning only. |
| **40** | `patch_and_generate.py` | Applies targeted academic and formatting corrections to `generate_final_corrected_report.py` and compiles docx. | **A** | **YES** | One-off report patching and compilation helper. |
| **41** | `prepare_hecm_hindi.py` | Full preprocessing, conflict group removal, deduplication, and stratified train/dev/test splitting for Hindi HECM corpus. | **C** | **NO (KEEP)** | **Core ML Preprocessing Pipeline:** Produces `Data/processed/hindi/` datasets. |
| **42** | `reach_exact_target.py` | Adjusts `generate_optimized_report.py` spacing to achieve 47 pages. | **A** | **YES** | Word document pagination tuning only. |
| **43** | `run_all_experiments.py` | Orchestrates all 12 factorial training, evaluation, SHAP explainability, robustness, and calibration experiments across 4 languages. | **C** | **NO (KEEP)** | **Core ML Master Runner:** Top-level execution orchestrator for the entire deep learning benchmark. |
| **44** | `run_audit.py` | Phase 1 dataset auditing runner executing `src.data.audit.run_comprehensive_audit()`. | **C** | **NO (KEEP)** | **Core ML Dataset Audit:** Required entrypoint for data verification and leakage audit. |
| **45** | `run_audit_telugu.py` | Evaluates Telugu processed dataset for cross-split overlap, intra-split duplication, and label distribution. | **C** | **NO (KEEP)** | **Core ML Dataset Audit:** Telugu dataset verification. |
| **46** | `run_preprocessing.py` | Phase 2 preprocessing runner executing `src.data.preprocessing.run_all_preprocessing()`. | **C** | **NO (KEEP)** | **Core ML Preprocessing:** Master entrypoint for raw-to-processed dataset pipeline. |
| **47** | `train.py` | CLI training entrypoint accepting `--config`, `--epochs`, `--batch_size`, and `--lr` flags to train models via `src.training.trainer`. | **C** | **NO (KEEP)** | **Core ML Training Engine:** Standard PyTorch/Transformers training CLI. |
| **48** | `tune_spacing.py` | Spacing and figure sizing adjustments in `generate_optimized_report.py`. | **A** | **YES** | Word document pagination tuning only. |
| **49** | `update_front_matter_pages.py` | Injects verified page numbers into Table of Contents and front matter lists. | **A** | **YES** | Word report TOC pagination helper only. |
| **50** | `update_output_docx.py` | Updates output document path in `generate_final_corrected_report.py`. | **A** | **YES** | One-off report generator string replacement helper. |
| **51** | `verify_each_report_pages.ps1` | PowerShell Word COM script checking page counts across all student report files. | **B** | **YES** | Word COM pagination check script only. |
| **52** | `verify_formulas.py` | Validates reported Telugu+mBERT metrics against exact formulas in `src/calibration`, `src/robustness`, `src/explainability`, and `src/twi`. | **C** | **NO (KEEP)** | **ML Mathematical Verification:** Documents and mathematically verifies metric formulas and definitions. |
| **53** | `verify_presentation.py` | Automated structural audit script inspecting slide counts, widescreen dimensions, embedded figures, and speaker notes in PPTX. | **B** | **YES** | Presentation auditing helper; audit results already preserved in `reports/TRUST_X_Review_2_Presentation_Audit.md`. |

---

## 3. CLEANUP PLAN & MANIFEST SUMMARY

- **Total Scripts Audited:** 53
- **Scripts Confirmed Safe to Delete:** **42** (22 Category A + 20 Category B)
- **Core ML Scripts Strictly Preserved:** **11** (11 Category C)
- **Uncertain / Mixed Purpose Scripts:** **0** (All scripts conclusively categorized based on source code analysis)

### Manifest of 42 Confirmed Deletion Candidates:

```
Category A: Report / Presentation Generation (22 scripts)
  1. append_remaining_chapters.py
  2. apply_final_47_pages.py
  3. build_all_contribution_reports.py
  4. compact_spillover.py
  5. condense_for_47.py
  6. extract_all_media.py
  7. finalize_47_pages.py
  8. format_toc_2tuple.py
  9. generate_final_corrected_report.py
 10. generate_individual_contribution_docs.py
 11. generate_optimized_report.py
 12. generate_project_report.py
 13. generate_review2_presentation.py
 14. hit_47_pages.py
 15. lock_47_pages.py
 16. make_exactly_47_pages.py
 17. optimize_to_target_pages.py
 18. patch_and_generate.py
 19. reach_exact_target.py
 20. tune_spacing.py
 21. update_front_matter_pages.py
 22. update_output_docx.py

Category B: Report / Presentation Auditing (20 scripts)
 23. analyze_section_lengths.py
 24. audit_all_reports.py
 25. audit_corrected_docx.py
 26. audit_drawings.py
 27. audit_existing_doc.py
 28. check_individual_page_counts.ps1
 29. find_add_figures.py
 30. find_fig3.py
 31. get_corrected_exact_item_pages.ps1
 32. get_corrected_page_count.ps1
 33. get_exact_item_pages.ps1
 34. get_page_count.ps1
 35. inspect_images_detail.py
 36. inspect_structure.py
 37. inspect_zip_media.py
 38. kill_word.ps1
 39. list_all_tables.py
 40. match_extracted_images.py
 41. verify_each_report_pages.ps1
 42. verify_presentation.py
```

### Manifest of 11 Strictly Preserved Core ML Scripts:

```
Category C: Core ML Pipeline & Research Tools (11 scripts)
  1. audit_run_all.py
  2. create_colab_notebook.py
  3. generate_final_analysis.py
  4. inspect_current_state.py
  5. prepare_hecm_hindi.py
  6. run_all_experiments.py
  7. run_audit.py
  8. run_audit_telugu.py
  9. run_preprocessing.py
 10. train.py
 11. verify_formulas.py
```

All generated reports (`reports/*.docx`), presentation (`reports/*.pptx`), figures (`figures/`), tables (`tables/`), raw and processed data (`Data/`), source code (`src/`), checkpoints (`checkpoints/`), Streamlit application (`app/`), and documentation remain 100% untouched.

