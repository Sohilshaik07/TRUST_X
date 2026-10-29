# TRUST-X: An Integrated Trustworthiness Evaluation Framework for Transformer-Based Multilingual Code-Mixed Hate and Offensive Speech Detection

**Academic Deep Learning Project** | 3-Person Undergraduate Research Team

---

## 1. Research Question & Objective

Standard multilingual harmful-language classification research focuses predominantly on maximizing predictive metrics like Macro-F1. However, real-world content moderation demands models that are:
1. **Accurate & Generalizable** across Romanized code-mixing and native scripts.
2. **Confidence-Calibrated** (predicted probabilities match true likelihood of harmfulness).
3. **Linguistically Robust** against casual typings, intentional leet-speak obfuscations, and emoji/symbol insertions.
4. **Interpretable** at the token attribution level (SHAP feature attribution).

**Core Question**: *"Among pretrained transformer architectures fine-tuned for multilingual code-mixed harmful speech, which model offers the most trustworthy and deployable profile—not merely the highest F1 score?"*

---

## 2. Active Languages & Dataset Specifications

> [!IMPORTANT]
> **HINDI STATUS: POSTPONED**. As per project priority guidelines, the Hindi dataset has unresolved provenance questions and is intentionally excluded from the current phase. `Data/raw/hindi/` remains completely untouched.

| Language Setting | Dataset | Task Definition | Classes | Processed Train / Dev / Test Samples | Code-Mixing (Latin %) | Native Script % |
|---|---|---|---|---|---|---|
| **Telugu-English** | HOLD (EACL 2024) | Hate Speech Detection | Binary (`hate` vs `non-hate`) | 2,758 / 591 / 592 | 48.52% | 35.69% |
| **Tamil-English** | DravidianCodeMix-2020 | Offensive Language Detection | Mapped to Binary (Non-offensive vs Offensive) | 33,431 / 4,216 / 4,232 | 65.95% | 15.14% |
| **Malayalam-English** | DravidianCodeMix-2020 | Offensive Language Detection | Mapped to Binary (Non-offensive vs Offensive) | 10,802 / 1,836 / 1,844 | 56.53% | 25.99% |

### Key Preprocessing Integrity Rules:
- **Zero modification to `Data/raw/`**: All raw files remain strictly untouched in read-only mode.
- **Conservative text cleaning**: Emojis, punctuation, casing, and code-mixing are completely preserved.
- **Official Split Integrity**: Official train/dev/test splits are preserved for Tamil and Malayalam.
- **Telugu Ground-Truth Resolution**: The raw Telugu test file contains no labels (shared-task holdout format) and overlaps with training samples. To guarantee genuine evaluation without leakage, the 4,000 labelled samples were deduplicated and partitioned into a stratified 70/15/15 split.
- **Class-Weighted Loss**: Malayalam exhibits severe class imbalance (only 3.98% offensive samples in train); balanced class weighting is enforced in loss calculation.

---

## 3. Evaluated Transformer Models

All experiments evaluate well-established, standard Hugging Face model checkpoints:
1. **mBERT**: `bert-base-multilingual-cased` (104 languages, Wikipedia pretraining)
2. **IndicBERT**: `ai4bharat/indic-bert` (ALBERT-style architecture pretrained on 12 Indian languages + English)
3. **XLM-RoBERTa**: `xlm-roberta-base` (2.5TB CommonCrawl pretraining across 100 languages)

**Total Experimental Matrix**: 3 Languages × 3 Models = **9 Main GPU Experiments**.

---

## 4. Evaluation Pillars & Trustworthiness Index (TWI)

### Pillar 1: Predictive Performance
- Test Accuracy, Macro-Precision, Macro-Recall, Macro-F1, and per-class F1.
- Confusion matrices with normalized error proportions.

### Pillar 2: Confidence Calibration
- **Expected Calibration Error (ECE)** using 10 equal-width bins.
- **Brier Score**.
- **Reliability Diagrams** visualizing the gap between predicted confidence and observed empirical accuracy.
- **Temperature Scaling** (optimized strictly on validation logits; evaluated on test).

### Pillar 3: Linguistic Robustness
Evaluates model behavior under 6 realistic adversarial/typographical perturbations:
1. Leet-speak character substitutions (`a -> @`, `e -> 3`, `i -> 1`, `s -> $`)
2. Character deletion (casual mobile typing drops)
3. Character repetition (elongated emphasis, e.g. `idiot` -> `idiiiot`)
4. Punctuation insertion (`h.a.t.e`)
5. Whitespace manipulation (merging/splitting)
6. Composite perturbation
- Metrics: Perturbed F1, Performance Drop, F1 Retention Rate, and Prediction Flip Rate.

### Pillar 4: SHAP Explainability
- Evaluates token-level feature attribution using `shap.Explainer` on representative test subsets (True Positives, True Negatives, False Positives, False Negatives).
- Measures explanation compactness (energy concentration in top tokens).

### Integrated Metric: Trustworthiness Index (TWI)
$$\text{TWI} = w_1 \cdot \text{Perf} + w_2 \cdot \text{Expl} + w_3 \cdot \text{Rob} + w_4 \cdot \text{Calib}$$
where $w_1 + w_2 + w_3 + w_4 = 1.0$.
Includes sensitivity analysis across Balanced, Performance-Heavy, Robustness-Heavy, and Calibration-Heavy profiles.

---

## 5. Execution Environment & Google Colab Workflow

> [!NOTE]
> The local Antigravity environment is CPU-only and is utilized for codebase structuring, data audit, and pipeline verification. **Full model training is executed on Google Colab with GPU acceleration (T4/V100/A100).**

### Running in Google Colab:
1. Open [`notebooks/TRUST_X_Colab_Runner.ipynb`](notebooks/TRUST_X_Colab_Runner.ipynb) in Google Colab.
2. Select **Runtime -> Change runtime type -> GPU (T4 or higher)**.
3. Mount Google Drive to persist outputs:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```
4. Run all experiments:
   ```bash
   # Run pilot experiment (Telugu + mBERT)
   !python scripts/train.py --config configs/telugu_mbert.yaml

   # Run all 9 experiments
   !python scripts/run_all_experiments.py
   ```

### Running Locally (Pipeline / Debugging):
```bash
# Run data audit
python scripts/run_audit.py

# Run conservative preprocessing
python scripts/run_preprocessing.py

# Run a single experiment
python scripts/train.py --config configs/telugu_mbert.yaml
```

---

## 6. Repository Directory Layout

```
DL_PROJECT/
├── Data/
│   ├── raw/                           # Untouched raw datasets (Read-Only)
│   │   ├── hindi/                     # Empty (POSTPONED)
│   │   ├── telugu/                    # Raw Telugu Excel files
│   │   ├── tamil/                     # DravidianCodeMix-2020 TSVs
│   │   └── malayalam/                 # DravidianCodeMix-2020 TSVs
│   ├── processed/                     # Cleaned, standardized CSV splits
│   │   ├── telugu/                    # train.csv, dev.csv, test.csv
│   │   ├── tamil/                     # train.csv, dev.csv, test.csv
│   │   └── malayalam/                 # train.csv, dev.csv, test.csv
│   └── results/
├── src/
│   ├── data/
│   │   ├── audit.py                   # Programmatic data audit
│   │   ├── preprocessing.py           # Conservative cleaning & mapping
│   │   └── loaders.py                 # PyTorch Dataset & DataLoaders
│   ├── models/
│   │   └── model_factory.py           # Model loading & GPU/CPU detection
│   ├── training/
│   │   └── trainer.py                 # Unified training, eval, AMP loop
│   ├── evaluation/
│   │   └── metrics.py                 # Accuracy, F1, confusion matrices
│   ├── calibration/
│   │   └── calibration.py             # ECE, Brier, reliability diagrams
│   ├── robustness/
│   │   ├── perturbations.py           # 6 perturbation transformations
│   │   └── evaluate.py                # Robustness testing runner
│   ├── explainability/
│   │   └── shap_analysis.py           # SHAP token attribution
│   └── twi/
│       └── twi.py                     # TWI formula & sensitivity analysis
├── configs/                           # 9 YAML configs for all experiments
├── notebooks/
│   └── TRUST_X_Colab_Runner.ipynb    # Google Colab GPU Master Runner
├── scripts/
│   ├── run_audit.py                   # Audit execution script
│   ├── run_preprocessing.py           # Preprocessing execution script
│   ├── train.py                       # CLI training script
│   └── run_all_experiments.py         # 9-experiment orchestrator
├── checkpoints/                       # Model checkpoints & predictions
├── figures/                           # Audit & evaluation figures
├── tables/                            # Audit & metrics tables
├── requirements.txt                   # Environment dependencies
└── README.md
```

---

## 7. Absolute Academic Honesty Policy
- **Zero Fabrication**: No experimental metrics, training curves, confusion matrices, SHAP explanations, or TWI rankings are fabricated or estimated.
- Unexecuted runs are explicitly designated as **NOT YET RUN**.
- The final report will incorporate exclusively actual metrics generated from genuine Google Colab GPU training runs.
