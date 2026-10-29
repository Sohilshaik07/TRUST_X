"""
TRUST-X Application Configuration
Centralized registry for file paths, model configurations, and scientific metadata.
"""

from pathlib import Path

# Project root directory (DL_PROJECT/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data and Results Paths
DATA_DIR = ROOT_DIR / "data"
TABLES_DIR = ROOT_DIR / "tables"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = ROOT_DIR / "figures" / "final"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"

# Canonical Frozen CSV Files
EXPERIMENTS_CSV = ROOT_DIR / "experiments.csv"
FINAL_RESULTS_CSV = TABLES_DIR / "final_results.csv"
MODEL_AVERAGES_CSV = TABLES_DIR / "model_averages.csv"
LANGUAGE_AVERAGES_CSV = TABLES_DIR / "language_averages.csv"
TWI_SENSITIVITY_CSV = TABLES_DIR / "twi_sensitivity.csv"
TWI_SENSITIVITY_MODEL_CSV = TABLES_DIR / "twi_sensitivity_model_averages.csv"
TWI_SENSITIVITY_LANG_CSV = TABLES_DIR / "twi_sensitivity_language_averages.csv"

# Canonical Frozen Figure Paths
FIG_MACRO_F1 = FIGURES_DIR / "macro_f1_comparison.png"
FIG_TWI = FIGURES_DIR / "twi_comparison.png"
FIG_PILLARS = FIGURES_DIR / "trust_pillars_comparison.png"
FIG_ACC_VS_F1 = FIGURES_DIR / "accuracy_vs_macro_f1.png"
FIG_F1_VS_TWI = FIGURES_DIR / "macro_f1_vs_twi.png"
FIG_SENSITIVITY = FIGURES_DIR / "twi_sensitivity.png"

# Supported Languages and Display Names
LANGUAGES = ["hindi", "telugu", "tamil", "malayalam"]
LANGUAGE_DISPLAY = {
    "hindi": "Hindi",
    "telugu": "Telugu",
    "tamil": "Tamil",
    "malayalam": "Malayalam",
}
LANGUAGE_REVERSE = {v: k for k, v in LANGUAGE_DISPLAY.items()}

# Supported Models and Display Names
MODELS = ["mbert", "indicbert", "xlmr"]
MODEL_DISPLAY = {
    "mbert": "mBERT (Multilingual BERT)",
    "indicbert": "IndicBERT",
    "xlmr": "XLM-RoBERTa (XLM-R)",
}
MODEL_SHORT_DISPLAY = {
    "mbert": "mBERT",
    "indicbert": "IndicBERT",
    "xlmr": "XLM-R",
}
MODEL_REVERSE = {
    "mBERT (Multilingual BERT)": "mbert",
    "IndicBERT": "indicbert",
    "XLM-RoBERTa (XLM-R)": "xlmr",
    "mBERT": "mbert",
    "XLM-R": "xlmr",
}

# HuggingFace Pretrained IDs for fallback
HF_FALLBACK_MODELS = {
    "mbert": "bert-base-multilingual-cased",
    "indicbert": "ai4bharat/indic-bert",
    "xlmr": "xlm-roberta-base",
}

# Expected local checkpoint directory structure
CHECKPOINT_TEMPLATE = CHECKPOINTS_DIR / "{language}" / "{model}" / "best_checkpoint"

# Pillar Definitions & Default Balanced Weights
PILLARS = {
    "Performance": {
        "metric": "Macro-F1",
        "description": "Predictive ability across both harmful and non-harmful classes under class imbalance.",
        "weight": 0.25,
    },
    "Explainability": {
        "metric": "SHAP Compactness",
        "description": "Concentration of attribution mass on the most salient tokens (higher = more compact/faithful).",
        "weight": 0.25,
    },
    "Robustness": {
        "metric": "Noise Resilience Score",
        "description": "Composite retention of F1 score and prediction stability under character and word-level noise.",
        "weight": 0.25,
    },
    "Calibration": {
        "metric": "Calibration Score",
        "description": "Composite alignment between predicted confidence and actual accuracy (1 - ECE, 1 - 2*Brier).",
        "weight": 0.25,
    },
}

# Sample code-mixed sentences for testing in Live Detection UI
SAMPLE_TEXTS = {
    "hindi": "Yeh video bilkul bakwas hai, time waste mat karo.",
    "telugu": "Idi chala chetta movie, theater ki velli time waste cheyoddu.",
    "tamil": "Idhu romba worst post, delete panunga bro.",
    "malayalam": "Ithrem mosham content njan kandittilla, report adikk.",
}

# Explicit Task Label Semantics per Language/Dataset
# Note: Hindi HECM dataset uses 0 = Hateful, 1 = Non-Hateful (inverted from standard 0=safe/1=harmful)
TASK_LABEL_MAPPINGS = {
    "telugu": {
        "task_name": "Telugu-English Hate Speech Detection",
        0: {"name": "Non-Hate", "is_harmful": False, "badge_bg": "#d1e7dd", "badge_color": "#0f5132"},
        1: {"name": "Hate", "is_harmful": True, "badge_bg": "#f8d7da", "badge_color": "#842029"},
    },
    "hindi": {
        "task_name": "Hindi-English HECM Hate Speech Detection",
        0: {"name": "Hateful", "is_harmful": True, "badge_bg": "#f8d7da", "badge_color": "#842029"},
        1: {"name": "Non-Hateful", "is_harmful": False, "badge_bg": "#d1e7dd", "badge_color": "#0f5132"},
    },
    "tamil": {
        "task_name": "Tamil-English Offensive Language Detection",
        0: {"name": "Not Offensive", "is_harmful": False, "badge_bg": "#d1e7dd", "badge_color": "#0f5132"},
        1: {"name": "Offensive", "is_harmful": True, "badge_bg": "#f8d7da", "badge_color": "#842029"},
    },
    "malayalam": {
        "task_name": "Malayalam-English Offensive Language Detection",
        0: {"name": "Not Offensive", "is_harmful": False, "badge_bg": "#d1e7dd", "badge_color": "#0f5132"},
        1: {"name": "Offensive", "is_harmful": True, "badge_bg": "#f8d7da", "badge_color": "#842029"},
    },
}

def get_label_semantics(language: str, class_idx: int) -> dict:
    """Retrieve label semantics dictionary for a given language and class index."""
    lang_map = TASK_LABEL_MAPPINGS.get(language.lower(), TASK_LABEL_MAPPINGS["hindi"])
    return lang_map.get(class_idx, {"name": f"Class {class_idx}", "is_harmful": False, "badge_bg": "#e2e3e5", "badge_color": "#41464b"})
