"""
TRUST-X Conservative Data Preprocessing Module
Implements conservative, non-destructive cleaning for multilingual code-mixed text:
- Preserves Unicode, emojis, punctuation, code-mixing, and casing.
- Standardizes whitespaces, removes null/corrupted strings.
- Implements explicit, documented label mappings.
- Preserves official train/dev/test splits for Tamil and Malayalam.
- Performs stratified splitting for Telugu (since raw test set is unlabelled).
- Outputs standardized schema: ['id', 'text', 'label', 'original_label', 'language', 'task', 'split']
"""

import os
import sys
import io
import json
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "Data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "Data", "processed")
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")
TABLES_DIR = os.path.join(BASE_DIR, "tables")

for d in [PROCESSED_DIR, CONFIGS_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

# Documented Label Mappings
LABEL_MAPPINGS = {
    "telugu": {
        "task": "Hate Speech Detection",
        "description": "Binary classification: 0 = non-hate, 1 = hate",
        "mapping": {
            "non-hate": 0,
            "hate": 1
        },
        "excluded": []
    },
    "tamil": {
        "task": "Offensive Language Detection",
        "description": "Binary mapping: 0 = Not_offensive, 1 = Offensive (all subtypes). 'not-Tamil' is excluded.",
        "mapping": {
            "Not_offensive": 0,
            "Offensive_Untargetede": 1,
            "Offensive_Targeted_Insult_Group": 1,
            "Offensive_Targeted_Insult_Individual": 1,
            "Offensive_Targeted_Insult_Other": 1
        },
        "excluded": ["not-Tamil"]
    },
    "malayalam": {
        "task": "Offensive Language Detection",
        "description": "Binary mapping: 0 = Not_offensive, 1 = Offensive (all subtypes). 'not-malayalam' is excluded.",
        "mapping": {
            "Not_offensive": 0,
            "Offensive_Targeted_Insult_Individual": 1,
            "Offensive_Untargetede": 1,
            "Offensive_Targeted_Insult_Group": 1
        },
        "excluded": ["not-malayalam"]
    }
}

def clean_text(text):
    """
    Conservative text cleaner:
    - Preserves emojis, scripts, casing, code-mixing, punctuation.
    - Normalizes multiple spaces/newlines/tabs into a single clean space.
    - Strips leading/trailing whitespaces.
    """
    if not isinstance(text, str):
        return ""
    # Normalize multiple whitespace characters
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_telugu(seed=42):
    """
    Preprocesses Telugu-English HOLD dataset.
    Since raw test set is unlabelled (competition holdout),
    the 4000 labelled samples are cleaned, deduplicated, and split:
    70% Train, 15% Validation (Dev), 15% Test with stratification.
    """
    print("\n" + "="*50)
    print("PREPROCESSING: TELUGU-ENGLISH (Hate Speech)")
    print("="*50)
    raw_path = os.path.join(RAW_DIR, "telugu", "training_data_telugu-hate.xlsx")
    df = pd.read_excel(raw_path)
    
    # 1. Cleaning & validation
    df['text'] = df['Comments'].apply(clean_text)
    df = df[df['text'].str.len() > 0].copy()
    
    # 2. Label mapping
    mapping = LABEL_MAPPINGS["telugu"]["mapping"]
    df['original_label'] = df['Label'].astype(str).str.strip().str.lower()
    df = df[df['original_label'].isin(mapping.keys())].copy()
    df['label'] = df['original_label'].map(mapping)
    
    # 3. Deduplication before splitting (Rule 7: Prevent duplicate leakage)
    initial_len = len(df)
    df = df.drop_duplicates(subset=['text']).copy().reset_index(drop=True)
    dedup_removed = initial_len - len(df)
    print(f"[*] Telugu: Initial valid rows={initial_len}, Deduplicated={len(df)} (Removed {dedup_removed} duplicate texts)")
    
    # 4. Stratified splitting: 70% Train, 15% Dev, 15% Test
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df['label'], random_state=seed
    )
    dev_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['label'], random_state=seed
    )
    
    splits = {
        "train": train_df.copy().reset_index(drop=True),
        "dev": dev_df.copy().reset_index(drop=True),
        "test": test_df.copy().reset_index(drop=True)
    }
    
    out_dir = os.path.join(PROCESSED_DIR, "telugu")
    os.makedirs(out_dir, exist_ok=True)
    
    summary = []
    for split_name, s_df in splits.items():
        s_df['id'] = [f"te_{split_name}_{i:05d}" for i in range(len(s_df))]
        s_df['language'] = "telugu"
        s_df['task'] = "hate_speech"
        s_df['split'] = split_name
        
        cols = ['id', 'text', 'label', 'original_label', 'language', 'task', 'split']
        out_file = os.path.join(out_dir, f"{split_name}.csv")
        s_df[cols].to_csv(out_file, index=False)
        
        vc = s_df['label'].value_counts().to_dict()
        pct_harmful = (vc.get(1, 0) / len(s_df)) * 100
        print(f"    -> Split [{split_name.upper()}]: {len(s_df)} rows | Label 0: {vc.get(0, 0)}, Label 1: {vc.get(1, 0)} ({pct_harmful:.2f}% hate)")
        summary.append({
            "language": "telugu",
            "split": split_name,
            "total_samples": len(s_df),
            "non_harmful_0": vc.get(0, 0),
            "harmful_1": vc.get(1, 0),
            "harmful_pct": round(pct_harmful, 2)
        })
        
    return summary

def preprocess_tamil():
    """
    Preprocesses Tamil-English DravidianCodeMix-2020 dataset.
    Preserves official train, dev, and test splits.
    Maps multi-class offensive labels to binary: 0 = Not_offensive, 1 = Offensive.
    Excludes non-Tamil comments ('not-Tamil').
    """
    print("\n" + "="*50)
    print("PREPROCESSING: TAMIL-ENGLISH (Offensive Language)")
    print("="*50)
    
    raw_files = {
        "train": os.path.join(RAW_DIR, "tamil", "tamil_offensive_full_train.csv"),
        "dev": os.path.join(RAW_DIR, "tamil", "tamil_offensive_full_dev.csv"),
        "test": os.path.join(RAW_DIR, "tamil", "tamil_offensive_full_test.csv")
    }
    
    mapping = LABEL_MAPPINGS["tamil"]["mapping"]
    excluded = LABEL_MAPPINGS["tamil"]["excluded"]
    out_dir = os.path.join(PROCESSED_DIR, "tamil")
    os.makedirs(out_dir, exist_ok=True)
    
    summary = []
    
    for split_name, fpath in raw_files.items():
        # DravidianCodeMix has no header; col 0: text, col 1: label
        df = pd.read_csv(fpath, sep='\t', header=None, usecols=[0, 1])
        df.columns = ['raw_text', 'raw_label']
        
        df['text'] = df['raw_text'].apply(clean_text)
        df = df[df['text'].str.len() > 0].copy()
        
        df['original_label'] = df['raw_label'].astype(str).str.strip()
        # Exclude non-Tamil samples
        num_excluded = df['original_label'].isin(excluded).sum()
        df = df[~df['original_label'].isin(excluded)].copy()
        
        # Keep only known mapped labels
        df = df[df['original_label'].isin(mapping.keys())].copy()
        df['label'] = df['original_label'].map(mapping)
        
        # Deduplicate within train split only (preserve official test/dev as benchmark)
        if split_name == "train":
            initial_len = len(df)
            df = df.drop_duplicates(subset=['text']).copy().reset_index(drop=True)
            print(f"[*] Tamil Train: Deduplicated from {initial_len} to {len(df)} rows (Removed {initial_len - len(df)} duplicates)")
        else:
            df = df.reset_index(drop=True)
            
        df['id'] = [f"ta_{split_name}_{i:05d}" for i in range(len(df))]
        df['language'] = "tamil"
        df['task'] = "offensive_language"
        df['split'] = split_name
        
        cols = ['id', 'text', 'label', 'original_label', 'language', 'task', 'split']
        out_file = os.path.join(out_dir, f"{split_name}.csv")
        df[cols].to_csv(out_file, index=False)
        
        vc = df['label'].value_counts().to_dict()
        pct_harmful = (vc.get(1, 0) / len(df)) * 100
        print(f"    -> Split [{split_name.upper()}]: {len(df)} rows (Excluded {num_excluded} non-Tamil) | Label 0: {vc.get(0,0)}, Label 1: {vc.get(1,0)} ({pct_harmful:.2f}% offensive)")
        summary.append({
            "language": "tamil",
            "split": split_name,
            "total_samples": len(df),
            "non_harmful_0": vc.get(0, 0),
            "harmful_1": vc.get(1, 0),
            "harmful_pct": round(pct_harmful, 2)
        })
        
    return summary

def preprocess_malayalam():
    """
    Preprocesses Malayalam-English DravidianCodeMix-2020 dataset.
    Preserves official train, dev, and test splits.
    Maps multi-class offensive labels to binary: 0 = Not_offensive, 1 = Offensive.
    Excludes non-Malayalam comments ('not-malayalam').
    """
    print("\n" + "="*50)
    print("PREPROCESSING: MALAYALAM-ENGLISH (Offensive Language)")
    print("="*50)
    
    raw_files = {
        "train": os.path.join(RAW_DIR, "malayalam", "mal_full_offensive_train.csv"),
        "dev": os.path.join(RAW_DIR, "malayalam", "mal_full_offensive_dev.csv"),
        "test": os.path.join(RAW_DIR, "malayalam", "mal_full_offensive_test.csv")
    }
    
    mapping = LABEL_MAPPINGS["malayalam"]["mapping"]
    excluded = LABEL_MAPPINGS["malayalam"]["excluded"]
    out_dir = os.path.join(PROCESSED_DIR, "malayalam")
    os.makedirs(out_dir, exist_ok=True)
    
    summary = []
    
    for split_name, fpath in raw_files.items():
        df = pd.read_csv(fpath, sep='\t', header=None, usecols=[0, 1])
        df.columns = ['raw_text', 'raw_label']
        
        df['text'] = df['raw_text'].apply(clean_text)
        df = df[df['text'].str.len() > 0].copy()
        
        df['original_label'] = df['raw_label'].astype(str).str.strip()
        num_excluded = df['original_label'].isin(excluded).sum()
        df = df[~df['original_label'].isin(excluded)].copy()
        
        df = df[df['original_label'].isin(mapping.keys())].copy()
        df['label'] = df['original_label'].map(mapping)
        
        # Deduplicate within train split only
        if split_name == "train":
            initial_len = len(df)
            df = df.drop_duplicates(subset=['text']).copy().reset_index(drop=True)
            print(f"[*] Malayalam Train: Deduplicated from {initial_len} to {len(df)} rows (Removed {initial_len - len(df)} duplicate texts)")
        else:
            df = df.reset_index(drop=True)
            
        df['id'] = [f"ml_{split_name}_{i:05d}" for i in range(len(df))]
        df['language'] = "malayalam"
        df['task'] = "offensive_language"
        df['split'] = split_name
        
        cols = ['id', 'text', 'label', 'original_label', 'language', 'task', 'split']
        out_file = os.path.join(out_dir, f"{split_name}.csv")
        df[cols].to_csv(out_file, index=False)
        
        vc = df['label'].value_counts().to_dict()
        pct_harmful = (vc.get(1, 0) / len(df)) * 100
        print(f"    -> Split [{split_name.upper()}]: {len(df)} rows (Excluded {num_excluded} non-Malayalam) | Label 0: {vc.get(0,0)}, Label 1: {vc.get(1,0)} ({pct_harmful:.2f}% offensive)")
        summary.append({
            "language": "malayalam",
            "split": split_name,
            "total_samples": len(df),
            "non_harmful_0": vc.get(0, 0),
            "harmful_1": vc.get(1, 0),
            "harmful_pct": round(pct_harmful, 2)
        })
        
    return summary

def run_all_preprocessing():
    print("="*80)
    print("STARTING TRUST-X DATA PREPROCESSING (PHASE 2)")
    print("="*80)
    
    # Save label mapping config
    mapping_path = os.path.join(CONFIGS_DIR, "label_mappings.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(LABEL_MAPPINGS, f, indent=4)
    print(f"[+] Saved label mappings configuration to: {mapping_path}")
    
    telugu_summary = preprocess_telugu(seed=42)
    tamil_summary = preprocess_tamil()
    malayalam_summary = preprocess_malayalam()
    
    all_summaries = telugu_summary + tamil_summary + malayalam_summary
    summary_df = pd.DataFrame(all_summaries)
    
    summary_csv = os.path.join(TABLES_DIR, "processed_datasets_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[+] Saved processed datasets summary to: {summary_csv}")
    
    print("\n" + "="*80)
    print("PROCESSED DATASETS OVERVIEW:")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    run_all_preprocessing()
