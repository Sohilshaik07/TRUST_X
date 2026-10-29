#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_hecm_hindi.py
=====================
TRUST-X Project — Hindi (HECM) Dataset Preprocessing Script
Author confirmation: 0 = Hateful, 1 = Non-hateful (Prashant K.)

Pipeline:
  1. Read raw HECM.txt and HECM_Class.txt (preserves original files untouched)
  2. Map numeric label: 0 -> Hateful, 1 -> Non-hateful
  3. Identify and exclude all 25 conflicting-label text groups (50 total rows)
  4. Deduplicate non-conflicting exact duplicates (removes 93 excess rows)
  5. Minimal conservative cleaning (whitespace normalization only, preserving all emojis, casing, URLs, mentions, hashtags)
  6. Stratified train/dev/test split (70% / 15% / 15%) with random_state=42
  7. Export Data/processed/hindi/{train,dev,test}.csv in standard TRUST-X schema
  8. Perform exact and normalized leakage audit
  9. Generate supporting tables:
     - tables/hecm_hindi_split_distribution.csv
     - tables/hecm_hindi_duplicate_audit.csv
     - tables/hecm_hindi_preprocessing_stats.csv
"""

import os
import sys
import io
import re
import unicodedata
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "Data", "raw", "hindi")
PROCESSED_DIR = os.path.join(BASE_DIR, "Data", "processed", "hindi")
TABLES_DIR = os.path.join(BASE_DIR, "tables")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def clean_text(text):
    """
    Conservative whitespace normalization only.
    Preserves casing, emojis, punctuation, URLs, mentions, hashtags.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def run_hecm_preprocessing():
    print("=" * 80)
    print("TRUST-X HECM HINDI PREPROCESSING")
    print("=" * 80)

    raw_text_path = os.path.join(RAW_DIR, "HECM.txt")
    raw_class_path = os.path.join(RAW_DIR, "HECM_Class.txt")

    # 1. Read raw files without modifying originals
    with open(raw_text_path, "r", encoding="utf-8-sig") as f:
        raw_texts = [line.rstrip("\r\n") for line in f]

    with open(raw_class_path, "r", encoding="utf-8-sig") as f:
        raw_labels = [line.rstrip("\r\n") for line in f]

    assert len(raw_texts) == len(raw_labels), "Raw texts and labels line count mismatch!"
    total_raw = len(raw_texts)
    print(f"[*] Loaded raw HECM dataset: {total_raw} aligned pairs.")

    # 2. Build initial dataframe
    cleaned_texts = [clean_text(t) for t in raw_texts]
    labels = [l.strip() for l in raw_labels]

    df = pd.DataFrame({
        "raw_index": range(total_raw),
        "text": cleaned_texts,
        "raw_label": labels
    })

    # 3. Duplicate and conflict analysis
    text_counts = df["text"].value_counts()
    dup_text_groups = text_counts[text_counts > 1]
    total_dup_rows = dup_text_groups.sum()
    excess_dup_rows = total_dup_rows - len(dup_text_groups)

    label_diversity = df.groupby("text")["raw_label"].nunique()
    conflicting_texts = set(label_diversity[label_diversity > 1].index.tolist())
    conflict_rows = df[df["text"].isin(conflicting_texts)]
    num_conflicting_groups = len(conflicting_texts)
    num_conflicting_rows = len(conflict_rows)

    print(f"[*] Total duplicate text groups: {len(dup_text_groups)} ({total_dup_rows} rows, {excess_dup_rows} excess)")
    print(f"[*] Conflicting-label text groups: {num_conflicting_groups} ({num_conflicting_rows} rows)")

    # Exclude conflicting-label text groups completely
    df_non_conflict = df[~df["text"].isin(conflicting_texts)].copy()
    rows_after_conflicts = len(df_non_conflict)

    # Deduplicate non-conflicting duplicates (keep first)
    non_conflict_counts = df_non_conflict["text"].value_counts()
    non_conflict_dup_groups = non_conflict_counts[non_conflict_counts > 1]
    excess_non_conflict_rows = non_conflict_dup_groups.sum() - len(non_conflict_dup_groups)

    df_clean = df_non_conflict.drop_duplicates(subset=["text"], keep="first").copy().reset_index(drop=True)
    final_modeling_samples = len(df_clean)
    print(f"[*] Rows after excluding conflicts: {rows_after_conflicts}")
    print(f"[*] Non-conflicting duplicate excess rows removed: {excess_non_conflict_rows}")
    print(f"[*] Final unique modeling dataset: {final_modeling_samples} samples")

    # 4. Map labels
    # Author confirmation: 0 = Hateful, 1 = Non-hateful
    df_clean["label"] = df_clean["raw_label"].astype(int)
    label_map_human = {0: "Hateful", 1: "Non-hateful"}
    df_clean["original_label"] = df_clean["label"].map(label_map_human)

    # 5. Stratified train/dev/test split (70% / 15% / 15%)
    train_df, temp_df = train_test_split(
        df_clean, test_size=0.30, stratify=df_clean["label"], random_state=42
    )
    dev_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
    )

    splits = {
        "train": train_df.copy().reset_index(drop=True),
        "dev": dev_df.copy().reset_index(drop=True),
        "test": test_df.copy().reset_index(drop=True)
    }

    # 6. Save processed CSVs with TRUST-X schema
    # Schema: ['id', 'text', 'label', 'original_label', 'language', 'task', 'split']
    for split_name, s_df in splits.items():
        s_df["id"] = [f"hi_{split_name}_{i:05d}" for i in range(len(s_df))]
        s_df["language"] = "hindi"
        s_df["task"] = "hate_speech"
        s_df["split"] = split_name
        
        cols = ["id", "text", "label", "original_label", "language", "task", "split"]
        out_csv = os.path.join(PROCESSED_DIR, f"{split_name}.csv")
        s_df[cols].to_csv(out_csv, index=False, encoding="utf-8")
        print(f"[+] Saved {split_name} split to {out_csv} ({len(s_df)} rows)")

    # 7. Leakage check
    s_train = set(splits["train"]["text"])
    s_dev = set(splits["dev"]["text"])
    s_test = set(splits["test"]["text"])

    exact_td = len(s_train.intersection(s_dev))
    exact_tt = len(s_train.intersection(s_test))
    exact_dt = len(s_dev.intersection(s_test))

    def norm_text(t):
        return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', t)).strip().lower()

    train_norm = set(splits["train"]["text"].apply(norm_text))
    dev_norm = set(splits["dev"]["text"].apply(norm_text))
    test_norm = set(splits["test"]["text"].apply(norm_text))

    norm_td = len(train_norm.intersection(dev_norm))
    norm_tt = len(train_norm.intersection(test_norm))
    norm_dt = len(dev_norm.intersection(test_norm))

    print(f"[*] Exact text overlap: train-dev={exact_td}, train-test={exact_tt}, dev-test={exact_dt}")
    print(f"[*] Normalized text overlap: train-dev={norm_td}, train-test={norm_tt}, dev-test={norm_dt}")

    # 8. Tables generation
    # Table 1: Split distribution
    split_dist_rows = []
    for split_name in ["full_clean", "train", "dev", "test"]:
        target_df = df_clean if split_name == "full_clean" else splits[split_name]
        total_s = len(target_df)
        vc = target_df["label"].value_counts().to_dict()
        n_hateful = vc.get(0, 0)
        n_non_hateful = vc.get(1, 0)
        split_dist_rows.append({
            "split": split_name,
            "total_samples": total_s,
            "hateful_label_0": n_hateful,
            "hateful_pct": round(n_hateful / total_s * 100, 2),
            "non_hateful_label_1": n_non_hateful,
            "non_hateful_pct": round(n_non_hateful / total_s * 100, 2),
            "imbalance_ratio_0_over_1": round(n_hateful / n_non_hateful, 4)
        })
    split_dist_df = pd.DataFrame(split_dist_rows)
    split_dist_path = os.path.join(TABLES_DIR, "hecm_hindi_split_distribution.csv")
    split_dist_df.to_csv(split_dist_path, index=False)
    print(f"[+] Saved split distribution table to {split_dist_path}")

    # Table 2: Duplicate audit
    dup_audit_rows = [
        {"metric": "total_raw_rows", "count": total_raw, "description": "Original line count in HECM.txt and HECM_Class.txt"},
        {"metric": "unique_raw_texts", "count": len(text_counts), "description": "Unique text values before deduplication"},
        {"metric": "total_duplicate_text_groups", "count": len(dup_text_groups), "description": "Text values occurring > 1 time in raw data"},
        {"metric": "total_duplicate_rows", "count": int(total_dup_rows), "description": "Total rows belonging to duplicate groups"},
        {"metric": "duplicate_excess_rows", "count": int(excess_dup_rows), "description": "Excess rows from all duplicate texts"},
        {"metric": "conflicting_text_groups", "count": num_conflicting_groups, "description": "Texts associated with both label 0 and label 1"},
        {"metric": "conflicting_rows_excluded", "count": num_conflicting_rows, "description": "Total rows excluded due to label conflicts"},
        {"metric": "rows_after_conflict_exclusion", "count": rows_after_conflicts, "description": "Dataset size after excluding conflicting groups"},
        {"metric": "non_conflicting_dup_groups", "count": len(non_conflict_dup_groups), "description": "Duplicate text groups with consistent labels"},
        {"metric": "non_conflicting_excess_rows_removed", "count": int(excess_non_conflict_rows), "description": "Redundant identical rows removed"},
        {"metric": "final_unique_modeling_samples", "count": final_modeling_samples, "description": "Final unambiguous modeling samples"}
    ]
    dup_audit_df = pd.DataFrame(dup_audit_rows)
    dup_audit_path = os.path.join(TABLES_DIR, "hecm_hindi_duplicate_audit.csv")
    dup_audit_df.to_csv(dup_audit_path, index=False)
    print(f"[+] Saved duplicate audit table to {dup_audit_path}")

    # Table 3: Preprocessing text statistics
    def compute_stats(name, texts):
        chars = [len(t) for t in texts]
        words = [len(t.split()) for t in texts]
        return {
            "dataset_version": name,
            "samples": len(texts),
            "unique_texts": len(set(texts)),
            "min_chars": int(np.min(chars)),
            "max_chars": int(np.max(chars)),
            "mean_chars": round(float(np.mean(chars)), 2),
            "median_chars": round(float(np.median(chars)), 2),
            "min_words": int(np.min(words)),
            "max_words": int(np.max(words)),
            "mean_words": round(float(np.mean(words)), 2),
            "median_words": round(float(np.median(words)), 2),
            "p95_words": round(float(np.percentile(words, 95)), 2)
        }

    stats_rows = [
        compute_stats("raw_hecm", cleaned_texts),
        compute_stats("cleaned_hecm_full", df_clean["text"]),
        compute_stats("hecm_train", splits["train"]["text"]),
        compute_stats("hecm_dev", splits["dev"]["text"]),
        compute_stats("hecm_test", splits["test"]["text"])
    ]
    stats_df = pd.DataFrame(stats_rows)
    stats_path = os.path.join(TABLES_DIR, "hecm_hindi_preprocessing_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"[+] Saved preprocessing stats table to {stats_path}")

    print("\n" + "=" * 80)
    print("HECM HINDI PREPROCESSING SUMMARY")
    print("=" * 80)
    print(split_dist_df.to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    run_hecm_preprocessing()
