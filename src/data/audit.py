"""
TRUST-X Dataset Audit Module
Programmatically inspects raw datasets for Telugu, Tamil, and Malayalam.
Computes comprehensive statistics, class distributions, text lengths,
script characteristics, missing values, and duplicates without modifying raw files.
"""

import os
import sys
import io
import re
import unicodedata
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Data", "raw")
TABLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tables")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "figures")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports")

for d in [TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

def detect_script_fractions(text):
    if not isinstance(text, str) or len(text) == 0:
        return {"telugu": 0.0, "tamil": 0.0, "malayalam": 0.0, "latin": 0.0, "other": 0.0}
    
    telugu_count = 0
    tamil_count = 0
    malayalam_count = 0
    latin_count = 0
    other_count = 0
    
    for ch in text:
        cp = ord(ch)
        if 0x0C00 <= cp <= 0x0C7F: # Telugu Unicode block
            telugu_count += 1
        elif 0x0B80 <= cp <= 0x0BFF: # Tamil Unicode block
            tamil_count += 1
        elif 0x0D00 <= cp <= 0x0D7F: # Malayalam Unicode block
            malayalam_count += 1
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A): # Basic Latin (A-Z, a-z)
            latin_count += 1
        else:
            other_count += 1
            
    total = len(text)
    return {
        "telugu": telugu_count / total,
        "tamil": tamil_count / total,
        "malayalam": malayalam_count / total,
        "latin": latin_count / total,
        "other": other_count / total
    }

def audit_file(file_path, language, split_name, sep=None, has_header=True):
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.xlsx', '.xls']:
        if has_header:
            df = pd.read_excel(file_path)
        else:
            df = pd.read_excel(file_path, header=None)
    else:
        if has_header:
            df = pd.read_csv(file_path, sep=sep if sep else '\t')
        else:
            df = pd.read_csv(file_path, sep=sep if sep else '\t', header=None)
            
    num_rows, num_cols = df.shape
    col_names = [str(c) for c in df.columns]
    
    # Identify text and label column
    text_col = None
    label_col = None
    
    if language == "telugu":
        if "Comments" in df.columns:
            text_col = "Comments"
        elif 1 in df.columns or "1" in col_names:
            text_col = 1 if 1 in df.columns else "1"
            
        if "Label" in df.columns:
            label_col = "Label"
        else:
            label_col = None
    elif language in ["tamil", "malayalam"]:
        if split_name == "full":
            # In full.csv files, Col 0 is Label, Col 1 is Text
            label_col = df.columns[0]
            text_col = df.columns[1]
        else:
            # In train, dev, test: Col 0 is Text, Col 1 is Label, Col 2 is empty/trailing tab
            text_col = df.columns[0]
            label_col = df.columns[1] if num_cols > 1 else None
            
    # Compute missing values
    missing_total = df.isna().sum().to_dict()
    missing_text = df[text_col].isna().sum() if text_col is not None else 0
    missing_label = df[label_col].isna().sum() if label_col is not None else num_rows
    
    # Text series
    if text_col is not None:
        text_series = df[text_col].dropna().astype(str)
        char_lengths = text_series.str.len()
        token_lengths = text_series.apply(lambda s: len(s.split()))
        
        dup_rows = int(df.duplicated().sum())
        dup_texts = int(df[text_col].duplicated().sum())
        
        # Script fractions
        script_stats = [detect_script_fractions(t) for t in text_series]
        script_df = pd.DataFrame(script_stats)
        mean_scripts = script_df.mean().to_dict() if not script_df.empty else {}
        
        len_stats = {
            "char_mean": float(char_lengths.mean()) if not char_lengths.empty else 0.0,
            "char_median": float(char_lengths.median()) if not char_lengths.empty else 0.0,
            "char_min": int(char_lengths.min()) if not char_lengths.empty else 0,
            "char_max": int(char_lengths.max()) if not char_lengths.empty else 0,
            "char_std": float(char_lengths.std()) if not char_lengths.empty else 0.0,
            "char_p25": float(char_lengths.quantile(0.25)) if not char_lengths.empty else 0.0,
            "char_p75": float(char_lengths.quantile(0.75)) if not char_lengths.empty else 0.0,
            "char_p95": float(char_lengths.quantile(0.95)) if not char_lengths.empty else 0.0,
            
            "token_mean": float(token_lengths.mean()) if not token_lengths.empty else 0.0,
            "token_median": float(token_lengths.median()) if not token_lengths.empty else 0.0,
            "token_min": int(token_lengths.min()) if not token_lengths.empty else 0,
            "token_max": int(token_lengths.max()) if not token_lengths.empty else 0,
            "token_std": float(token_lengths.std()) if not token_lengths.empty else 0.0,
            "token_p95": float(token_lengths.quantile(0.95)) if not token_lengths.empty else 0.0,
        }
    else:
        char_lengths = pd.Series([], dtype=float)
        token_lengths = pd.Series([], dtype=float)
        dup_rows = 0
        dup_texts = 0
        mean_scripts = {}
        len_stats = {}
        
    # Class distribution
    if label_col is not None and label_col in df.columns:
        class_dist = df[label_col].value_counts(dropna=False).to_dict()
        class_dist_clean = {str(k): int(v) for k, v in class_dist.items()}
    else:
        class_dist_clean = {"UNLABELLED": num_rows}
        
    audit_record = {
        "language": language,
        "split": split_name,
        "filename": os.path.basename(file_path),
        "file_size_bytes": os.path.getsize(file_path),
        "num_rows": num_rows,
        "num_cols": num_cols,
        "col_names": col_names,
        "text_col": str(text_col),
        "label_col": str(label_col) if label_col is not None else "NONE",
        "missing_text_count": int(missing_text),
        "missing_label_count": int(missing_label),
        "duplicate_rows": dup_rows,
        "duplicate_texts": dup_texts,
        "class_distribution": class_dist_clean,
        "script_stats": mean_scripts,
        **len_stats
    }
    
    return audit_record, df, text_col, label_col

def run_comprehensive_audit():
    files_to_audit = [
        # Telugu
        {"lang": "telugu", "split": "train", "file": os.path.join(RAW_DIR, "telugu", "training_data_telugu-hate.xlsx"), "sep": None, "has_header": True},
        {"lang": "telugu", "split": "test", "file": os.path.join(RAW_DIR, "telugu", "telugu-hate-speech-test.xlsx"), "sep": None, "has_header": False},
        
        # Tamil
        {"lang": "tamil", "split": "train", "file": os.path.join(RAW_DIR, "tamil", "tamil_offensive_full_train.csv"), "sep": "\t", "has_header": False},
        {"lang": "tamil", "split": "dev", "file": os.path.join(RAW_DIR, "tamil", "tamil_offensive_full_dev.csv"), "sep": "\t", "has_header": False},
        {"lang": "tamil", "split": "test", "file": os.path.join(RAW_DIR, "tamil", "tamil_offensive_full_test.csv"), "sep": "\t", "has_header": False},
        {"lang": "tamil", "split": "full", "file": os.path.join(RAW_DIR, "tamil", "tamil_offensive_full.csv"), "sep": "\t", "has_header": False},
        
        # Malayalam
        {"lang": "malayalam", "split": "train", "file": os.path.join(RAW_DIR, "malayalam", "mal_full_offensive_train.csv"), "sep": "\t", "has_header": False},
        {"lang": "malayalam", "split": "dev", "file": os.path.join(RAW_DIR, "malayalam", "mal_full_offensive_dev.csv"), "sep": "\t", "has_header": False},
        {"lang": "malayalam", "split": "test", "file": os.path.join(RAW_DIR, "malayalam", "mal_full_offensive_test.csv"), "sep": "\t", "has_header": False},
        {"lang": "malayalam", "split": "full", "file": os.path.join(RAW_DIR, "malayalam", "mal_full_offensive.csv"), "sep": "\t", "has_header": False},
    ]
    
    audit_results = []
    loaded_dfs = {}
    
    print("=" * 80)
    print("STARTING PROGRAMMATIC DATASET AUDIT (PHASE 1)")
    print("=" * 80)
    
    for item in files_to_audit:
        fpath = item["file"]
        if not os.path.exists(fpath):
            print(f"WARNING: File not found: {fpath}")
            continue
            
        record, df, t_col, l_col = audit_file(
            fpath, 
            language=item["lang"], 
            split_name=item["split"], 
            sep=item["sep"], 
            has_header=item["has_header"]
        )
        audit_results.append(record)
        loaded_dfs[f"{item['lang']}_{item['split']}"] = (df, t_col, l_col)
        
        print(f"[*] Audited: {item['lang'].upper()} - {item['split'].upper()} ({record['filename']})")
        print(f"    Rows: {record['num_rows']} | Cols: {record['num_cols']} | Missing Text: {record['missing_text_count']} | Dup Texts: {record['duplicate_texts']}")
        print(f"    Labels: {record['class_distribution']}")
        print(f"    Text len (chars): Mean={record.get('char_mean',0):.1f}, Median={record.get('char_median',0):.1f}, Max={record.get('char_max',0)}")
        print(f"    Text len (tokens): Mean={record.get('token_mean',0):.1f}, Median={record.get('token_median',0):.1f}, Max={record.get('token_max',0)}")
        print(f"    Script breakdown: {record.get('script_stats',{})}")
        print("-" * 80)
        
    # Check split containment for Tamil and Malayalam
    print("\nVERIFYING SPLIT INTEGRITY AND CONTAINMENT:")
    for lang in ["tamil", "malayalam"]:
        if f"{lang}_full" in loaded_dfs and f"{lang}_train" in loaded_dfs and f"{lang}_dev" in loaded_dfs and f"{lang}_test" in loaded_dfs:
            full_df = loaded_dfs[f"{lang}_full"][0]
            train_df = loaded_dfs[f"{lang}_train"][0]
            dev_df = loaded_dfs[f"{lang}_dev"][0]
            test_df = loaded_dfs[f"{lang}_test"][0]
            
            sum_splits = len(train_df) + len(dev_df) + len(test_df)
            print(f"  {lang.capitalize()} Full count: {len(full_df)} | Sum of Train({len(train_df)}) + Dev({len(dev_df)}) + Test({len(test_df)}) = {sum_splits}")
            if len(full_df) == sum_splits:
                print(f"  -> Exact partition match for {lang}: Full == Train + Dev + Test.")
            else:
                print(f"  -> Discrepancy for {lang}: Diff = {len(full_df) - sum_splits}")
                
    # Check Cross-split duplicate leakage
    print("\nCHECKING CROSS-SPLIT DUPLICATE LEAKAGE:")
    for lang in ["tamil", "malayalam"]:
        train_texts = set(loaded_dfs[f"{lang}_train"][0][0].dropna().astype(str))
        dev_texts = set(loaded_dfs[f"{lang}_dev"][0][0].dropna().astype(str))
        test_texts = set(loaded_dfs[f"{lang}_test"][0][0].dropna().astype(str))
        
        train_dev_overlap = len(train_texts.intersection(dev_texts))
        train_test_overlap = len(train_texts.intersection(test_texts))
        dev_test_overlap = len(dev_texts.intersection(test_texts))
        
        print(f"  {lang.capitalize()} text overlap between Train & Dev: {train_dev_overlap}")
        print(f"  {lang.capitalize()} text overlap between Train & Test: {train_test_overlap}")
        print(f"  {lang.capitalize()} text overlap between Dev & Test: {dev_test_overlap}")

    # Check Telugu test overlap with Telugu train
    telugu_train_texts = set(loaded_dfs["telugu_train"][0]["Comments"].dropna().astype(str))
    telugu_test_texts = set(loaded_dfs["telugu_test"][0][1].dropna().astype(str))
    print(f"  Telugu text overlap between Train & Test: {len(telugu_train_texts.intersection(telugu_test_texts))}")

    # Save summary audit table
    summary_rows = []
    for r in audit_results:
        summary_rows.append({
            "Language": r["language"],
            "Split": r["split"],
            "Filename": r["filename"],
            "Rows": r["num_rows"],
            "Cols": r["num_cols"],
            "Text_Column": r["text_col"],
            "Label_Column": r["label_col"],
            "Missing_Text": r["missing_text_count"],
            "Missing_Label": r["missing_label_count"],
            "Duplicate_Rows": r["duplicate_rows"],
            "Duplicate_Texts": r["duplicate_texts"],
            "Char_Mean": round(r.get("char_mean", 0), 2),
            "Char_Median": r.get("char_median", 0),
            "Char_Max": r.get("char_max", 0),
            "Token_Mean": round(r.get("token_mean", 0), 2),
            "Token_Median": r.get("token_median", 0),
            "Token_Max": r.get("token_max", 0),
            "Class_Distribution": str(r["class_distribution"]),
        })
        
    audit_summary_df = pd.DataFrame(summary_rows)
    audit_csv_path = os.path.join(TABLES_DIR, "dataset_audit.csv")
    audit_summary_df.to_csv(audit_csv_path, index=False)
    print(f"\n[+] Saved audit summary to: {audit_csv_path}")
    
    # Save detailed class distribution table
    class_rows = []
    for r in audit_results:
        if r["split"] != "full": # avoid double counting full
            for lbl, count in r["class_distribution"].items():
                pct = (count / r["num_rows"]) * 100
                class_rows.append({
                    "Language": r["language"],
                    "Split": r["split"],
                    "Original_Label": lbl,
                    "Count": count,
                    "Percentage": round(pct, 2)
                })
    class_df = pd.DataFrame(class_rows)
    class_csv_path = os.path.join(TABLES_DIR, "class_distributions.csv")
    class_df.to_csv(class_csv_path, index=False)
    print(f"[+] Saved class distributions to: {class_csv_path}")

    # Generate Publication-Quality Visualizations
    generate_audit_figures(loaded_dfs, audit_results)
    
    return audit_results, audit_summary_df

def generate_audit_figures(loaded_dfs, audit_results):
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})
    
    # 1. Class Distribution Barplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Raw Class Distributions Across Active Datasets (Train Splits)", fontsize=14, fontweight='bold', y=1.02)
    
    # Telugu Train
    telugu_train_dist = [r["class_distribution"] for r in audit_results if r["language"]=="telugu" and r["split"]=="train"][0]
    labels_tel = list(telugu_train_dist.keys())
    counts_tel = list(telugu_train_dist.values())
    sns.barplot(x=labels_tel, y=counts_tel, ax=axes[0], palette="crest")
    axes[0].set_title("Telugu Train (Hate Speech)", fontweight='bold')
    axes[0].set_ylabel("Sample Count")
    for i, v in enumerate(counts_tel):
        axes[0].text(i, v + 25, f"{v}\n({v/sum(counts_tel):.1%})", ha='center', fontsize=9)
        
    # Tamil Train
    tamil_train_dist = [r["class_distribution"] for r in audit_results if r["language"]=="tamil" and r["split"]=="train"][0]
    labels_tam = list(tamil_train_dist.keys())
    counts_tam = list(tamil_train_dist.values())
    sns.barplot(x=counts_tam, y=labels_tam, ax=axes[1], palette="magma", orient='h')
    axes[1].set_title("Tamil Train (Offensive Language)", fontweight='bold')
    axes[1].set_xlabel("Sample Count")
    for i, v in enumerate(counts_tam):
        axes[1].text(v + 200, i, f" {v} ({v/sum(counts_tam):.1%})", va='center', fontsize=8)

    # Malayalam Train
    mal_train_dist = [r["class_distribution"] for r in audit_results if r["language"]=="malayalam" and r["split"]=="train"][0]
    labels_mal = list(mal_train_dist.keys())
    counts_mal = list(mal_train_dist.values())
    sns.barplot(x=counts_mal, y=labels_mal, ax=axes[2], palette="viridis", orient='h')
    axes[2].set_title("Malayalam Train (Offensive Language)", fontweight='bold')
    axes[2].set_xlabel("Sample Count")
    for i, v in enumerate(counts_mal):
        axes[2].text(v + 100, i, f" {v} ({v/sum(counts_mal):.1%})", va='center', fontsize=8)
        
    plt.tight_layout()
    dist_fig_path = os.path.join(FIGURES_DIR, "audit_class_distributions.png")
    plt.savefig(dist_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved figure: {dist_fig_path}")

    # 2. Text Length Distributions (Tokens)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Text Length (Word Token Count) Distributions Across Datasets", fontsize=14, fontweight='bold', y=1.02)
    
    telugu_tokens = loaded_dfs["telugu_train"][0]["Comments"].dropna().astype(str).apply(lambda s: len(s.split()))
    tamil_tokens = loaded_dfs["tamil_train"][0][0].dropna().astype(str).apply(lambda s: len(s.split()))
    mal_tokens = loaded_dfs["malayalam_train"][0][0].dropna().astype(str).apply(lambda s: len(s.split()))
    
    # Clip at 99th percentile for clean visualization
    sns.histplot(telugu_tokens[telugu_tokens <= telugu_tokens.quantile(0.99)], bins=30, kde=True, ax=axes[0], color="#2b5c8f")
    axes[0].set_title(f"Telugu Train (Median={telugu_tokens.median():.0f}, 95th%={telugu_tokens.quantile(0.95):.0f})", fontweight='bold')
    axes[0].set_xlabel("Word Token Count")
    axes[0].set_ylabel("Frequency")
    
    sns.histplot(tamil_tokens[tamil_tokens <= tamil_tokens.quantile(0.99)], bins=30, kde=True, ax=axes[1], color="#a13d63")
    axes[1].set_title(f"Tamil Train (Median={tamil_tokens.median():.0f}, 95th%={tamil_tokens.quantile(0.95):.0f})", fontweight='bold')
    axes[1].set_xlabel("Word Token Count")
    axes[1].set_ylabel("Frequency")
    
    sns.histplot(mal_tokens[mal_tokens <= mal_tokens.quantile(0.99)], bins=30, kde=True, ax=axes[2], color="#2a9d8f")
    axes[2].set_title(f"Malayalam Train (Median={mal_tokens.median():.0f}, 95th%={mal_tokens.quantile(0.95):.0f})", fontweight='bold')
    axes[2].set_xlabel("Word Token Count")
    axes[2].set_ylabel("Frequency")
    
    plt.tight_layout()
    len_fig_path = os.path.join(FIGURES_DIR, "audit_text_lengths.png")
    plt.savefig(len_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved figure: {len_fig_path}")

    # 3. Script Composition Breakdown
    fig, ax = plt.subplots(figsize=(10, 5))
    train_records = [r for r in audit_results if r["split"]=="train"]
    langs = [r["language"].capitalize() for r in train_records]
    
    native_script = []
    latin_script = []
    other_script = []
    
    for r in train_records:
        l = r["language"]
        scripts = r.get("script_stats", {})
        native_script.append(scripts.get(l, 0.0) * 100)
        latin_script.append(scripts.get("latin", 0.0) * 100)
        other_script.append((scripts.get("other", 0.0) + sum(v for k, v in scripts.items() if k not in [l, "latin", "other"])) * 100)
        
    x = np.arange(len(langs))
    width = 0.55
    
    p1 = ax.bar(x, native_script, width, label='Native Indic Script', color='#e76f51')
    p2 = ax.bar(x, latin_script, width, bottom=native_script, label='Latin / Romanized Script', color='#457b9d')
    bottom_other = [i + j for i, j in zip(native_script, latin_script)]
    p3 = ax.bar(x, other_script, width, bottom=bottom_other, label='Punctuation / Symbols / Other', color='#a8dadc')
    
    ax.set_ylabel('Character Percentage (%)', fontweight='bold')
    ax.set_title('Code-Mixing & Script Composition in Training Sets', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)
    
    for i in range(len(langs)):
        ax.text(i, native_script[i]/2, f"{native_script[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold')
        ax.text(i, native_script[i] + latin_script[i]/2, f"{latin_script[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold')
        
    plt.tight_layout()
    script_fig_path = os.path.join(FIGURES_DIR, "audit_script_composition.png")
    plt.savefig(script_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved figure: {script_fig_path}")

if __name__ == "__main__":
    run_comprehensive_audit()
