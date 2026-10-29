"""
TRUST-X Final Analysis & Reporting Script
==========================================
Reads the completed experiments.csv (12 rows) and produces:

TABLES
------
  tables/final_results.csv        -- all 12 experiments, all metrics
  tables/model_averages.csv       -- per-model averages across 4 languages
  tables/language_averages.csv    -- per-language averages across 3 models

FIGURES  (saved to figures/final/)
-------
  macro_f1_comparison.png         -- grouped bar chart (languages x models)
  twi_comparison.png              -- grouped bar chart (languages x models)
  trust_pillars_comparison.png    -- radar chart per model
  accuracy_vs_macro_f1.png        -- scatter plot
  macro_f1_vs_twi.png             -- scatter plot

Usage (from project root):
  python scripts/generate_final_analysis.py [--csv PATH_TO_EXPERIMENTS_CSV]

Calibration score is re-derived from ECE and Brier using the same formula
as src/twi/twi.py:normalize_calibration_score():
  calib_score = 0.5 * (1 - ECE) + 0.5 * max(0, 1 - 2*Brier)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_CSV = os.path.join(ROOT, "experiments.csv")
FIGURES_DIR = os.path.join(ROOT, "figures", "final")
TABLES_DIR  = os.path.join(ROOT, "tables")

# ── Display ordering & identity ──────────────────────────────────────────────
LANGUAGE_ORDER = ["telugu", "tamil", "malayalam", "hindi"]
MODEL_ORDER    = ["mbert", "indicbert", "xlmr"]

MODEL_DISPLAY = {
    "mbert":     "mBERT",
    "indicbert": "IndicBERT",
    "xlmr":      "XLM-R",
}
LANGUAGE_DISPLAY = {
    "telugu":    "Telugu",
    "tamil":     "Tamil",
    "malayalam": "Malayalam",
    "hindi":     "Hindi",
}

MODEL_COLORS = {
    "mbert":     "#3a86ff",   # vivid blue
    "indicbert": "#ff006e",   # vivid pink/rose
    "xlmr":      "#fb5607",   # vivid orange
}
LANGUAGE_MARKERS = {
    "telugu":    "o",
    "tamil":     "s",
    "malayalam": "^",
    "hindi":     "D",
}

PILLAR_LABELS = [
    "Macro-F1\n(Performance)",
    "Explainability\n(Compactness)",
    "Robustness",
    "Calibration",
]

# ── Global plot style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})


# ═════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

REQUIRED_COLS = [
    "experiment_id", "language", "model",
    "test_accuracy", "test_macro_f1",
    "ece", "brier",
    "robustness_score", "explainability_score",
    "twi", "status",
]

def load_and_validate(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\n[!] experiments.csv not found at:\n    {csv_path}\n"
            "    Please sync it from Google Colab/Drive before running.\n"
        )

    df = pd.read_csv(csv_path)
    print(f"[+] Loaded {len(df)} rows from: {csv_path}")
    df.columns = df.columns.str.strip().str.lower()

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[!] experiments.csv is missing columns: {missing}")

    completed = df[df["status"] == "COMPLETED"]
    if len(completed) != 12:
        print(f"[!] WARNING: Expected 12 COMPLETED rows, found {len(completed)}.")
    else:
        print("[+] All 12 experiments confirmed COMPLETED.")

    df["language"] = df["language"].str.lower().str.strip()
    df["model"]    = df["model"].str.lower().str.strip()

    # Derive calibration score from ECE + Brier (mirrors src/twi/twi.py)
    df["calibration_score"] = (
        0.5 * (1.0 - df["ece"]).clip(lower=0.0)
        + 0.5 * (1.0 - 2.0 * df["brier"]).clip(lower=0.0)
    )
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 2.  TABLE GENERATION
# ═════════════════════════════════════════════════════════════════════════════

NUMERIC_COLS = [
    "test_accuracy", "test_macro_f1",
    "ece", "brier", "calibration_score",
    "robustness_score", "explainability_score", "twi",
]

def make_tables(df: pd.DataFrame):
    os.makedirs(TABLES_DIR, exist_ok=True)

    # 2a. Full results
    cols_out = ["experiment_id", "language", "model"] + NUMERIC_COLS + ["status"]
    out = df[[c for c in cols_out if c in df.columns]].copy()
    out["language"] = out["language"].map(LANGUAGE_DISPLAY).fillna(out["language"])
    out["model"]    = out["model"].map(MODEL_DISPLAY).fillna(out["model"])
    out = out.round(4)
    path = os.path.join(TABLES_DIR, "final_results.csv")
    out.to_csv(path, index=False)
    print(f"[+] Saved: {path}")

    # 2b. Model averages
    model_avg = (
        df.groupby("model")[NUMERIC_COLS].mean()
        .reindex([m for m in MODEL_ORDER if m in df["model"].unique()])
        .round(4).reset_index()
    )
    model_avg["model"] = model_avg["model"].map(MODEL_DISPLAY).fillna(model_avg["model"])
    path = os.path.join(TABLES_DIR, "model_averages.csv")
    model_avg.to_csv(path, index=False)
    print(f"[+] Saved: {path}")

    # 2c. Language averages
    lang_avg = (
        df.groupby("language")[NUMERIC_COLS].mean()
        .reindex([l for l in LANGUAGE_ORDER if l in df["language"].unique()])
        .round(4).reset_index()
    )
    lang_avg["language"] = lang_avg["language"].map(LANGUAGE_DISPLAY).fillna(lang_avg["language"])
    path = os.path.join(TABLES_DIR, "language_averages.csv")
    lang_avg.to_csv(path, index=False)
    print(f"[+] Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 3.  FIGURE HELPER
# ═════════════════════════════════════════════════════════════════════════════

def savefig(name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved figure: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 4.  FIGURE 1 – Macro-F1 Grouped Bar Chart
# ═════════════════════════════════════════════════════════════════════════════

def plot_macro_f1_comparison(df: pd.DataFrame):
    langs  = [l for l in LANGUAGE_ORDER if l in df["language"].unique()]
    models = [m for m in MODEL_ORDER   if m in df["model"].unique()]
    x = np.arange(len(langs))
    width   = 0.22
    offsets = np.linspace(-(len(models)-1)/2, (len(models)-1)/2, len(models)) * width

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(models):
        vals = []
        for lang in langs:
            row = df[(df["language"] == lang) & (df["model"] == model)]
            vals.append(float(row["test_macro_f1"].values[0]) if len(row) else 0.0)
        bars = ax.bar(x + offsets[i], vals, width=width*0.88,
                      color=MODEL_COLORS[model], label=MODEL_DISPLAY[model],
                      edgecolor="white", linewidth=0.6, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color=MODEL_COLORS[model])

    ax.set_xticks(x)
    ax.set_xticklabels([LANGUAGE_DISPLAY[l] for l in langs], fontsize=11)
    ax.set_ylabel("Test Macro-F1", fontsize=12, fontweight="bold")
    ax.set_title("TRUST-X: Macro-F1 Comparison Across Languages & Models",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_ylim(0, 1.08)
    ax.legend(title="Model", fontsize=10, title_fontsize=10,
              framealpha=0.85, loc="upper right")
    ax.set_axisbelow(True)
    fig.tight_layout()
    savefig("macro_f1_comparison.png")


# ═════════════════════════════════════════════════════════════════════════════
# 5.  FIGURE 2 – TWI Grouped Bar Chart
# ═════════════════════════════════════════════════════════════════════════════

def plot_twi_comparison(df: pd.DataFrame):
    langs  = [l for l in LANGUAGE_ORDER if l in df["language"].unique()]
    models = [m for m in MODEL_ORDER   if m in df["model"].unique()]
    x = np.arange(len(langs))
    width   = 0.22
    offsets = np.linspace(-(len(models)-1)/2, (len(models)-1)/2, len(models)) * width

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(models):
        vals = []
        for lang in langs:
            row = df[(df["language"] == lang) & (df["model"] == model)]
            vals.append(float(row["twi"].values[0]) if len(row) else 0.0)
        bars = ax.bar(x + offsets[i], vals, width=width*0.88,
                      color=MODEL_COLORS[model], label=MODEL_DISPLAY[model],
                      edgecolor="white", linewidth=0.6, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color=MODEL_COLORS[model])

    ax.set_xticks(x)
    ax.set_xticklabels([LANGUAGE_DISPLAY[l] for l in langs], fontsize=11)
    ax.set_ylabel("Trustworthiness Index (TWI – Balanced Profile)",
                  fontsize=11, fontweight="bold")
    ax.set_title("TRUST-X: TWI Comparison Across Languages & Models",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_ylim(0, 1.08)
    ax.legend(title="Model", fontsize=10, title_fontsize=10,
              framealpha=0.85, loc="upper right")
    ax.set_axisbelow(True)
    fig.tight_layout()
    savefig("twi_comparison.png")


# ═════════════════════════════════════════════════════════════════════════════
# 6.  FIGURE 3 – Trust Pillars Radar Chart
# ═════════════════════════════════════════════════════════════════════════════

def plot_trust_pillars_radar(df: pd.DataFrame):
    models      = [m for m in MODEL_ORDER if m in df["model"].unique()]
    pillar_cols = ["test_macro_f1", "explainability_score",
                   "robustness_score", "calibration_score"]
    N      = len(PILLAR_LABELS)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(PILLAR_LABELS, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],
                       fontsize=8, color="grey")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.spines["polar"].set_visible(False)

    for model in models:
        subset = df[df["model"] == model]
        vals   = [float(subset[c].mean()) for c in pillar_cols]
        vals  += vals[:1]
        ax.plot(angles, vals, color=MODEL_COLORS[model], linewidth=2.2,
                label=MODEL_DISPLAY[model], zorder=3)
        ax.fill(angles, vals, color=MODEL_COLORS[model], alpha=0.12)

    ax.set_title(
        "TRUST-X: Trust Pillar Profiles by Model\n(Average Across 4 Languages)",
        fontsize=12, fontweight="bold", pad=20
    )
    ax.legend(loc="lower right", fontsize=10, framealpha=0.85,
              bbox_to_anchor=(1.28, -0.08))
    plt.tight_layout()
    savefig("trust_pillars_comparison.png")


# ═════════════════════════════════════════════════════════════════════════════
# 7.  FIGURE 4 – Accuracy vs Macro-F1 Scatter
# ═════════════════════════════════════════════════════════════════════════════

# Specific annotation layout offsets & leader line config per experiment
ACC_F1_ANNOTATION_CONFIGS = {
    "telugu_indicbert_seed42": {
        "xytext": (14, -5),
        "ha": "left",
        "va": "center",
        "arrow": False
    },
    "telugu_mbert_seed42": {
        "xytext": (-45, 18),
        "ha": "right",
        "va": "bottom",
        "arrow": True
    },
    "telugu_xlmr_seed42": {
        "xytext": (-45, -20),
        "ha": "right",
        "va": "top",
        "arrow": True
    },
    "hindi_indicbert_seed42": {
        "xytext": (-15, -34),
        "ha": "center",
        "va": "top",
        "arrow": True
    },
    "hindi_xlmr_seed42": {
        "xytext": (-48, 20),
        "ha": "right",
        "va": "bottom",
        "arrow": True
    },
    "hindi_mbert_seed42": {
        "xytext": (-20, 28),
        "ha": "right",
        "va": "bottom",
        "arrow": True
    },
    "tamil_xlmr_seed42": {
        "xytext": (25, -25),
        "ha": "left",
        "va": "top",
        "arrow": True
    },
    "tamil_indicbert_seed42": {
        "xytext": (32, -3),
        "ha": "left",
        "va": "center",
        "arrow": True
    },
    "tamil_mbert_seed42": {
        "xytext": (25, 20),
        "ha": "left",
        "va": "bottom",
        "arrow": True
    },
    "malayalam_xlmr_seed42": {
        "xytext": (-65, 0),
        "ha": "right",
        "va": "center",
        "arrow": True
    },
    "malayalam_mbert_seed42": {
        "xytext": (-65, 18),
        "ha": "right",
        "va": "bottom",
        "arrow": True
    },
    "malayalam_indicbert_seed42": {
        "xytext": (-65, -20),
        "ha": "right",
        "va": "top",
        "arrow": True
    },
}

def plot_accuracy_vs_macro_f1(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 7))

    for _, row in df.iterrows():
        lang   = row["language"]
        model  = row["model"]
        exp_id = row.get("experiment_id", f"{lang}_{model}_seed42")
        x      = row["test_accuracy"]
        y      = row["test_macro_f1"]

        ax.scatter(
            x, y,
            color=MODEL_COLORS.get(model, "grey"),
            marker=LANGUAGE_MARKERS.get(lang, "o"),
            s=130, edgecolors="white", linewidths=1.0, zorder=5
        )

        label = f"{LANGUAGE_DISPLAY.get(lang, lang)}-{MODEL_DISPLAY.get(model, model)}"
        cfg   = ACC_F1_ANNOTATION_CONFIGS.get(
            exp_id, {"xytext": (8, 6), "ha": "left", "va": "bottom", "arrow": False}
        )

        arrowprops = None
        if cfg.get("arrow", False):
            arrowprops = dict(
                arrowstyle="->",
                color="#555555",
                lw=0.9,
                shrinkA=2,
                shrinkB=5,
                connectionstyle="arc3,rad=0.0"
            )

        ax.annotate(
            label,
            xy=(x, y),
            xytext=cfg["xytext"],
            textcoords="offset points",
            fontsize=9,
            fontweight="medium",
            color="#222222",
            ha=cfg.get("ha", "left"),
            va=cfg.get("va", "bottom"),
            arrowprops=arrowprops,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
            zorder=6
        )

    ax.set_xlim(0.62, 0.99)
    ax.set_ylim(0.44, 0.83)

    lims = [
        max(ax.get_xlim()[0], ax.get_ylim()[0]),
        min(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.25, linewidth=1, zorder=2)

    model_handles = [
        mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_DISPLAY[m])
        for m in MODEL_ORDER if m in df["model"].unique()
    ]
    lang_handles = [
        Line2D([0], [0], marker=LANGUAGE_MARKERS[l], color="#666666",
               markerfacecolor="#666666", markersize=8, linestyle="None",
               label=LANGUAGE_DISPLAY[l])
        for l in LANGUAGE_ORDER if l in df["language"].unique()
    ]
    ax.legend(handles=model_handles + lang_handles,
              fontsize=9, framealpha=0.9, loc="lower center",
              bbox_to_anchor=(0.5, -0.20), ncol=4)

    ax.set_xlabel("Test Accuracy", fontsize=12, fontweight="bold")
    ax.set_ylabel("Test Macro-F1", fontsize=12, fontweight="bold")
    ax.set_title("TRUST-X: Accuracy vs. Macro-F1 (All 12 Experiments)",
                 fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    savefig("accuracy_vs_macro_f1.png")


# ═════════════════════════════════════════════════════════════════════════════
# 8.  FIGURE 5 – Macro-F1 vs TWI Scatter
# ═════════════════════════════════════════════════════════════════════════════

def plot_macro_f1_vs_twi(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 6))

    for _, row in df.iterrows():
        lang  = row["language"]
        model = row["model"]
        ax.scatter(
            row["test_macro_f1"], row["twi"],
            color=MODEL_COLORS.get(model, "grey"),
            marker=LANGUAGE_MARKERS.get(lang, "o"),
            s=120, edgecolors="white", linewidths=0.8, zorder=4
        )
        ax.annotate(
            f"{LANGUAGE_DISPLAY.get(lang, lang)}\n{MODEL_DISPLAY.get(model, model)}",
            xy=(row["test_macro_f1"], row["twi"]),
            xytext=(5, 4), textcoords="offset points",
            fontsize=7, color="#333333"
        )

    lo = min(df["test_macro_f1"].min(), df["twi"].min()) - 0.02
    hi = max(df["test_macro_f1"].max(), df["twi"].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.2, linewidth=1)

    model_handles = [
        mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_DISPLAY[m])
        for m in MODEL_ORDER if m in df["model"].unique()
    ]
    lang_handles = [
        Line2D([0], [0], marker=LANGUAGE_MARKERS[l], color="grey",
               markerfacecolor="grey", markersize=8, linestyle="None",
               label=LANGUAGE_DISPLAY[l])
        for l in LANGUAGE_ORDER if l in df["language"].unique()
    ]
    ax.legend(handles=model_handles + lang_handles,
              fontsize=8, framealpha=0.85, loc="lower right", ncol=2)

    ax.set_xlabel("Test Macro-F1 (Performance Pillar)", fontsize=12, fontweight="bold")
    ax.set_ylabel("TWI – Balanced Profile", fontsize=12, fontweight="bold")
    ax.set_title(
        "TRUST-X: Macro-F1 vs. TWI\nPerformance vs. Holistic Trustworthiness",
        fontsize=12, fontweight="bold", pad=12
    )
    fig.tight_layout()
    savefig("macro_f1_vs_twi.png")


# ═════════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TRUST-X Final Analysis – generates all tables and figures."
    )
    parser.add_argument(
        "--csv", type=str, default=DEFAULT_CSV,
        help=f"Path to experiments.csv (default: {DEFAULT_CSV})"
    )
    parser.add_argument(
        "--figure", type=str, default=None,
        choices=["macro_f1_comparison", "twi_comparison", "trust_pillars_comparison", "accuracy_vs_macro_f1", "macro_f1_vs_twi"],
        help="Generate only a specific figure."
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("TRUST-X FINAL ANALYSIS & REPORTING")
    print("=" * 70)

    df = load_and_validate(args.csv)
    df = df[df["status"] == "COMPLETED"].reset_index(drop=True)

    if args.figure:
        print(f"\n[*] Generating single figure: {args.figure} ...")
        fig_map = {
            "macro_f1_comparison": plot_macro_f1_comparison,
            "twi_comparison": plot_twi_comparison,
            "trust_pillars_comparison": plot_trust_pillars_radar,
            "accuracy_vs_macro_f1": plot_accuracy_vs_macro_f1,
            "macro_f1_vs_twi": plot_macro_f1_vs_twi,
        }
        fig_map[args.figure](df)
    else:
        print("\n[*] Generating tables ...")
        make_tables(df)

        print("\n[*] Generating figures ...")
        plot_macro_f1_comparison(df)
        plot_twi_comparison(df)
        plot_trust_pillars_radar(df)
        plot_accuracy_vs_macro_f1(df)
        plot_macro_f1_vs_twi(df)

    print("\n" + "=" * 70)
    print("TRUST-X FINAL ANALYSIS COMPLETE")
    if not args.figure:
        print(f"  Tables  -> {TABLES_DIR}/")
    print(f"  Figures -> {FIGURES_DIR}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
