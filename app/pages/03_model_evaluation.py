"""
TRUST-X Application: Page 03 - Model Evaluation
Inspect frozen evaluation metrics, calibration, and trust pillar breakdown for any of the 12 experiments.
"""

import streamlit as st

from app.config import (
    LANGUAGES,
    MODELS,
    LANGUAGE_DISPLAY,
    MODEL_DISPLAY,
    MODEL_SHORT_DISPLAY,
    LANGUAGE_REVERSE,
    MODEL_REVERSE,
)
from app.data_loader import get_experiment_record
from app.components.metric_cards import render_experiment_metric_cards
from app.components.trust_pillar_gauge import render_trust_pillars_breakdown


def render():
    st.title("📈 Model Evaluation")
    st.caption("Deep-dive inspection of frozen test metrics across the four trustworthiness pillars.")

    # Top Selection Controls
    col_lang, col_model = st.columns(2)
    with col_lang:
        selected_lang_display = st.selectbox(
            "Select Language:",
            options=[LANGUAGE_DISPLAY[l] for l in LANGUAGES],
            index=0,
        )
        selected_lang = LANGUAGE_REVERSE[selected_lang_display]

    with col_model:
        selected_model_display = st.selectbox(
            "Select Model Architecture:",
            options=[MODEL_SHORT_DISPLAY[m] for m in MODELS],
            index=2,  # Default to XLM-R
        )
        selected_model = MODEL_REVERSE[selected_model_display]

    # Fetch Frozen Data
    record = get_experiment_record(selected_lang, selected_model)

    if record is None:
        st.error(f"No experimental data found for {selected_lang_display} - {selected_model_display}.")
        return

    # Experiment Info Banner
    exp_id = record.get("experiment_id", f"{selected_lang}_{selected_model}")
    status = record.get("status", "COMPLETED")
    
    st.markdown(
        f"""
        <div style="background-color: #f1f3f5; padding: 10px 16px; border-radius: 6px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong>Experiment ID:</strong> <code>{exp_id}</code> &nbsp;|&nbsp;
                <strong>Language:</strong> {selected_lang_display} &nbsp;|&nbsp;
                <strong>Model:</strong> {MODEL_DISPLAY[selected_model]}
            </div>
            <div>
                <span style="background-color: #198754; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold;">
                    Status: {status} (Frozen)
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Metric Cards
    render_experiment_metric_cards(record)

    st.divider()

    # 2. Trust Pillars Breakdown
    render_trust_pillars_breakdown(record)

    st.divider()

    # 3. Scientific Discussion Note
    st.markdown("##### 🔍 Pillar Analysis for this Experiment")
    f1 = float(record.get("test_macro_f1", 0.0))
    twi = float(record.get("twi", 0.0))
    ece = float(record.get("ece", 0.0))
    rob = float(record.get("robustness_score", 0.0))
    gap = twi - f1

    st.markdown(
        f"""
        - **Performance vs Trustworthiness:** This model achieved a **Macro-F1 of {f1:.4f}** and a **Balanced TWI of {twi:.4f}** 
          (Gap: `{gap:+.4f}`). A positive gap indicates superior robustness, calibration, or explainability compared to raw accuracy alone.
        - **Calibration Audit:** ECE is `{ece:.4f}`. {"Strong calibration with low confidence error." if ece < 0.15 else "Higher confidence error; requires probability scaling before automated deployment."}
        - **Adversarial Resilience:** Robustness score is `{rob:.4f}` under character and word-level perturbations.
        """
    )


if __name__ == "__main__":
    render()
