"""
TRUST-X Application: Page 05 - TWI Sensitivity
Sensitivity analysis across diverse pillar weighting profiles (Balanced, Performance-Heavy, Robustness-Heavy, Calibration-Heavy).
"""

import streamlit as st
import pandas as pd

from app.config import (
    FIG_SENSITIVITY,
)
from app.data_loader import (
    load_twi_sensitivity_df,
    load_twi_sensitivity_model_df,
    load_twi_sensitivity_lang_df,
)


def render():
    st.title("🔬 TWI Sensitivity Analysis")
    st.caption("Evaluating ranking stability and sensitivity of the Trustworthiness Index under alternative pillar weighting schemes.")

    st.markdown(
        """
        To confirm that model rankings are not an artifact of an arbitrary equal-weight assumption, 
        we evaluate the 12 experiments under **four distinct weighting schemes** representing different operational priorities.
        """
    )

    # 4 Weighting Schemes Cards
    st.subheader("⚖️ The Four Operational Weight Profiles")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div style="border: 1px solid #ced4da; border-radius: 6px; padding: 10px; background: #f8f9fa;">
                <strong>1. Balanced (Default)</strong><br>
                <small style="color: #495057;">Equal importance across all trust dimensions.</small>
                <ul style="font-size: 0.8em; margin-bottom: 0; padding-left: 16px;">
                    <li>Perf: 25%</li>
                    <li>Expl: 25%</li>
                    <li>Rob: 25%</li>
                    <li>Calib: 25%</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div style="border: 1px solid #ced4da; border-radius: 6px; padding: 10px; background: #f8f9fa;">
                <strong>2. Performance-Heavy</strong><br>
                <small style="color: #495057;">Emphasizes raw classification ability.</small>
                <ul style="font-size: 0.8em; margin-bottom: 0; padding-left: 16px;">
                    <li>Perf: 50%</li>
                    <li>Expl: 15%</li>
                    <li>Rob: 20%</li>
                    <li>Calib: 15%</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div style="border: 1px solid #ced4da; border-radius: 6px; padding: 10px; background: #f8f9fa;">
                <strong>3. Robustness-Heavy</strong><br>
                <small style="color: #495057;">Prioritizes noise and adversarial resilience.</small>
                <ul style="font-size: 0.8em; margin-bottom: 0; padding-left: 16px;">
                    <li>Perf: 20%</li>
                    <li>Expl: 15%</li>
                    <li>Rob: 45%</li>
                    <li>Calib: 20%</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div style="border: 1px solid #ced4da; border-radius: 6px; padding: 10px; background: #f8f9fa;">
                <strong>4. Calibration-Heavy</strong><br>
                <small style="color: #495057;">Prioritizes reliability of confidence probabilities.</small>
                <ul style="font-size: 0.8em; margin-bottom: 0; padding-left: 16px;">
                    <li>Perf: 20%</li>
                    <li>Expl: 15%</li>
                    <li>Rob: 20%</li>
                    <li>Calib: 45%</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # Sensitivity Figure
    st.subheader("📈 Sensitivity Visualization")
    if FIG_SENSITIVITY.exists():
        st.image(str(FIG_SENSITIVITY), caption="Figure 6: TWI across the four operational weight profiles.", use_container_width=True)
    else:
        st.error(f"Figure not found at: {FIG_SENSITIVITY}")

    st.divider()

    # Model Averages under Sensitivity Schemes
    st.subheader("📊 Cross-Language Model Averages Across Profiles")
    st.markdown("Mean TWI for each transformer architecture under each weighting profile:")
    try:
        sens_model_df = load_twi_sensitivity_model_df().copy()
        for col in ["balanced_twi", "performance_heavy_twi", "robustness_heavy_twi", "calibration_heavy_twi"]:
            if col in sens_model_df.columns:
                sens_model_df[col] = sens_model_df[col].apply(lambda x: f"{float(x):.4f}")
        
        sens_model_df.rename(columns={
            "model": "Model",
            "balanced_twi": "Balanced TWI (25/25/25/25)",
            "performance_heavy_twi": "Performance-Heavy (50/15/20/15)",
            "robustness_heavy_twi": "Robustness-Heavy (20/15/45/20)",
            "calibration_heavy_twi": "Calibration-Heavy (20/15/20/45)",
        }, inplace=True)
        st.dataframe(sens_model_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Error loading sensitivity model averages: {e}")

    st.divider()

    # Per-Experiment Sensitivity Table
    st.subheader("📋 Complete 12-Experiment Sensitivity Breakdown")
    try:
        sens_df = load_twi_sensitivity_df().copy()
        for col in ["balanced_twi", "performance_heavy_twi", "robustness_heavy_twi", "calibration_heavy_twi"]:
            if col in sens_df.columns:
                sens_df[col] = sens_df[col].apply(lambda x: f"{float(x):.4f}")
        
        sens_df.rename(columns={
            "experiment_id": "Experiment ID",
            "language": "Language",
            "model": "Model",
            "balanced_twi": "Balanced",
            "performance_heavy_twi": "Perf-Heavy",
            "robustness_heavy_twi": "Rob-Heavy",
            "calibration_heavy_twi": "Calib-Heavy",
        }, inplace=True)
        st.dataframe(sens_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Error loading experiment sensitivity table: {e}")

    st.divider()

    # Scientific Conclusion Box
    st.success(
        """
        ✅ **Key Scientific Takeaway:**
        The hierarchy **XLM-RoBERTa > mBERT > IndicBERT** remains strictly invariant across all 4 operational weighting schemes.
        Rank correlation (Spearman's $\\rho$) between the balanced index and alternative schemes exceeds **0.93**, 
        confirming that TRUST-X rankings reflect fundamental model properties rather than weighting artifacts.
        """
    )


if __name__ == "__main__":
    render()
