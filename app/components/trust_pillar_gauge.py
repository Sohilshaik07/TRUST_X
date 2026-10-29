"""
TRUST-X Trust Pillar Gauge Component
Visualizes the 4 normalized trust pillars and their contributions to the Trustworthiness Index (TWI).
"""

from typing import Dict, Any
import pandas as pd
import streamlit as st


def render_trust_pillars_breakdown(record: Dict[str, Any]):
    """
    Renders a clear breakdown of the four TRUST-X pillars:
    - Performance (Macro-F1)
    - Explainability
    - Robustness
    - Calibration
    """
    st.markdown("##### 🏛️ Trust Pillars Breakdown")
    
    perf = float(record.get("test_macro_f1", 0.0))
    expl = float(record.get("explainability_score", 0.0))
    rob = float(record.get("robustness_score", 0.0))
    calib = float(record.get("calibration_score", 0.0))
    twi = float(record.get("twi", 0.0))
    
    col_chart, col_details = st.columns([3, 2])
    
    with col_chart:
        # Create a clean DataFrame for the bar chart
        pillars_df = pd.DataFrame({
            "Pillar": ["1. Performance", "2. Explainability", "3. Robustness", "4. Calibration"],
            "Score": [perf, expl, rob, calib],
        }).set_index("Pillar")
        
        st.bar_chart(pillars_df, horizontal=True, color="#1f77b4")
        st.caption("All 4 pillars normalized to [0, 1] range (higher = more trustworthy).")
        
    with col_details:
        st.markdown(
            f"""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e9ecef;">
                <h6 style="margin-top:0; color: #333;">Balanced TWI Weighting (0.25 each):</h6>
                <ul style="padding-left: 20px; font-size: 0.9em; margin-bottom: 0;">
                    <li><strong>Performance (0.25 × {perf:.4f}):</strong> {0.25 * perf:.4f}</li>
                    <li><strong>Explainability (0.25 × {expl:.4f}):</strong> {0.25 * expl:.4f}</li>
                    <li><strong>Robustness (0.25 × {rob:.4f}):</strong> {0.25 * rob:.4f}</li>
                    <li><strong>Calibration (0.25 × {calib:.4f}):</strong> {0.25 * calib:.4f}</li>
                </ul>
                <hr style="margin: 10px 0; border: none; border-top: 1px solid #ddd;">
                <div style="font-size: 1.05em; font-weight: bold; color: #0d6efd;">
                    Composite TWI: {twi:.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
