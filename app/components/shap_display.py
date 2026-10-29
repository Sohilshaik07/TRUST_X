"""
TRUST-X SHAP Token Attribution Display Component
Renders token-level attributions with intuitive color coding.
"""

from typing import List, Optional
import html
import streamlit as st


def render_token_attributions(
    tokens: List[str],
    attributions: List[float],
    predicted_label: Optional[str] = None,
    confidence: Optional[float] = None,
):
    """
    Renders token attribution highlights as colored HTML spans.
    Positive attributions (pushing toward Harmful) are highlighted in shades of red.
    Negative attributions (pushing toward Non-Harmful) are highlighted in shades of blue.
    """
    if not tokens or not attributions or len(tokens) != len(attributions):
        st.info("No token attribution data available to display.")
        return

    max_abs = max(abs(a) for a in attributions) if attributions else 1.0
    if max_abs == 0:
        max_abs = 1.0

    html_spans = []
    for token, attr in zip(tokens, attributions):
        norm_score = attr / max_abs
        safe_token = html.escape(token)
        
        if norm_score > 0:
            # Positive contribution to class (e.g. Harmful) -> Red tint
            alpha = min(abs(norm_score) * 0.7 + 0.1, 0.8)
            bg_color = f"rgba(220, 53, 69, {alpha:.2f})"
            color = "#000000"
        elif norm_score < 0:
            # Negative contribution to class -> Blue tint
            alpha = min(abs(norm_score) * 0.7 + 0.1, 0.8)
            bg_color = f"rgba(13, 110, 253, {alpha:.2f})"
            color = "#000000"
        else:
            bg_color = "transparent"
            color = "#495057"

        span = (
            f'<span style="background-color: {bg_color}; color: {color}; '
            f'padding: 2px 5px; margin: 2px; border-radius: 4px; display: inline-block; '
            f'font-family: monospace; font-size: 0.95em;" '
            f'title="Attribution: {attr:+.4f}">{safe_token}</span>'
        )
        html_spans.append(span)

    container_html = f"""
    <div style="padding: 16px; border: 1px solid #dee2e6; border-radius: 8px; background: #ffffff; line-height: 2.0;">
        {' '.join(html_spans)}
    </div>
    <div style="margin-top: 8px; font-size: 0.85em; color: #6c757d; display: flex; justify-content: space-between;">
        <span><span style="color: #0d6efd; font-weight: bold;">■ Blue:</span> Pushes toward Non-Harmful</span>
        <span><span style="color: #dc3545; font-weight: bold;">■ Red:</span> Pushes toward Harmful</span>
    </div>
    """
    st.markdown(container_html, unsafe_allow_html=True)


def render_shap_placeholder_banner():
    """
    Displays the explanation banner when SHAP analysis is not active or in Phase 1.
    """
    st.markdown(
        """
        <div style="background-color: #e9ecef; padding: 12px; border-radius: 6px; border-left: 4px solid #6c757d;">
            <strong>Explainability Pipeline (SHAP):</strong><br>
            <span style="font-size: 0.9em; color: #495057;">
                Token attribution calculates Shapley values using a partition text masker.
                Salient tokens that strongly influence the hate/offensive prediction are highlighted.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
