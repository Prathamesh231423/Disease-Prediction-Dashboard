import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def get_risk_color(probability: float) -> str:
    """Return color based on risk probability thresholds."""
    if probability < 0.4:
        return "green"
    elif probability < 0.7:
        return "orange"
    else:
        return "red"

def create_risk_gauge(probability: float) -> go.Figure:
    """Create a Plotly gauge chart for risk probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': get_risk_color(probability)},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "lightyellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': probability * 100}
        }
    ))
    fig.update_layout(height=300)
    return fig

def get_prediction_verdict(probability: float):
    """Return colored verdict based on probability."""
    if probability < 0.4:
        return st.success(f"Verdict: Low Risk (Probability: {probability:.1%})")
    elif probability < 0.7:
        return st.warning(f"Verdict: Moderate Risk (Probability: {probability:.1%})")
    else:
        return st.error(f"Verdict: High Risk (Probability: {probability:.1%})")
