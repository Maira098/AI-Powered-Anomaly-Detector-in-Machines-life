import json
import os
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from agents import (
    get_lstm_anomaly_agent,
    get_supervisor_explainer_agent,
    lstm_anomaly_tool,
    evaluation_metrics_tool,
)

# =========================
# ENHANCED STREAMLIT CONFIG WITH VIBRANT BACKGROUND
# =========================
st.set_page_config(
    page_title="🤖 AI-Powered Anomaly Detector in Machines",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enhanced CSS with vibrant gradient background and modern effects
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, 
            #0a0a1a 0%, 
            #1a1a2e 25%, 
            #16213e 50%, 
            #0f3460 75%, 
            #1a1a2e 100%);
        background-size: 400% 400%;
        animation: gradientFlow 15s ease infinite;
        color: #f1f5f9;
    }

    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glowing border effect for main container */
    .main > div {
        position: relative;
    }

    .main > div:before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, 
            #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57, #ff6b6b);
        background-size: 400% 400%;
        z-index: -1;
        animation: borderGlow 10s ease infinite;
        border-radius: 12px;
        opacity: 0.7;
        filter: blur(8px);
    }

    @keyframes borderGlow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    /* Enhanced metric cards with glassmorphism */
    .stMetric {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .stMetric:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.1), 
            transparent);
        transition: left 0.7s;
    }

    .stMetric:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2),
            0 0 30px rgba(59, 130, 246, 0.3);
        border-color: rgba(59, 130, 246, 0.4);
    }

    .stMetric:hover:before {
        left: 100%;
    }

    .metric-value {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Enhanced tabs with vibrant colors */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }

    .stTabs [data-baseweb="tab"] {
        height: 60px;
        border-radius: 14px;
        padding: 0 28px;
        background: transparent;
        color: #94a3b8;
        font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        border: 1px solid transparent;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.08);
        color: #60a5fa;
        border-color: rgba(96, 165, 250, 0.3);
        transform: translateY(-2px);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, 
            rgba(255, 107, 107, 0.9), 
            rgba(78, 205, 196, 0.9), 
            rgba(69, 183, 209, 0.9)) !important;
        color: white !important;
        font-weight: 700;
        box-shadow: 
            0 10px 30px rgba(255, 107, 107, 0.3),
            0 0 20px rgba(78, 205, 196, 0.2);
        border: none !important;
        animation: tabPulse 2s infinite;
    }

    @keyframes tabPulse {
        0%, 100% { box-shadow: 0 10px 30px rgba(255, 107, 107, 0.3); }
        50% { box-shadow: 0 10px 40px rgba(255, 107, 107, 0.5); }
    }

    /* Enhanced buttons with gradient glow */
    .stButton button {
        background: linear-gradient(135deg, #ff6b6b, #4ecdc4, #45b7d1);
        color: white;
        border: none;
        padding: 16px 32px;
        border-radius: 16px;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 8px 25px rgba(255, 107, 107, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
        letter-spacing: 0.5px;
    }

    .stButton button:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.2), 
            transparent);
        transition: left 0.7s;
    }

    .stButton button:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 
            0 15px 45px rgba(255, 107, 107, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.3),
            0 0 30px rgba(78, 205, 196, 0.4);
    }

    .stButton button:hover:before {
        left: 100%;
    }

    /* Enhanced card styling */
    .card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.8rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .card:before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
    }

    .card:hover {
        transform: translateY(-5px);
        box-shadow: 
            0 15px 50px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            0 0 30px rgba(78, 205, 196, 0.2);
        border-color: rgba(78, 205, 196, 0.3);
    }

    .divider {
        height: 2px;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(78, 205, 196, 0.5), 
            transparent);
        margin: 2.5rem 0;
        animation: dividerPulse 3s infinite;
    }

    @keyframes dividerPulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }

    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, 
            rgba(26, 26, 46, 0.95), 
            rgba(15, 52, 96, 0.95));
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(78, 205, 196, 0.2);
    }

    /* Enhanced dataframe styling */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        overflow: hidden;
    }

    /* File uploader styling */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(78, 205, 196, 0.3);
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s ease;
    }

    .stFileUploader > div:hover {
        border-color: rgba(78, 205, 196, 0.6);
        background: rgba(255, 255, 255, 0.08);
    }

    /* Text input styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: #f1f5f9;
        padding: 12px;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: rgba(78, 205, 196, 0.5);
        box-shadow: 0 0 20px rgba(78, 205, 196, 0.2);
    }

    /* Slider styling */
    .stSlider > div > div > div {
        background: rgba(78, 205, 196, 0.2);
    }

    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
    }

    /* Download button styling */
    .stDownloadButton button {
        background: linear-gradient(135deg, #45b7d1, #4ecdc4);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stDownloadButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(69, 183, 209, 0.4);
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #ff6b6b, #4ecdc4);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #4ecdc4, #45b7d1);
    }

    /* Alert boxes */
    .stAlert {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }

    /* Success message */
    .stSuccess {
        background: rgba(78, 205, 196, 0.1);
        border-left: 4px solid #4ecdc4;
    }

    /* Error message */
    .stError {
        background: rgba(255, 107, 107, 0.1);
        border-left: 4px solid #ff6b6b;
    }

    /* Info message */
    .stInfo {
        background: rgba(69, 183, 209, 0.1);
        border-left: 4px solid #45b7d1;
    }

    /* Warning message */
    .stWarning {
        background: rgba(254, 202, 87, 0.1);
        border-left: 4px solid #feca57;
    }

    /* Plotly chart container */
    .js-plotly-plot {
        border-radius: 16px;
        overflow: hidden;
        background: rgba(255, 255, 255, 0.03);
        padding: 1rem;
    }

    /* Spinner animation */
    .stSpinner > div {
        border-top-color: #4ecdc4 !important;
    }

    /* Balloons effect enhancement */
    .element-container {
        position: relative;
    }

    /* Enhanced headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Links */
    a {
        color: #4ecdc4;
        text-decoration: none;
        transition: all 0.3s ease;
    }

    a:hover {
        color: #45b7d1;
        text-shadow: 0 0 10px rgba(78, 205, 196, 0.5);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# SESSION STATE INITIALIZATION
# =========================
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "lstm_data" not in st.session_state:
    st.session_state.lstm_data = []
if "metrics_text" not in st.session_state:
    st.session_state.metrics_text = ""
if "anomaly_scores" not in st.session_state:
    st.session_state.anomaly_scores = []
if "labels" not in st.session_state:
    st.session_state.labels = []
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.0
if "machine_id" not in st.session_state:
    st.session_state.machine_id = ""


# =========================
# HELPER FUNCTIONS
# =========================
def load_csv(uploaded_file: Optional[st.runtime.uploaded_file_manager.UploadedFile]) -> pd.DataFrame:
    """Load and return CSV data from uploaded file."""
    if uploaded_file is None:
        return pd.DataFrame()
    return pd.read_csv(uploaded_file)


def compute_threshold_and_labels(
        lstm_df: pd.DataFrame, metrics_text: str
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute threshold and binary labels from LSTM anomaly scores.
    Uses optimal threshold of 0.535 based on F1 score analysis.
    """
    # Optimal threshold from your test_model.py analysis
    DEFAULT_THRESHOLD = 0.8

    if lstm_df.empty or "anomaly_score" not in lstm_df.columns:
        return np.array([]), np.array([]), DEFAULT_THRESHOLD

    anomaly_scores = lstm_df["anomaly_score"].values

    # Try to extract threshold from metrics, otherwise use optimal default
    threshold = DEFAULT_THRESHOLD  # Changed from 0.8 to 0.535
    try:
        metrics_dict = json.loads(metrics_text)
        # If metrics contain a threshold, use it
        if "threshold" in metrics_dict:
            threshold = float(metrics_dict["threshold"])
    except:
        pass

    # Use prediction column if available, otherwise compute from threshold
    if "prediction" in lstm_df.columns:
        labels = lstm_df["prediction"].values.astype(int)
    else:
        labels = (anomaly_scores >= threshold).astype(int)

    return anomaly_scores, labels, threshold


def get_machine_health_metrics(
        anomaly_scores: np.ndarray,
        labels: np.ndarray,
        threshold: float
) -> Dict[str, Any]:
    """
    Calculate comprehensive machine health metrics from anomaly data.
    """
    total = len(anomaly_scores)
    if total == 0:
        return {
            "health_score": 0,
            "anomaly_count": 0,
            "anomaly_percentage": 0.0,
            "health_status": "Unknown",
            "risk_level": "Unknown",
            "score_mean": 0.0,
            "score_std": 0.0,
            "threshold": threshold,
        }

    anomaly_count = int(labels.sum())
    anomaly_percentage = (anomaly_count / total) * 100

    # Health score: 100 - anomaly_percentage (capped at 0-100)
    health_score = max(0, min(100, 100 - anomaly_percentage))

    # Determine health status
    if anomaly_percentage < 3:
        health_status = "Excellent"
        risk_level = "Low"
    elif anomaly_percentage < 8:
        health_status = "Good"
        risk_level = "Low"
    elif anomaly_percentage < 15:
        health_status = "Fair"
        risk_level = "Medium"
    elif anomaly_percentage < 30:
        health_status = "Poor"
        risk_level = "High"
    else:
        health_status = "Critical"
        risk_level = "Critical"

    return {
        "health_score": round(health_score, 1),
        "anomaly_count": anomaly_count,
        "anomaly_percentage": round(anomaly_percentage, 2),
        "health_status": health_status,
        "risk_level": risk_level,
        "score_mean": float(np.mean(anomaly_scores)),
        "score_std": float(np.std(anomaly_scores)),
        "threshold": threshold,
    }


def create_timeline_plot(
        anomaly_scores: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        machine_id: str
) -> go.Figure:
    """
    Create an enhanced timeline plot showing anomaly scores with threshold.
    """
    indices = np.arange(len(anomaly_scores))

    fig = go.Figure()

    # Normal points
    normal_mask = labels == 0
    fig.add_trace(
        go.Scatter(
            x=indices[normal_mask],
            y=anomaly_scores[normal_mask],
            mode="markers",
            name="Normal",
            marker=dict(
                size=8,
                color="#4ecdc4",
                opacity=0.7,
                line=dict(width=1, color="white"),
            ),
            hovertemplate="<b>Index:</b> %{x}<br><b>Score:</b> %{y:.3f}<br><b>Status:</b> Normal<extra></extra>",
        )
    )

    # Anomaly points
    anomaly_mask = labels == 1
    if anomaly_mask.any():
        fig.add_trace(
            go.Scatter(
                x=indices[anomaly_mask],
                y=anomaly_scores[anomaly_mask],
                mode="markers",
                name="Anomaly",
                marker=dict(
                    size=12,
                    color="#ff6b6b",
                    opacity=0.9,
                    symbol="x",
                    line=dict(width=2, color="white"),
                ),
                hovertemplate="<b>Index:</b> %{x}<br><b>Score:</b> %{y:.3f}<br><b>Status:</b> Anomaly<extra></extra>",
            )
        )

    # Threshold line
    fig.add_hline(
        y=threshold,
        line=dict(color="#feca57", width=2, dash="dash"),
        annotation=dict(
            text=f"Threshold: {threshold:.3f}",
            font=dict(size=12, color="#feca57", family="Inter"),
            xanchor="right",
            x=1,
        ),
    )

    fig.update_layout(
        title=dict(
            text=f"🔍 Anomaly Detection Timeline - Machine {machine_id}",
            font=dict(size=24, color="white", family="Space Grotesk"),
            x=0.5,
        ),
        xaxis_title="Sample Index",
        yaxis_title="Anomaly Score",
        template="plotly_dark",
        hovermode="closest",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255, 255, 255, 0.03)",
        font=dict(color="#cbd5e1", family="Inter"),
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.05)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
        ),
        margin=dict(l=60, r=60, t=100, b=60),
    )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(100, 116, 139, 0.1)",
        tickfont=dict(color="#cbd5e1", size=12, family="Inter"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(100, 116, 139, 0.1)",
        tickfont=dict(color="#cbd5e1", size=12, family="Inter"),
    )

    return fig


def create_severity_chart(anomaly_scores: np.ndarray, threshold: float) -> Optional[go.Figure]:
    """
    Create a pie chart showing severity distribution of anomalies.
    """
    # Only consider points above threshold
    anomalous = anomaly_scores[anomaly_scores >= threshold]

    if len(anomalous) == 0:
        return None

    # Categorize by severity
    critical = np.sum(anomaly_scores >= threshold * 1.2)
    high = np.sum((anomaly_scores >= threshold * 1.1) & (anomaly_scores < threshold * 1.2))
    medium = np.sum((anomaly_scores >= threshold) & (anomaly_scores < threshold * 1.1))

    if critical + high + medium == 0:
        return None

    labels_sev = []
    values_sev = []
    colors_sev = []

    if critical > 0:
        labels_sev.append(f"Critical")
        values_sev.append(critical)
        colors_sev.append("#ff6b6b")

    if high > 0:
        labels_sev.append(f"High")
        values_sev.append(high)
        colors_sev.append("#feca57")

    if medium > 0:
        labels_sev.append(f"Medium")
        values_sev.append(medium)
        colors_sev.append("#45b7d1")

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels_sev,
                values=values_sev,
                marker=dict(colors=colors_sev, line=dict(color="white", width=2)),
                textinfo="label+percent",
                textfont=dict(size=14, color="white", family="Inter"),
                hole=0.4,
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="⚠️ Anomaly Severity Distribution",
            font=dict(size=22, color="white", family="Space Grotesk"),
            x=0.5,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255, 255, 255, 0.03)",
        font=dict(color="#cbd5e1", family="Inter"),
        height=450,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            bgcolor="rgba(255, 255, 255, 0.05)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
        ),
        margin=dict(l=40, r=120, t=80, b=40),
    )

    return fig


def create_histogram(anomaly_scores: np.ndarray, threshold: float) -> go.Figure:
    """
    Create a histogram showing distribution of anomaly scores.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=anomaly_scores,
            nbinsx=50,
            marker=dict(
                color="#4ecdc4",
                opacity=0.7,
                line=dict(width=1, color="white"),
            ),
            name="Score Distribution",
            hovertemplate="<b>Score Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>",
        )
    )

    # Add threshold line
    fig.add_vline(
        x=threshold,
        line=dict(color="#ff6b6b", width=3, dash="dash"),
        annotation=dict(
            text=f"Threshold: {threshold:.3f}",
            font=dict(size=12, color="#ff6b6b", family="Inter"),
            yanchor="bottom",
        ),
    )

    fig.update_layout(
        title=dict(
            text="📊 Anomaly Score Distribution",
            font=dict(size=24, color="white", family="Space Grotesk"),
            x=0.5,
        ),
        xaxis_title="Anomaly Score",
        yaxis_title="Frequency",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255, 255, 255, 0.03)",
        font=dict(color="#cbd5e1", family="Inter"),
        height=450,
        showlegend=False,
        margin=dict(l=60, r=60, t=100, b=60),
    )

    fig.update_xaxes(tickfont=dict(color="#cbd5e1", size=12, family="Inter"))
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(100, 116, 139, 0.1)'
    )

    return fig


def create_correlation_heatmap(lstm_df: pd.DataFrame):
    """
    Create correlation heatmap for numeric features.
    """
    if lstm_df is None or lstm_df.empty:
        return None

    numeric_df = lstm_df.select_dtypes(include=[np.number]).copy()

    cols_pref = [
        "anomaly_score",
        "prediction",
        "elapsed",
        "pi",
        "po",
        "speed",
        "efficiency",
        "scrap",
        "scrap_rate",
        "tp_sec",
    ]
    cols_use = [c for c in cols_pref if c in numeric_df.columns]
    if cols_use:
        numeric_df = numeric_df[cols_use]

    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        labels=dict(color="Correlation"),
    )

    fig.update_layout(
        title=dict(
            text="🔗 Feature Correlation Matrix",
            font=dict(size=24, color="white", family="Space Grotesk"),
            x=0.5
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255, 255, 255, 0.03)",
        font=dict(color="#cbd5e1", family="Inter"),
        height=600,
        margin=dict(l=80, r=80, t=120, b=80),
    )

    fig.update_xaxes(
        tickfont=dict(color="#cbd5e1", size=10, family="Inter")
    )
    fig.update_yaxes(
        tickfont=dict(color="#cbd5e1", size=10, family="Inter")
    )

    return fig


def create_recommendations(health_metrics: Dict[str, Any], machine_id: str):
    """
    Generate maintenance recommendations based on health metrics.
    """
    st.markdown("### 🛠️ Maintenance & Action Plan")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='card'>
            <h4 style='color: #4ecdc4; margin-bottom: 1rem;'>⚡ Immediate Actions</h4>
        """, unsafe_allow_html=True)

        risk = health_metrics["risk_level"]
        if risk == "Critical":
            st.error("**🚨 EMERGENCY SHUTDOWN REQUIRED**")
            st.markdown("- ⚠️ **Stop equipment immediately**")
            st.markdown("- 📞 Call emergency maintenance team")
            st.markdown("- 👷 Notify safety officer on-site")
            st.markdown("- 🔧 Isolate affected components")
        elif risk == "High":
            st.warning("**⚠️ URGENT MAINTENANCE NEEDED**")
            st.markdown("- 📅 Schedule maintenance within 24 hours")
            st.markdown("- 👀 Increase monitoring to hourly checks")
            st.markdown("- 🔍 Perform detailed inspection")
            st.markdown("- 📊 Document all anomalies")
        elif risk == "Medium":
            st.info("**🛡️ PREVENTIVE MAINTENANCE**")
            st.markdown("- 📅 Schedule within 1 week")
            st.markdown("- 📈 Monitor daily trends")
            st.markdown("- 🔄 Check related components")
            st.markdown("- 📝 Update maintenance logs")
        else:
            st.success("**✅ NORMAL OPERATION**")
            st.markdown("- 📊 Continue regular monitoring")
            st.markdown("- 📅 Follow standard maintenance schedule")
            st.markdown("- 📋 Review monthly performance reports")
            st.markdown("- 🎯 Focus on optimization")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card'>
            <h4 style='color: #4ecdc4; margin-bottom: 1rem;'>🛡️ Preventive Measures</h4>
        """, unsafe_allow_html=True)
        st.markdown("**🔧 Recommended checks:**")
        st.markdown("- ⚙️ Lubrication system inspection")
        st.markdown("- 🔩 Bearings and alignment verification")
        st.markdown("- 🌡️ Temperature and vibration monitoring")
        st.markdown("- 🔌 Electrical connections check")
        st.markdown("- 🧹 Cleaning and visual inspection")
        st.markdown("- 📏 Calibration verification")
        st.markdown("- 🔄 Wear and tear assessment")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='card'>
            <h4 style='color: #4ecdc4; margin-bottom: 1rem;'>📈 Long-term Improvements</h4>
        """, unsafe_allow_html=True)
        ap = health_metrics["anomaly_percentage"]
        if ap > 15:
            st.markdown("**💡 Major Upgrades Recommended:**")
            st.markdown("- 💰 Budget for major component replacement")
            st.markdown("- 🔄 Consider equipment upgrade timeline")
            st.markdown("- 📋 Review operating procedures")
            st.markdown("- 🎯 Implement predictive maintenance")
            st.markdown("- 📊 Advanced analytics integration")
        elif ap > 8:
            st.markdown("**🔄 Optimization Needed:**")
            st.markdown("- ⚙️ Tune process parameters")
            st.markdown("- 📡 Add extra sensors if needed")
            st.markdown("- 🤖 Improve monitoring system")
            st.markdown("- 📚 Additional training for staff")
            st.markdown("- 🔧 Component life extension")
        else:
            st.markdown("**✅ Good Condition:**")
            st.markdown("- 🏗️ Keep current setup")
            st.markdown("- 💡 Plan small reliability projects")
            st.markdown("- 📈 Keep logging anomalies for future models")
            st.markdown("- 🎯 Focus on efficiency improvements")
            st.markdown("- 🤝 Regular team training")
        st.markdown("</div>", unsafe_allow_html=True)


# =========================
# ENHANCED MAIN APP
# =========================
def main():
    # Header with gradient and updated title
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='font-size: 2.8rem; background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                       margin-bottom: 0.5rem; font-family: "Space Grotesk", sans-serif;'>
                🤖 AI-Powered Anomaly Detector in Machines
            </h1>
            <p style='color: #94a3b8; font-size: 1.1rem; max-width: 800px; margin: 0 auto;'>
                Advanced LSTM-based anomaly detection with intelligent supervisor agent 
                for predictive maintenance insights
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <h2 style='color: #4ecdc4; font-family: "Space Grotesk", sans-serif;'>📊 Data & Configuration</h2>
                <div style='height: 2px; background: linear-gradient(90deg, transparent, #4ecdc4, transparent); 
                          margin: 1rem 0;'></div>
            </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "📁 Upload CSV Data",
            type=["csv"],
            help="Upload your raw_data.csv file with equipment metrics"
        )

        machine_id_input = st.text_input(
            "🏷️ Machine ID",
            "s_3",
            help="Enter the equipment_ID to analyze"
        )

        max_rows = st.slider(
            "📊 Rows to Analyze",
            500, 20000, 5000,
            help="Number of rows to process for analysis"
        )

        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

        run_btn = st.button(
            "🚀 Run LSTM Analysis",
            use_container_width=True,
            type="primary",
            help="Execute LSTM-based anomaly detection with supervisor agent"
        )

    df = load_csv(uploaded)
    if df.empty:
        st.info(
            """
            📤 **Upload a CSV to begin analysis**

            Expected columns include:
            - `equipment_ID` (machine identifier)
            - `interval_start` (timestamp)
            - Process metrics (speed, efficiency, scrap_rate, etc.)
            """
        )
        return

    # Enhanced data preview
    st.markdown("### 📋 Raw Data Preview")
    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True,
        column_config={
            col: st.column_config.Column(width="medium")
            for col in df.columns
        }
    )

    # === RUN ANALYSIS AND STORE IN SESSION STATE ===
    if run_btn:
        if "equipment_ID" not in df.columns:
            st.error("❌ CSV must contain an `equipment_ID` column.")
            return

        df_m = df[df["equipment_ID"] == machine_id_input].copy()
        if df_m.empty:
            st.error(f"❌ No data found for machine `{machine_id_input}`.")
            return

        df_m = df_m.sort_values(df_m.columns[0])
        records = df_m.to_dict(orient="records")[:max_rows]
        data_json = json.dumps(records)

        with st.spinner("🤖 Running LSTM AI analysis..."):
            col1, col2 = st.columns(2)

            with col1:
                st.info("🧠 Running LSTM sequence analysis...")
                lstm_agent = get_lstm_anomaly_agent()
                lstm_json = lstm_anomaly_tool.invoke({"data": data_json})

            with col2:
                st.info("📊 Calculating evaluation metrics...")
                metrics_text = evaluation_metrics_tool.invoke({"lstm_json": lstm_json})

        try:
            lstm_data = json.loads(lstm_json)
        except Exception:
            lstm_data = []

        lstm_df = pd.DataFrame(lstm_data)
        if lstm_df.empty:
            st.error("❌ LSTM analysis returned no data.")
            return

        anomaly_scores, labels, threshold = compute_threshold_and_labels(
            lstm_df, metrics_text
        )

        st.session_state.analysis_done = True
        st.session_state.lstm_data = lstm_data
        st.session_state.metrics_text = metrics_text
        st.session_state.anomaly_scores = anomaly_scores
        st.session_state.labels = labels
        st.session_state.threshold = threshold
        st.session_state.machine_id = machine_id_input

        st.success("✅ Analysis complete! View results in the tabs below.")
        st.balloons()

    if not st.session_state.analysis_done:
        st.info(
            """
            👈 **Click 'Run LSTM Analysis' in the sidebar** 

            This will execute:
            1. LSTM-based anomaly detection
            2. Performance evaluation metrics
            3. Supervisor agent analysis
            4. Generate maintenance recommendations
            """
        )
        return

    # Load from session state
    lstm_data = st.session_state.lstm_data
    metrics_text = st.session_state.metrics_text
    anomaly_scores = st.session_state.anomaly_scores
    labels = st.session_state.labels
    threshold = st.session_state.threshold
    machine_id = st.session_state.machine_id

    lstm_df = pd.DataFrame(lstm_data)
    health_metrics = get_machine_health_metrics(anomaly_scores, labels, threshold)

    # build figures once
    timeline_fig = create_timeline_plot(anomaly_scores, labels, threshold, machine_id)
    sev_fig = create_severity_chart(anomaly_scores, threshold)
    hist_fig = create_histogram(anomaly_scores, threshold)
    corr_fig = create_correlation_heatmap(lstm_df)

    try:
        metrics_dict = json.loads(metrics_text)
    except Exception:
        metrics_dict = {}

    # Enhanced tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Machine Health Dashboard",
        "📈 Analytics & Recommendations",
        "🧠 AI Supervisor Insights",
        "📥 Export Results"
    ])

    # TAB 1: Dashboard
    with tab1:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h2 style='color: #4ecdc4; font-family: "Space Grotesk", sans-serif;'>📊 Machine Health Dashboard</h2>
                <p style='color: #94a3b8;'>Real-time anomaly detection and machine health monitoring</p>
            </div>
        """, unsafe_allow_html=True)

        # Health metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                label="🎯 Health Score",
                value=f"{health_metrics['health_score']}/100",
                delta=None if health_metrics['health_score'] >= 80 else f"{health_metrics['health_score'] - 80:.1f}",
                delta_color="normal",
                help="Overall machine health score (0-100)"
            )

        with col2:
            status_color = {
                "Excellent": "🟢",
                "Good": "🟡",
                "Fair": "🟠",
                "Poor": "🔴",
                "Critical": "🚨"
            }.get(health_metrics['health_status'], "⚪")
            st.metric(
                label=f"{status_color} Status",
                value=health_metrics['health_status'],
                help="Current operational status"
            )

        with col3:
            risk_color = {
                "Low": "🟢",
                "Medium": "🟡",
                "High": "🔴",
                "Critical": "🚨"
            }.get(health_metrics['risk_level'], "⚪")
            st.metric(
                label=f"{risk_color} Risk Level",
                value=health_metrics['risk_level'],
                help="Risk assessment for equipment failure"
            )

        with col4:
            st.metric(
                label="⚠️ Anomalies",
                value=f"{health_metrics['anomaly_count']}",
                delta=f"{health_metrics['anomaly_percentage']:.1f}%",
                delta_color="inverse",
                help="Total anomaly count and percentage"
            )

        with col5:
            st.metric(
                label="📊 Avg Score",
                value=f"{health_metrics['score_mean']:.3f}",
                delta=f"±{health_metrics['score_std']:.3f} std",
                delta_color="off",
                help="Average anomaly score with standard deviation"
            )

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        st.markdown("### 📈 Anomaly Visualization")
        tcol, scol = st.columns([2, 1])
        with tcol:
            st.plotly_chart(timeline_fig, use_container_width=True, key="tl_dashboard")
        with scol:
            if sev_fig:
                st.plotly_chart(sev_fig, use_container_width=True, key="sev_dashboard")
            else:
                st.info("✅ No anomalous points above threshold detected.")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        st.markdown("### 🔗 Feature Correlation Analysis")
        if corr_fig:
            st.plotly_chart(corr_fig, use_container_width=True, key="corr_dashboard")
        else:
            st.info("Not enough numeric features to build correlation matrix.")

    # TAB 2: Enhanced distributions
    with tab2:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h2 style='color: #4ecdc4; font-family: "Space Grotesk", sans-serif;'>📈 Analytics & Recommendations</h2>
                <p style='color: #94a3b8;'>Detailed analysis and maintenance planning</p>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(hist_fig, use_container_width=True, key="hist_distribution")
            st.markdown("""
                <div class='card'>
                    <h4>📝 Distribution Insights</h4>
                    <p style='color: #94a3b8;'>
                        The histogram shows the frequency distribution of LSTM anomaly scores. 
                        Scores crossing the red threshold line indicate potential issues 
                        requiring attention.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            if sev_fig:
                st.plotly_chart(sev_fig, use_container_width=True, key="sev_distribution")
                st.markdown("""
                    <div class='card'>
                        <h4>⚠️ Severity Classification</h4>
                        <p style='color: #94a3b8;'>
                            Anomalies are classified by severity based on their score 
                            relative to the threshold. Critical anomalies require 
                            immediate attention.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("✅ No severe anomalies detected.")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        if health_metrics:
            create_recommendations(health_metrics, machine_id)

    # TAB 3: Enhanced supervisor agent
    with tab3:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h2 style='color: #4ecdc4; font-family: "Space Grotesk", sans-serif;'>🧠 AI Supervisor Insights</h2>
                <p style='color: #94a3b8;'>Intelligent analysis and expert recommendations</p>
            </div>
        """, unsafe_allow_html=True)

        lstm_preview_str = json.dumps(lstm_data[:50])

        sup_question = st.text_area(
            "💬 Supervisor Question",
            value=(
                f"Analyze anomalies for machine {machine_id} and provide:\n"
                "1. Severity assessment of detected anomalies\n"
                "2. Potential root causes (technical perspective)\n"
                "3. Practical maintenance recommendations\n"
                "4. Operational adjustments if needed\n\n"
                "Please structure the answer for maintenance engineers."
            ),
            height=180,
            key="sup_question",
            help="Customize the question for the AI supervisor"
        )

        sup_input = (
            "**ANOMALY ANALYSIS REQUEST**\n\n"
            f"Machine ID: {machine_id}\n"
            f"Health Status: {health_metrics['health_status']}\n"
            f"Anomaly Rate: {health_metrics['anomaly_percentage']:.2f}%\n\n"
            f"LSTM Results Preview (first 50 entries):\n{lstm_preview_str[:1000]}...\n\n"
            f"Evaluation Metrics:\n{metrics_text[:1500]}...\n\n"
            f"**QUESTION:**\n{sup_question}"
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            ask_btn = st.button(
                "🤖 Ask AI Supervisor",
                use_container_width=True,
                key="ask_supervisor_btn",
                help="Get AI-powered insights and recommendations"
            )

        if ask_btn:
            with st.spinner("🧠 AI Supervisor analyzing data..."):
                sup_out = get_supervisor_explainer_agent().invoke({"input": sup_input})

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            st.markdown("### 💡 Supervisor Response")
            st.markdown("""
                <div class='card' style='background: rgba(26, 26, 46, 0.9);'>
            """, unsafe_allow_html=True)
            st.markdown(sup_out)
            st.markdown("</div>", unsafe_allow_html=True)

    # TAB 4: Enhanced export
    with tab4:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h2 style='color: #4ecdc4; font-family: "Space Grotesk", sans-serif;'>📥 Export & Reports</h2>
                <p style='color: #94a3b8;'>Download analysis results and reports</p>
            </div>
        """, unsafe_allow_html=True)

        export_df = lstm_df.copy()
        export_df["is_anomaly"] = labels
        export_df["anomaly_score"] = anomaly_scores
        export_df["machine_id"] = machine_id
        export_df["analysis_timestamp"] = pd.Timestamp.now()
        export_df["health_status"] = health_metrics["health_status"]
        export_df["risk_level"] = health_metrics["risk_level"]

        col1, col2, col3 = st.columns(3)

        with col1:
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📊 Download CSV Report",
                data=csv_bytes,
                file_name=f"{machine_id}_anomaly_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download full analysis results as CSV",
                use_container_width=True
            )

        with col2:
            json_bytes = export_df.to_json(orient="records", indent=2).encode("utf-8")
            st.download_button(
                label="📝 Download JSON Data",
                data=json_bytes,
                file_name=f"{machine_id}_anomaly_data.json",
                mime="application/json",
                help="Download analysis results in JSON format",
                use_container_width=True
            )

        with col3:
            st.download_button(
                label="📋 Download Summary Report",
                data=f"""
                Machine Anomaly Analysis Report
                ================================

                Machine ID: {machine_id}
                Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                Health Status: {health_metrics['health_status']}
                Risk Level: {health_metrics['risk_level']}

                Summary Metrics:
                - Health Score: {health_metrics['health_score']}/100
                - Anomaly Rate: {health_metrics['anomaly_percentage']:.2f}%
                - Total Anomalies: {health_metrics['anomaly_count']}
                - Detection Threshold: {health_metrics['threshold']:.3f}

                Model Information:
                - Detection Method: LSTM Neural Network
                - Features Used: 13 engineered features
                - Sequence-based deep learning approach

                Recommendations:
                Please refer to the AI Supervisor insights for detailed recommendations.

                Generated by AI-Powered Anomaly Detector (LSTM-based)
                """.encode("utf-8"),
                file_name=f"{machine_id}_summary_report.txt",
                mime="text/plain",
                help="Download summary report",
                use_container_width=True
            )

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        st.markdown("### 📋 Data Preview")
        st.dataframe(
            export_df.head(50),
            use_container_width=True,
            hide_index=True,
            column_config={
                "is_anomaly": st.column_config.Column(
                    "🚨 Anomaly",
                    help="1 = Anomaly detected, 0 = Normal"
                ),
                "anomaly_score": st.column_config.ProgressColumn(
                    "📊 Score",
                    help="Anomaly score (higher = more anomalous)",
                    min_value=0,
                    max_value=float(export_df["anomaly_score"].max() * 1.1)
                )
            }
        )


if __name__ == "__main__":
    main()