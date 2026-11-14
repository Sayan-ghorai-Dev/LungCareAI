import streamlit as st
import pandas as pd
import os

# Page Configuration
st.set_page_config(
    page_title="Home - LungCareAI Dashboard",
    page_icon="🫁",
    layout="wide"
)

# =====================================================================
# AUTHENTICATION CHECK (PROTECT PAGE & ADD SIDEBAR LOGOUT)
# =====================================================================
if not st.session_state.get("logged_in", False):
    st.markdown("<style>[data-testid='stSidebar'] {display: none;}</style>", unsafe_allow_html=True)
    st.warning("⚠️ Access Denied! Please Sign In from the main page first.")
    if st.button("🔐 Go to Login Page"):
        st.switch_page("app.py")
    st.stop()

with st.sidebar:
    st.markdown(f"""
    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 10px 14px; margin-bottom: 10px; font-size: 13px; color: #1E293B;">
        👤 <b>User:</b> {st.session_state.get('name', 'User')}<br>
        <span style="color: #64748B; font-size: 11px;">@{st.session_state.get('username', '')}</span>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚪 Logout", key="home_logout_btn", use_container_width=True):
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.session_state["name"] = ""
        st.switch_page("app.py")
# =====================================================================

# Load Admin Panel & Custom CSS
def load_css(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("assets/style.css")



# Top Header Banner
st.markdown("""
<div class="navbar-container">
    <div class="nav-brand">
        <span>🫁</span>
        <span>LungCare AI — Clinical Overview</span>
    </div>
    <div class="nav-menu">
        <span style="color: #0284c7; font-weight: 700;">● System Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Welcome & Overview Intro
st.markdown("""
<div style="margin-bottom: 28px;">
    <h1 style="font-size: 36px; font-weight: 800; color: #0f172a; margin-bottom: 8px;">
        Dataset & Risk Factor Insights
    </h1>
    <p style="font-size: 15px; color: #475569;">
        Explore key clinical statistics, symptom correlations, and dataset metrics powering the LungCare AI diagnostic engine.
    </p>
</div>
""", unsafe_allow_html=True)

# Load Dataset Dynamically (if exists)
data_path = "data/lung_cancer.csv"
total_records = "309"
features_count = "15"
positive_rate = "87.4%"

if os.path.exists(data_path):
    try:
        df = pd.read_csv(data_path)
        total_records = f"{len(df):,}"
        features_count = f"{df.shape[1] - 1}"
        target_col = df.columns[-1]
        pos_count = df[target_col].astype(str).str.upper().isin(['YES', '1', 'TRUE']).sum()
        positive_rate = f"{(pos_count / len(df)) * 100:.1f}%"
    except Exception:
        pass

# 4 Summary Metric Cards
st.markdown(f"""
<div class="metrics-grid" style="margin-top: 10px; margin-bottom: 35px;">
    <div class="metric-card">
        <div class="metric-icon-circle">📁</div>
        <div class="metric-val">{total_records}</div>
        <div class="metric-lbl">Total Patient Records</div>
    </div>
    <div class="metric-card">
        <div class="metric-icon-circle">🧬</div>
        <div class="metric-val">{features_count}</div>
        <div class="metric-lbl">Clinical Biomarkers</div>
    </div>
    <div class="metric-card">
        <div class="metric-icon-circle">⚠️</div>
        <div class="metric-val">{positive_rate}</div>
        <div class="metric-lbl">High-Risk Prevalence</div>
    </div>
    <div class="metric-card">
        <div class="metric-icon-circle">🤖</div>
        <div class="metric-val">3 Models</div>
        <div class="metric-lbl">Ensemble Classifiers</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Key Risk Factors & Early Detection Benefits
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="step-card" style="text-align: left; padding: 28px;">
        <h3 style="font-size: 20px; font-weight: 800; color: #0f172a; margin-bottom: 14px;">
            🔍 Primary Clinical Risk Indicators
        </h3>
        <ul style="color: #475569; font-size: 14px; line-height: 1.9; padding-left: 18px;">
            <li><b>Behavioral Factors:</b> Chronic smoking history, alcohol consumption, and peer exposure.</li>
            <li><b>Respiratory Symptoms:</b> Persistent coughing, shortness of breath, wheezing, and chest pain.</li>
            <li><b>Physical & Systemic Signs:</b> Yellowing of fingers, chronic fatigue, allergy, and swallowing difficulty.</li>
            <li><b>Demographic Baseline:</b> Age progression and biological gender risk distribution.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="step-card" style="text-align: left; padding: 28px;">
        <h3 style="font-size: 20px; font-weight: 800; color: #0f172a; margin-bottom: 14px;">
            🚀 Benefits of AI Early Screening
        </h3>
        <ul style="color: #475569; font-size: 14px; line-height: 1.9; padding-left: 18px;">
            <li><b>Rapid Triage:</b> Instant multi-symptom evaluation in under 2 seconds.</li>
            <li><b>High Sensitivity:</b> Minimizes false negatives using tuned gradient boosting (LightGBM & XGBoost).</li>
            <li><b>Preventive Action:</b> Identifies high-risk individuals before severe stage progression.</li>
            <li><b>Data-Driven Clarity:</b> Translates complex symptom combinations into clear probability scores.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# Quick Action Button to Prediction Page
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
with btn_col2:
    if st.button("🩺 Start Patient Risk Assessment →", use_container_width=True):
        st.switch_page("pages/2_Prediction.py")
