import streamlit as st
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="About Project | LungCare AI",
    page_icon="📘",
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
    if st.button("🚪 Logout", key="about_logout_btn", use_container_width=True):
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.session_state["name"] = ""
        st.switch_page("app.py")
# =====================================================================

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"

# Load External Admin Panel CSS
css_file = ASSETS_DIR / "style.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Top Navigation Banner
st.markdown("""
<div class="navbar-container">
    <div class="nav-brand">
        <span>📘</span>
        <span>LungCare AI — System Architecture</span>
    </div>
    <div class="nav-menu">
        <span style="color: #0284c7; font-weight: 700;">● Documentation</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Page Title & Subtitle
st.markdown("""
<div style="margin-bottom: 30px;">
    <h1 style="font-size: 34px; font-weight: 800; color: #0f172a; margin-bottom: 8px;">
        About LungCare AI
    </h1>
    <p style="font-size: 15px; color: #475569; max-width: 820px; line-height: 1.6;">
        An end-to-end machine learning diagnostic support prototype designed to evaluate early lung cancer susceptibility using structured clinical survey biomarkers and ensemble gradient boosting algorithms.
    </p>
</div>
""", unsafe_allow_html=True)

# 3 Core Pillars Cards
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown("""
    <div class="step-card" style="text-align: left; padding: 24px; height: 100%;">
        <div class="step-icon">🎯</div>
        <div class="step-title">Project Objective</div>
        <div class="step-desc" style="font-size: 14px;">
            Provide rapid, accessible preliminary risk stratification based on behavioral and symptomatic indicators to encourage timely medical screening.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="step-card" style="text-align: left; padding: 24px; height: 100%;">
        <div class="step-icon">🧠</div>
        <div class="step-title">ML Pipeline</div>
        <div class="step-desc" style="font-size: 14px;">
            Integrates automated label encoding, feature scaling, and comparative evaluation across Random Forest, XGBoost, and LightGBM classifiers.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="step-card" style="text-align: left; padding: 24px; height: 100%;">
        <div class="step-icon">⚡</div>
        <div class="step-title">Modular Design</div>
        <div class="step-desc" style="font-size: 14px;">
            Decoupled preprocessing, training, evaluation, and inference modules built for maintainability and real-time Streamlit deployment.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.divider()

# Detailed Workflow & Tech Stack Section
sec_col1, sec_col2 = st.columns([1.1, 0.9], gap="large")

with sec_col1:
    st.markdown("""
    <div class="step-card" style="text-align: left; padding: 28px;">
        <h3 style="font-size: 20px; font-weight: 800; color: #0f172a; margin-bottom: 16px;">
            🔄 End-to-End Methodology
        </h3>
        <div style="font-size: 14px; color: #475569; line-height: 1.8;">
            <p style="margin-bottom: 10px;"><b>1. Data Preprocessing:</b> Cleaning survey records, encoding categorical gender labels (<code>label_encoder.pkl</code>), and normalizing numerical ranges (<code>scaler.pkl</code>).</p>
            <p style="margin-bottom: 10px;"><b>2. Feature Engineering:</b> Evaluating 15 key clinical attributes including age, smoking history, chronic respiratory symptoms, and physical signs.</p>
            <p style="margin-bottom: 10px;"><b>3. Model Training & Serialization:</b> Training tree-based ensemble models and serializing optimal weights into <code>.pkl</code> artifacts for low-latency inference.</p>
            <p style="margin-bottom: 0;"><b>4. Interactive Analytics:</b> Visualizing accuracy, precision, recall, ROC-AUC, and confusion matrices inside a unified dashboard.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with sec_col2:
    st.markdown("""
    <div class="step-card" style="text-align: left; padding: 28px;">
        <h3 style="font-size: 20px; font-weight: 800; color: #0f172a; margin-bottom: 16px;">
            🛠️ Technology Stack
        </h3>
        <div style="display: flex; flex-direction: column; gap: 12px; font-size: 14px;">
            <div style="background: #F8FAFC; padding: 10px 14px; border-radius: 10px; border: 1px solid #E2E8F0;">
                <b style="color: #0284c7;">Frontend & UI:</b> Streamlit, Custom CSS3, Responsive Grid
            </div>
            <div style="background: #F8FAFC; padding: 10px 14px; border-radius: 10px; border: 1px solid #E2E8F0;">
                <b style="color: #0284c7;">Machine Learning:</b> Scikit-Learn, LightGBM, XGBoost
            </div>
            <div style="background: #F8FAFC; padding: 10px 14px; border-radius: 10px; border: 1px solid #E2E8F0;">
                <b style="color: #0284c7;">Data Processing:</b> Pandas, NumPy, Joblib
            </div>
            <div style="background: #F8FAFC; padding: 10px 14px; border-radius: 10px; border: 1px solid #E2E8F0;">
                <b style="color: #0284c7;">Visualization:</b> Matplotlib, Seaborn, Streamlit Charts
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)