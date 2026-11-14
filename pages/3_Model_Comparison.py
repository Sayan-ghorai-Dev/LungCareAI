import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Model Comparison | LungCare AI",
    page_icon="📊",
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
    if st.button("🚪 Logout", key="comp_logout_btn", use_container_width=True):
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.session_state["name"] = ""
        st.switch_page("app.py")
# =====================================================================

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
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
        <span>📊</span>
        <span>LungCare AI — Benchmark Analytics</span>
    </div>
    <div class="nav-menu">
        <span style="color: #0284c7; font-weight: 700;">● Evaluation Reports</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom: 24px;">
    <h1 style="font-size: 34px; font-weight: 800; color: #0f172a; margin-bottom: 6px;">
        Model Performance Comparison
    </h1>
    <p style="font-size: 14px; color: #64748b;">
        Comparative evaluation metrics, confusion matrices, and classification benchmarks across Random Forest, XGBoost, and LightGBM.
    </p>
</div>
""", unsafe_allow_html=True)

# 1. Load Model Comparison CSV
csv_path = REPORTS_DIR / "model_comparison.csv"

if csv_path.exists():
    df_comp = pd.read_csv(csv_path)
    
    # Top Highlight Metric Cards (Best Model Summary)
    try:
        # Find numeric accuracy/f1 column dynamically
        acc_col = [c for c in df_comp.columns if 'acc' in c.lower()][0]
        model_col = df_comp.columns[0]
        best_row = df_comp.loc[df_comp[acc_col].idxmax()]
        
        best_model_name = best_row[model_col]
        best_acc_val = float(best_row[acc_col])
        if best_acc_val <= 1.0:
            best_acc_val *= 100

        st.markdown(f"""
        <div class="metrics-grid" style="grid-template-columns: repeat(3, 1fr); margin-top: 10px; margin-bottom: 30px;">
            <div class="metric-card">
                <div class="metric-icon-circle">🏆</div>
                <div class="metric-val">{best_model_name}</div>
                <div class="metric-lbl">Top Performing Classifier</div>
            </div>
            <div class="metric-card">
                <div class="metric-icon-circle">🎯</div>
                <div class="metric-val">{best_acc_val:.1f}%</div>
                <div class="metric-lbl">Peak Validation Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-icon-circle">⚖️</div>
                <div class="metric-val">{len(df_comp)} Models</div>
                <div class="metric-lbl">Evaluated Algorithms</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        pass

    st.markdown("### 📋 Comparative Metrics Table")
    st.markdown(df_comp.to_html(index=False), unsafe_allow_html=True)
    
    # Interactive Bar Chart (Using Matplotlib to avoid pyarrow DLL issues)
    numeric_cols = df_comp.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if len(numeric_cols) > 0:
        st.write("")
        st.markdown("### 📈 Interactive Metric Breakdown")
        chart_df = df_comp.set_index(df_comp.columns[0])[numeric_cols]
        fig, ax = plt.subplots(figsize=(8, 3.8))
        chart_df.plot(kind="bar", ax=ax)
        plt.xticks(rotation=0)
        plt.ylabel("Score")
        plt.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)

else:
    st.warning("`reports/model_comparison.csv` not found. Run `train_model.py` to generate benchmark metrics.")

st.divider()

# 2. Evaluation Visual Charts (PNGs from reports/)
st.markdown("### 🖼️ Visual Evaluation Plots")
img_col1, img_col2 = st.columns(2, gap="large")

comp_img_path = REPORTS_DIR / "model_comparison.png"
cm_img_path = REPORTS_DIR / "confusion_matrix.png"

with img_col1:
    st.markdown("""
    <div class="step-card" style="padding: 16px; margin-bottom: 12px;">
        <div class="step-title" style="font-size: 16px;">Model Accuracy & F1 Comparison</div>
    </div>
    """, unsafe_allow_html=True)
    if comp_img_path.exists():
        st.image(str(comp_img_path), use_container_width=True)
    else:
        st.info("`model_comparison.png` chart will appear here after training.")

with img_col2:
    st.markdown("""
    <div class="step-card" style="padding: 16px; margin-bottom: 12px;">
        <div class="step-title" style="font-size: 16px;">Confusion Matrix Heatmap</div>
    </div>
    """, unsafe_allow_html=True)
    if cm_img_path.exists():
        st.image(str(cm_img_path), use_container_width=True)
    else:
        st.info("`confusion_matrix.png` heatmap will appear here after training.")

st.divider()

# 3. Detailed Classification Report (TXT)
report_txt_path = REPORTS_DIR / "classification_report.txt"
st.markdown("### 📄 Detailed Classification Report")

if report_txt_path.exists():
    with open(report_txt_path, "r", encoding="utf-8") as f:
        report_content = f.read()
    with st.expander("View Full Precision, Recall & F1-Score Matrix", expanded=True):
        st.code(report_content, language="text")
else:
    st.info("`classification_report.txt` not found in `reports/` directory.")