import streamlit as st
from pathlib import Path
from src.auth_db import create_user_table, add_userdata, login_user

# Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="LungCareAI - Early Lung Cancer Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Database Table
create_user_table()

# Initialize Session State for Authentication
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "name" not in st.session_state:
    st.session_state["name"] = ""

# Load External CSS safely using Path & UTF-8 encoding
BASE_DIR = Path(__file__).resolve().parent
css_path = BASE_DIR / "assets" / "style.css"

if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# =====================================================================
# 1. AUTHENTICATION SCREEN (IF USER IS NOT LOGGED IN)
# =====================================================================
if not st.session_state["logged_in"]:
    # Hide sidebar before login so users cannot access other pages
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {display: none;}
            .auth-header {
                text-align: center;
                padding: 20px 0 10px 0;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="auth-header">
            <h1>🫁 LungCare AI</h1>
            <p style="color: #64748b; font-size: 16px;">Please Sign In or Create an Account to access the Early Lung Cancer Detection Platform</p>
        </div>
    """, unsafe_allow_html=True)

    # Center the Login/Register box
    _, center_col, _ = st.columns([1, 1.5, 1])

    with center_col:
        tab1, tab2 = st.tabs(["🔐 Sign In", "📝 Register"])

        # --- SIGN IN TAB ---
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit_login = st.form_submit_button("Sign In", use_container_width=True)

                if submit_login:
                    if not username or not password:
                        st.warning("Please enter both username and password.")
                    else:
                        result = login_user(username, password)
                        if result:
                            st.session_state["logged_in"] = True
                            st.session_state["username"] = result[0][0]
                            st.session_state["name"] = result[0][1]
                            st.success(f"Welcome back, {result[0][1]}!")
                            st.rerun()
                        else:
                            st.error("Invalid Username or Password.")

        # --- REGISTER TAB ---
        with tab2:
            with st.form("register_form"):
                new_name = st.text_input("Full Name", placeholder="e.g. Sayan Ghorai")
                new_user = st.text_input("Choose Username", placeholder="Create a unique username")
                new_pass = st.text_input("Create Password", type="password", placeholder="Create a strong password")
                confirm_pass = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")
                submit_reg = st.form_submit_button("Create Account", use_container_width=True)

                if submit_reg:
                    if not new_name or not new_user or not new_pass:
                        st.warning("Please fill in all the fields.")
                    elif new_pass != confirm_pass:
                        st.error("Passwords do not match!")
                    else:
                        if add_userdata(new_user, new_name, new_pass):
                            st.success("Registration successful! Please switch to the 'Sign In' tab to login.")
                        else:
                            st.error("Username already exists. Please try a different username.")

# =====================================================================
# 2. MAIN WEBSITE FRONTEND (IF USER IS LOGGED IN)
# =====================================================================
else:
    # Custom CSS for the Interactive Navigation Bar
    st.markdown("""
    <style>
    /* Navbar Container Styling */
    div[data-testid="stHorizontalBlock"]:has(.nav-brand) {
        background-color: #0e1117;
        padding: 10px 20px;
        border-radius: 12px;
        border: 1px solid #262730;
        align-items: center;
        margin-bottom: 25px;
    }

    /* Brand Logo and Title */
    .nav-brand {
        font-size: 20px;
        font-weight: 700;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Navbar Buttons Styling */
    div[data-testid="stHorizontalBlock"]:has(.nav-brand) .stButton > button {
        background-color: transparent;
        color: #e0e0e0;
        border: none;
        font-size: 15px;
        font-weight: 500;
        padding: 6px 10px;
        transition: all 0.2s ease;
        width: 100%;
    }

    /* Navbar Button Hover Effect */
    div[data-testid="stHorizontalBlock"]:has(.nav-brand) .stButton > button:hover {
        background-color: #262730;
        color: #38bdf8;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar User Profile, Logout Button & System Status Badge
    with st.sidebar:
        st.markdown(f"""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 10px 14px; margin-bottom: 10px; font-size: 13px; color: #1E293B;">
            👤 <b>User:</b> {st.session_state['name']}<br>
            <span style="color: #64748B; font-size: 11px;">@{st.session_state['username']}</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
            st.session_state["logged_in"] = False
            st.session_state["username"] = ""
            st.session_state["name"] = ""
            st.rerun()

        st.markdown("""
        <div style="background: #F0F9FF; border: 1px solid #BAE6FD; border-radius: 12px; padding: 12px 14px; margin-top: 12px; font-size: 12px; color: #0369A1; line-height: 1.6;">
            <b>● System Status:</b> Online<br>
            <b>Active Engine:</b> LightGBM / XGBoost
        </div>
        """, unsafe_allow_html=True)

    # Top Navigation Bar with Working Buttons
    col_brand, col_space, c_home, c_about, c_works, c_acc = st.columns([2.2, 1.2, 1, 1.1, 1.3, 1])

    with col_brand:
        st.markdown("""
        <div class="nav-brand">
            <span>🫁</span>
            <span>LungCare AI</span>
        </div>
        """, unsafe_allow_html=True)

    with c_home:
        if st.button("Home", key="nav_home"):
            st.switch_page("app.py")

    with c_about:
        if st.button("About Project", key="nav_about"):
            try:
                st.switch_page("pages/4_About_Project.py")
            except Exception:
                st.toast("Select 'About Project' from the sidebar.")

    with c_works:
        if st.button("How It Works", key="nav_works"):
            try:
                st.switch_page("pages/2_Prediction.py")
            except Exception:
                st.toast("Select 'Prediction' from the sidebar.")

    with c_acc:
        if st.button("Accuracy", key="nav_acc"):
            try:
                st.switch_page("pages/3_Model_Comparison.py")
            except Exception:
                st.toast("Select 'Model Comparison' from the sidebar.")

    # Hero Section
    hero_col1, hero_col2 = st.columns([1.2, 0.8], gap="large")

    with hero_col1:
        st.markdown("""
        <div class="hero-wrapper">
            <h1 class="hero-title">Early Lung Cancer Detection, Redefined by AI.</h1>
            <p class="hero-subtitle">
                Leverage advanced Artificial Intelligence for fast, accurate risk analysis and early clinical intervention. 
                Prioritize your respiratory health with cutting-edge machine learning diagnostics.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation to Prediction Page
        if st.button("📋 Take Free Risk Assessment", key="hero_assessment_btn"):
            try:
                st.switch_page("pages/2_Prediction.py")
            except Exception:
                st.info("Select 'Prediction' from the left admin menu to start.")

    with hero_col2:
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; height: 100%; min-height: 280px;">
            <div style="
                width: 220px; 
                height: 220px; 
                border-radius: 50%; 
                background: radial-gradient(circle, rgba(2,132,199,0.18) 0%, rgba(255,255,255,0) 70%); 
                display: flex; 
                align-items: center; 
                justify-content: center; 
                font-size: 110px;
                box-shadow: 0 0 50px rgba(56, 189, 248, 0.25);
            ">
                🫁
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Performance Metric Cards
    st.markdown("""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-icon-circle">🎖️</div>
            <div class="metric-val">95.5%</div>
            <div class="metric-lbl">LightGBM Accuracy</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon-circle">✅</div>
            <div class="metric-val">95.0%</div>
            <div class="metric-lbl">Precision Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon-circle">📈</div>
            <div class="metric-val">95.0%</div>
            <div class="metric-lbl">Recall (Sensitivity)</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon-circle">🎯</div>
            <div class="metric-val">0.986</div>
            <div class="metric-lbl">ROC-AUC Score</div>
        </div>
    </div>
    <div class="metrics-footnote">
        Values validated by comparative model benchmarks & clinical cross-validation matrices.
    </div>
    """, unsafe_allow_html=True)

    st.write("---")

    # How Our AI Works Section
    st.markdown("""
    <div style="margin-top: 30px;">
        <h2 class="section-headline">How Our AI Works</h2>
        <p class="section-subheadline">A structured pipeline converting user health metrics into actionable predictive insights.</p>
    </div>
    """, unsafe_allow_html=True)

    step_col1, step_col2, step_col3 = st.columns(3, gap="medium")

    with step_col1:
        st.markdown("""
        <div class="step-card">
            <div class="step-icon">🗄️</div>
            <div class="step-title">Data Ingestion</div>
            <div class="step-desc">Captures patient demographics, lifestyle indicators, and respiratory symptoms securely.</div>
        </div>
        """, unsafe_allow_html=True)

    with step_col2:
        st.markdown("""
        <div class="step-card">
            <div class="step-icon">⚙️</div>
            <div class="step-title">ML Analysis</div>
            <div class="step-desc">Evaluates symptoms against an ensemble of LightGBM, Random Forest, and XGBoost models.</div>
        </div>
        """, unsafe_allow_html=True)

    with step_col3:
        st.markdown("""
        <div class="step-card">
            <div class="step-icon">📊</div>
            <div class="step-title">Instant Results</div>
            <div class="step-desc">Delivers precise risk probability breakdowns with personalized clinical recommendations.</div>
        </div>
        """, unsafe_allow_html=True)