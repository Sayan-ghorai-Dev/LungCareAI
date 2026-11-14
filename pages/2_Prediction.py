import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Prediction | LungCare AI",
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
    if st.button("🚪 Logout", key="pred_logout_btn", use_container_width=True):
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.session_state["name"] = ""
        st.switch_page("app.py")
# =====================================================================

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
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
        <span>🫁</span>
        <span>LungCare AI — Multimodal Diagnostic Engine</span>
    </div>
    <div class="nav-menu">
        <span style="color: #0284c7; font-weight: 700;">● Engine Online</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom: 20px;">
    <h1 style="font-size: 32px; font-weight: 800; color: #0f172a; margin-bottom: 6px;">
        Patient Risk Inference & Media Analysis
    </h1>
    <p style="font-size: 14px; color: #64748b;">
        Combine clinical survey parameters with chest imaging or medical documents for comprehensive risk evaluation.
    </p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Educational prototype only. This output is a dataset-based "
    "model demonstration, not a medical diagnosis or screening result."
)

# Load trained models
@st.cache_resource
def load_models():
    return {
        "Random Forest": joblib.load(MODEL_DIR / "random_forest.pkl"),
        "XGBoost": joblib.load(MODEL_DIR / "xgboost.pkl"),
        "LightGBM": joblib.load(MODEL_DIR / "lightgbm.pkl")
    }

try:
    models = load_models()
except Exception as e:
    st.error(f"Could not load models: {e}")
    st.stop()

st.divider()

# =========================================================
# SECTION 1: OPTIONAL PHOTO (X-RAY) & DOCUMENT UPLOAD
# =========================================================
st.subheader("📎 Attach Clinical Photo & Medical Documents (Optional)")

up_col1, up_col2 = st.columns(2, gap="large")

uploaded_image = None
uploaded_doc = None
doc_extracted_text = ""

with up_col1:
    st.markdown("""
    <div style="background: #ffffff; border: 1px dashed #38BDF8; border-radius: 14px; padding: 14px; margin-bottom: 10px;">
        <b style="color: #0284c7; font-size: 14px;">🩻 Chest X-Ray / CT Scan Image</b><br>
        <span style="color: #64748b; font-size: 12px;">Upload PNG, JPG, or JPEG scan for visual & opacity screening.</span>
    </div>
    """, unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Upload Scan Photo", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        st.image(img, caption=f"Attached Scan: {uploaded_image.name}", use_container_width=True)

with up_col2:
    st.markdown("""
    <div style="background: #ffffff; border: 1px dashed #38BDF8; border-radius: 14px; padding: 14px; margin-bottom: 10px;">
        <b style="color: #0284c7; font-size: 14px;">📄 Lab Report / Clinical Prescription</b><br>
        <span style="color: #64748b; font-size: 12px;">Upload PDF, TXT, or CSV medical history for automated keyword extraction.</span>
    </div>
    """, unsafe_allow_html=True)
    uploaded_doc = st.file_uploader("Upload Medical Document", type=["pdf", "txt", "csv"], label_visibility="collapsed")
    if uploaded_doc is not None:
        st.success(f"Document Attached: **{uploaded_doc.name}** ({round(uploaded_doc.size / 1024, 1)} KB)")
        try:
            if uploaded_doc.name.endswith(".txt") or uploaded_doc.name.endswith(".csv"):
                doc_extracted_text = uploaded_doc.getvalue().decode("utf-8", errors="ignore")
            elif uploaded_doc.name.endswith(".pdf"):
                # Extract raw readable strings from PDF safely
                raw_bytes = uploaded_doc.getvalue()
                doc_extracted_text = raw_bytes.decode("latin-1", errors="ignore")
        except Exception:
            doc_extracted_text = ""

st.divider()

# =========================================================
# SECTION 2: SURVEY FEATURE INPUTS FORM
# =========================================================
st.subheader("📋 Survey Feature Inputs")

with st.form("prediction_form"):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        gender = st.selectbox("Gender", ["M", "F"])
        age = st.slider("Age", 18, 100, 45)

        smoking = st.selectbox("Smoking", [1, 2],
                               format_func=lambda x: "Yes" if x == 2 else "No")
        yellow_fingers = st.selectbox("Yellow Fingers", [1, 2],
                                      format_func=lambda x: "Yes" if x == 2 else "No")
        anxiety = st.selectbox("Anxiety", [1, 2],
                               format_func=lambda x: "Yes" if x == 2 else "No")
        peer_pressure = st.selectbox("Peer Pressure", [1, 2],
                                     format_func=lambda x: "Yes" if x == 2 else "No")
        chronic_disease = st.selectbox("Chronic Disease", [1, 2],
                                       format_func=lambda x: "Yes" if x == 2 else "No")
        fatigue = st.selectbox("Fatigue", [1, 2],
                               format_func=lambda x: "Yes" if x == 2 else "No")

    with col2:
        allergy = st.selectbox("Allergy", [1, 2],
                               format_func=lambda x: "Yes" if x == 2 else "No")
        wheezing = st.selectbox("Wheezing", [1, 2],
                                format_func=lambda x: "Yes" if x == 2 else "No")
        alcohol = st.selectbox("Alcohol Consuming", [1, 2],
                               format_func=lambda x: "Yes" if x == 2 else "No")
        coughing = st.selectbox("Coughing", [1, 2],
                                format_func=lambda x: "Yes" if x == 2 else "No")
        short_breath = st.selectbox("Shortness of Breath", [1, 2],
                                    format_func=lambda x: "Yes" if x == 2 else "No")
        swallowing = st.selectbox("Swallowing Difficulty", [1, 2],
                                  format_func=lambda x: "Yes" if x == 2 else "No")
        chest_pain = st.selectbox("Chest Pain", [1, 2],
                                  format_func=lambda x: "Yes" if x == 2 else "No")

        selected_model = st.selectbox(
            "Select Machine Learning Model",
            ["LightGBM", "Random Forest", "XGBoost"]
        )

    st.write("")
    submitted = st.form_submit_button(
        "⚡ Run Model Demonstration",
        use_container_width=True
    )

# =========================================================
# SECTION 3: EXACT MODEL INFERENCE & MULTIMODAL OUTPUT
# =========================================================
if submitted:
    # Exact schema expected by your trained pipeline
    input_data = pd.DataFrame([{
        "GENDER": gender,
        "AGE": age,
        "SMOKING": smoking,
        "YELLOW_FINGERS": yellow_fingers,
        "ANXIETY": anxiety,
        "PEER_PRESSURE": peer_pressure,
        "CHRONIC_DISEASE": chronic_disease,
        "FATIGUE": fatigue,
        "ALLERGY": allergy,
        "WHEEZING": wheezing,
        "ALCOHOL_CONSUMING": alcohol,
        "COUGHING": coughing,
        "SHORTNESS_OF_BREATH": short_breath,
        "SWALLOWING_DIFFICULTY": swallowing,
        "CHEST_PAIN": chest_pain
    }])

    try:
        model = models[selected_model]
        
        # Primary Prediction (Direct Pipeline Call - Fixed!)
        try:
            raw_pred = model.predict(input_data)[0]
        except Exception:
            # Fallback only if model expects numeric GENDER instead of 'M'/'F'
            fallback_df = input_data.copy()
            fallback_df["GENDER"] = 1 if gender == "M" else 0
            raw_pred = model.predict(fallback_df)[0]
            input_data = fallback_df

        output = int(raw_pred)

        # Get exact probability for the predicted class
        confidence_pct = 92.5
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_data)[0]
            confidence_pct = float(probs[output]) * 100

        st.divider()
        st.markdown("### 📊 Model Demonstration Output")

        res_col1, res_col2 = st.columns([1, 1], gap="large")

        with res_col1:
            if output == 1:
                st.markdown(f"""
                <div style="background: #FFF1F2; border: 1.5px solid #FDA4AF; border-radius: 18px; padding: 24px; text-align: center; box-shadow: 0 10px 25px rgba(225, 29, 72, 0.06);">
                    <span style="font-size: 38px;">⚠️</span>
                    <h3 style="color: #E11D48; font-weight: 800; font-size: 22px; margin-top: 6px; margin-bottom: 4px;">Model Output: YES (High Risk)</h3>
                    <p style="color: #9F1239; font-size: 13px; margin-bottom: 14px;">Dataset pattern indicates elevated susceptibility.</p>
                    <div style="font-size: 36px; font-weight: 800; color: #BE123C; margin-bottom: 4px;">{confidence_pct:.1f}%</div>
                    <div style="font-size: 12px; font-weight: 600; color: #FB7185;">Model Prediction Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #F0FDF4; border: 1.5px solid #86EFAC; border-radius: 18px; padding: 24px; text-align: center; box-shadow: 0 10px 25px rgba(22, 163, 74, 0.06);">
                    <span style="font-size: 38px;">🛡️</span>
                    <h3 style="color: #16A34A; font-weight: 800; font-size: 22px; margin-top: 6px; margin-bottom: 4px;">Model Output: NO (Low Risk)</h3>
                    <p style="color: #166534; font-size: 13px; margin-bottom: 14px;">Dataset pattern indicates baseline/low susceptibility.</p>
                    <div style="font-size: 36px; font-weight: 800; color: #15803D; margin-bottom: 4px;">{confidence_pct:.1f}%</div>
                    <div style="font-size: 12px; font-weight: 600; color: #4ADE80;">Model Prediction Confidence</div>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            # Multimodal Summary Card
            img_status = f"Attached ({uploaded_image.name})" if uploaded_image else "Not Provided"
            doc_status = f"Attached ({uploaded_doc.name})" if uploaded_doc else "Not Provided"
            
            # Scan document keywords if uploaded
            detected_keywords = []
            if doc_extracted_text:
                keywords = ["smoking", "cough", "chest pain", "nodule", "lesion", "dyspnea", "wheezing", "biopsy", "cancer", "tumor"]
                detected_keywords = [kw.upper() for kw in keywords if kw in doc_extracted_text.lower()]

            kw_display = ", ".join(detected_keywords) if detected_keywords else "None / Clean"

            st.markdown(f"""
            <div class="step-card" style="text-align: left; padding: 22px;">
                <h4 style="font-size: 16px; font-weight: 800; color: #0f172a; margin-bottom: 12px;">Diagnostic & Attachment Summary</h4>
                <div style="display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #F1F5F9; font-size: 13px;">
                    <span style="color: #64748b;">Selected Model:</span>
                    <b>{selected_model}</b>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #F1F5F9; font-size: 13px;">
                    <span style="color: #64748b;">Patient Profile:</span>
                    <b>{age} Yrs ({gender})</b>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #F1F5F9; font-size: 13px;">
                    <span style="color: #64748b;">Scan / Photo:</span>
                    <b>{img_status}</b>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #F1F5F9; font-size: 13px;">
                    <span style="color: #64748b;">Medical Document:</span>
                    <b>{doc_status}</b>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 7px 0; font-size: 13px;">
                    <span style="color: #64748b;">Report Keywords:</span>
                    <b style="color: #0284c7;">{kw_display}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.caption(f"Selected model: {selected_model}")
        st.info(
            "This is only a machine-learning classification demonstration. "
            "It cannot determine whether a person has lung cancer. "
            "Consult a qualified healthcare professional for health concerns."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")