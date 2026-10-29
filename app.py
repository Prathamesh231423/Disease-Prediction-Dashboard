from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
from utils import create_risk_gauge, get_prediction_verdict

# Page configuration
st.set_page_config(
    page_title="Disease Prediction Dashboard",
    page_icon="❤️",
    layout="wide"
)

BASE_DIR = Path(__file__).parent.resolve()
DATA_PATH = BASE_DIR / "data" / "heart.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

@st.cache_resource
def load_trained_model():
    if not MODEL_PATH.exists(): return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_dataset():
    if not DATA_PATH.exists(): return None
    return pd.read_csv(DATA_PATH)

def main():
    st.title("❤️ Heart Disease Prediction Dashboard")
    df = load_dataset()
    model = load_trained_model()

    if model is None:
        st.error("Model file `model.pkl` not found. Please run `python train.py` to generate it.")
        return

    # Initialize session state for example loading
    if 'input_vals' not in st.session_state:
        st.session_state.input_vals = {
            "age": 55, "sex": 1, "cp": 0, "trestbps": 130, "chol": 240, "fbs": 0,
            "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2
        }

    # Sidebar: Navigation & Example Loading
    with st.sidebar:
        st.title("🎛️ Dashboard Settings")

        menu = st.radio(
            "Main Menu",
            ["🔍 Prediction Tool", "📊 Model Metrics", "📈 Data Explorer"],
            label_visibility="collapsed"
        )

        st.divider()
        st.header("Load Examples")
        if st.button("Low-Risk Case", use_container_width=True):
            st.session_state.input_vals = {
                "age": 40, "sex": 0, "cp": 2, "trestbps": 110, "chol": 190, "fbs": 0,
                "restecg": 0, "thalach": 180, "exang": 0, "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2
            }
            st.rerun()
        if st.button("High-Risk Case", use_container_width=True):
            st.session_state.input_vals = {
                "age": 65, "sex": 1, "cp": 0, "trestbps": 160, "chol": 300, "fbs": 1,
                "restecg": 2, "thalach": 110, "exang": 1, "oldpeak": 3.0, "slope": 0, "ca": 2, "thal": 3
            }
            st.rerun()
        st.divider()

    # Sidebar: Input Form (Only shown on Prediction Tool tab)
    if menu == "🔍 Prediction Tool":
        with st.sidebar:
            st.header("Patient Input")
            with st.form("input_form"):
                tabs = st.tabs(["Demographics", "Vitals", "Tests"])

                v = st.session_state.input_vals

                with tabs[0]: # Demographics
                    age = st.slider("Age", 18, 100, v["age"])
                    sex = st.radio("Sex", [("Male", 1), ("Female", 0)], index=0 if v["sex"]==1 else 1, format_func=lambda x: x[0])[1]

                with tabs[1]: # Vitals
                    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, v["trestbps"])
                    chol = st.slider("Cholesterol (mg/dl)", 100, 600, v["chol"])
                    thalach = st.slider("Max Heart Rate (bpm)", 70, 220, v["thalach"])

                with tabs[2]: # Tests
                    cp = st.selectbox("Chest Pain Type", [("Typical Angina", 0), ("Atypical Angina", 1), ("Non-anginal Pain", 2), ("Asymptomatic", 3)], index=v["cp"], format_func=lambda x: x[0])[1]
                    fbs = st.radio("Fasting Blood Sugar > 120mg/dl", [("No", 0), ("Yes", 1)], index=v["fbs"], format_func=lambda x: x[0])[1]
                    restecg = st.radio("Resting ECG", [("Normal", 0), ("Abnormal", 1), ("Hypertrophy", 2)], index=v["restecg"], format_func=lambda x: x[0])[1]
                    exang = st.radio("Exercise-Induced Angina", [("No", 0), ("Yes", 1)], index=v["exang"], format_func=lambda x: x[0])[1]
                    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, float(v["oldpeak"]), help="ST depression induced by exercise relative to rest")
                    slope = st.radio("Slope", [("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)], index=v["slope"], format_func=lambda x: x[0])[1]
                    ca = st.slider("Major Vessels (ca)", 0, 3, v["ca"], help="Number of major vessels (0-3) colored by fluoroscopy")
                    thal = st.radio("Thalassemia", [("Normal", 1), ("Fixed", 2), ("Reversible", 3)], index=v["thal"]-1, format_func=lambda x: x[0])[1]

                submitted = st.form_submit_button("Predict Heart Disease Risk", use_container_width=True)

    # Handle Navigation Tabs
    if menu == "🔍 Prediction Tool":
        st.subheader("Patient Clinical Assessment & Prediction")
        st.markdown("Use the sidebar input form to enter patient parameters, then click **Predict Heart Disease Risk**.")

        if 'submitted' in locals() and submitted:
            with st.spinner("Analyzing patient data..."):
                input_df = pd.DataFrame([{
                    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": fbs,
                    "restecg": restecg, "thalach": thalach, "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
                }])
                input_df = input_df[FEATURE_NAMES]
                prob = model.predict_proba(input_df)[0][1]

                st.subheader("Assessment Results")
                res_col1, res_col2 = st.columns([2, 1])
                with res_col1:
                    st.plotly_chart(create_risk_gauge(prob), use_container_width=True)
                with res_col2:
                    st.metric("Risk Probability", f"{prob:.1%}")
                    get_prediction_verdict(prob)
                    st.download_button("Download Prediction Report", input_df.assign(probability=prob).to_csv(index=False), "report.csv", "text/csv")

                # Suggestion Section for High/Moderate Risk
                if prob >= 0.7:
                    st.error("### ⚠️ Critical Action Steps")
                    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063198.png", width=100)
                    st.markdown("""
                    Based on the model's analysis, a **High Risk** of heart disease has been indicated.

                    **Immediate Recommended Steps:**
                    1.  **Consult a Physician:** Schedule an urgent appointment with a cardiologist or your primary healthcare provider.
                    2.  **Clinical Validation:** These results are statistical estimates. A professional clinical examination, including an EKG, stress test, or blood work, is necessary for an actual diagnosis.
                    3.  **Monitor Symptoms:** Note any chest pain, shortness of breath, dizziness, or unusual fatigue to report to your doctor.
                    4.  **Prepare Data:** You can download the prediction report above to share the clinical parameters used in this assessment with your provider.

                    *Do not ignore symptoms even if the model were to show low risk. Seek emergency care immediately if you experience severe chest pain.*
                    """)
                elif prob >= 0.4:
                    st.warning("### 🔔 Precautionary Steps")
                    st.image("https://cdn-icons-png.flaticon.com/512/2921/2921226.png", width=100)
                    st.markdown("""
                    A **Moderate Risk** has been indicated.

                    **Recommended Steps:**
                    1.  **Follow-up:** Mention these results to your doctor during your next visit.
                    2.  **Lifestyle Review:** Consider monitoring your blood pressure and cholesterol levels more closely.
                    3.  **Preventative Care:** Focus on heart-healthy habits like regular exercise and a balanced diet.
                    """)
                else:
                    st.success("### ✅ Low Risk - Healthy Status")
                    st.image("https://cdn-icons-png.flaticon.com/512/1077/1077978.png", width=100)
                    st.markdown("""
                    The model indicates a **Low Risk** of heart disease.

                    **Recommended Steps to Stay Healthy:**
                    1.  **Maintain Healthy Habits:** Continue eating a balanced diet rich in vegetables, fruits, and whole grains.
                    2.  **Regular Activity:** Aim for at least 30 minutes of moderate physical activity daily.
                    3.  **Routine Check-ups:** Keep up with regular preventative health screenings.
                    """)

                with st.expander("Why this prediction? (Feature Importance)"):
                    if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                        clf = model.named_steps['classifier']
                        if hasattr(clf, 'feature_importances_'):
                            feat_imp = pd.Series(clf.feature_importances_, index=FEATURE_NAMES).sort_values()
                            fig, ax = plt.subplots()
                            feat_imp.plot(kind='barh', ax=ax, color='skyblue')
                            ax.set_title("Feature Importance")
                            st.pyplot(fig)

    elif menu == "📊 Model Metrics":
        st.subheader("Model Evaluation Metrics")
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

        if df is not None:
            X = df[FEATURE_NAMES]
            y = df["target"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            y_pred = model.predict(X_test)
            y_probs = model.predict_proba(X_test)[:, 1]

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
            m_col2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.1%}")
            m_col3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.1%}")
            m_col4.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.1%}")

            e_col1, e_col2 = st.columns(2)
            with e_col1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)
            with e_col2:
                st.subheader("ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, y_probs)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                ax.plot([0, 1], [0, 1], 'k--')
                ax.legend()
                st.pyplot(fig)

    elif menu == "📈 Data Explorer":
        st.subheader("Dataset Preview")
        if df is not None:
            st.dataframe(df, use_container_width=True)
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.subheader("Age Distribution")
                fig, ax = plt.subplots()
                sns.histplot(df['age'], kde=True, ax=ax)
                st.pyplot(fig)
            with d_col2:
                st.subheader("Target Balance")
                st.bar_chart(df['target'].value_counts())

    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** This dashboard is for educational use only and does not provide medical diagnosis. Please consult a doctor for clinical advice.")

if __name__ == "__main__":
    main()
