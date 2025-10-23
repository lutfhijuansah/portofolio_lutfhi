# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Loan Default Prediction Application", layout="centered")

st.title("Loan Default Risk Prediction Application")
st.write("Fill out the form below to predict the likelihood of a customer defaulting on a loan.")
st.markdown("---")

# --- Load Model, Scaler, and Feature Names ---
try:
    model = joblib.load('logistic_regression_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    categorical_info = joblib.load('categorical_info.pkl')
    st.sidebar.success("✅ Model, Scaler, and Feature Info loaded successfully!")
except FileNotFoundError as e:
    st.sidebar.error(f"❌ Error: Missing model/scaler/feature file. {e}")
    st.stop()
except Exception as e:
    st.sidebar.error(f"❌ General error loading model/scaler/features: {e}")
    st.stop()

# ✅ Default threshold now set to 0.50 (but user can adjust it in UI)
DEFAULT_THRESHOLD = 0.50

def preprocess_input(input_data_raw, feature_names, categorical_info, scaler):
    input_df = pd.DataFrame([input_data_raw])

    # Handle categorical encoding
    for col, categories in categorical_info.items():
        if col in input_df.columns:
            input_df[col] = pd.Categorical(input_df[col], categories=categories)

    input_df_encoded = pd.get_dummies(input_df, drop_first=True)
    input_final = input_df_encoded.reindex(columns=feature_names, fill_value=0)

    # Scale numerical data
    input_scaled = scaler.transform(input_final)
    return input_scaled

# --- User Input Form ---
st.header("1. Financial & Personal Data:")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Age", 18, 70, 30)
    income = st.number_input("Annual Income ($)", 10000, 500000, 50000, step=1000)
    loan_amount = st.number_input("Loan Amount ($)", 1000, 1000000, 15000, step=1000)
with col2:
    credit_score = st.slider("Credit Score", 300, 850, 700)
    months_employed = st.slider("Months Employed", 0, 360, 60)
    num_credit_lines = st.slider("Number of Credit Lines", 1, 15, 2)
with col3:
    interest_rate = st.number_input("Loan Interest Rate (%)", 0.01, 30.00, 8.00, format="%.2f") / 100
    loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60, 72, 84, 96, 108, 120], index=2)
    dti_ratio = st.number_input("DTI Ratio (Debt-to-Income)", 0.01, 0.99, 0.40, format="%.2f")

st.header("2. Additional Information:")
col4, col5, col6 = st.columns(3)
with col4:
    education = st.selectbox("Education", categorical_info.get('Education', []))
    employment_type = st.selectbox("Employment Type", categorical_info.get('EmploymentType', []))
with col5:
    marital_status = st.selectbox("Marital Status", categorical_info.get('MaritalStatus', []))
    has_mortgage = st.selectbox("Has Mortgage?", categorical_info.get('HasMortgage', []))
with col6:
    has_dependents = st.selectbox("Has Dependents?", categorical_info.get('HasDependents', []))
    loan_purpose = st.selectbox("Loan Purpose", categorical_info.get('LoanPurpose', []))
    has_cosigner = st.selectbox("Has Co-signer?", categorical_info.get('HasCoSigner', []))

st.markdown("---")

# ✅ Optional threshold customization for advanced users/demo
st.subheader("⚙️ Threshold for Decision Making")
threshold = st.slider("Set decision threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01,
                      help="Higher threshold = stricter approval (fewer false positives, lower recall)")

# --- Predict Button ---
if st.button("Predict Default Risk"):
    raw_input = {
        'Age': age, 'Income': income, 'LoanAmount': loan_amount, 'CreditScore': credit_score,
        'MonthsEmployed': months_employed, 'NumCreditLines': num_credit_lines, 'InterestRate': interest_rate,
        'LoanTerm': loan_term, 'DTIRatio': dti_ratio, 'Education': education, 'EmploymentType': employment_type,
        'MaritalStatus': marital_status, 'HasMortgage': has_mortgage, 'HasDependents': has_dependents,
        'LoanPurpose': loan_purpose, 'HasCoSigner': has_cosigner
    }

    try:
        processed_input = preprocess_input(raw_input, feature_names, categorical_info, scaler)
    except Exception as e:
        st.error(f"⚠️ Error processing input: {e}")
        st.stop()

    prediction_proba = model.predict_proba(processed_input)[:, 1][0]
    prediction_class = int(prediction_proba >= threshold)

    st.subheader("Prediction Results:")
    st.write(f"**Probability of Default:** `{prediction_proba:.2%}`")

    if prediction_class == 1:
        st.error("**Model Decision:** Customer is **LIKELY TO DEFAULT** (High Risk)")
    else:
        st.success("**Model Decision:** Customer is **UNLIKELY TO DEFAULT** (Low Risk)")

    st.info(f"*(Threshold used for decision: {threshold:.4f})*")
    st.caption("This prediction is a model-based recommendation. The final decision rests with the bank.")
