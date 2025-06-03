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
    st.sidebar.success("Model, Scaler, and Feature Info loaded successfully!")
except FileNotFoundError as e:
    st.sidebar.error(f"Error: One of the model/scaler/feature files was not found. Ensure the .pkl files are in the same directory as app.py. Details: {e}")
    st.stop()
except Exception as e:
    st.sidebar.error(f"General error loading model/scaler/features: {e}. Please check your .pkl files.")
    st.stop()

OPTIMAL_THRESHOLD = 0.6231

def preprocess_input(input_data_raw, feature_names, categorical_info, scaler):
    input_df = pd.DataFrame([input_data_raw])
    for col, categories in categorical_info.items():
        if col in input_df.columns:
            input_df[col] = pd.Categorical(input_df[col], categories=categories)
    input_df_encoded = pd.get_dummies(input_df, drop_first=True) # drop_first=True is common for OHE
    input_final = input_df_encoded.reindex(columns=feature_names, fill_value=0)
    input_scaled = scaler.transform(input_final)
    return input_scaled

st.header("1. Financial & Personal Data:")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Age", min_value=18, max_value=70, value=30, help="Customer's age in years")
    income = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=50000, step=1000, help="Customer's gross annual income")
    loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=1000000, value=15000, step=1000, help="Total loan amount requested")
with col2:
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=700, help="Customer's credit score (e.g., FICO Score)")
    months_employed = st.slider("Months Employed", min_value=0, max_value=360, value=60, help="Duration the customer has been at their current job")
    num_credit_lines = st.slider("Number of Active Credit Lines", min_value=1, max_value=15, value=2, help="How many active credit accounts the customer has")
with col3:
    interest_rate = st.number_input("Loan Interest Rate (%)", min_value=0.01, max_value=30.00, value=8.00, format="%.2f", help="Offered loan interest rate") / 100
    loan_term = st.selectbox("Loan Term (Months)", options=[12, 24, 36, 48, 60, 72, 84, 96, 108, 120], index=2, help="Loan duration in months")
    dti_ratio = st.number_input("DTI Ratio (Debt-to-Income)", min_value=0.01, max_value=0.99, value=0.40, format="%.2f", help="Ratio of total monthly debt payments to monthly income (0.01 to 0.99)")

st.header("2. Additional Information:")
col4, col5, col6 = st.columns(3)
with col4:
    education = st.selectbox("Education", options=categorical_info.get('Education', []), help="Customer's highest education level")
    employment_type = st.selectbox("Employment Type", options=categorical_info.get('EmploymentType', []), help="Customer's employment status type")
with col5:
    marital_status = st.selectbox("Marital Status", options=categorical_info.get('MaritalStatus', []), help="Customer's marital status")
    has_mortgage = st.selectbox("Has Mortgage?", options=categorical_info.get('HasMortgage', []), help="Does the customer have a mortgage?")
with col6:
    has_dependents = st.selectbox("Has Dependents?", options=categorical_info.get('HasDependents', []), help="Does the customer have dependents (children, etc.)?")
    loan_purpose = st.selectbox("Loan Purpose", options=categorical_info.get('LoanPurpose', []), help="Main purpose for which the customer is applying for the loan")
    has_cosigner = st.selectbox("Has Co-signer?", options=categorical_info.get('HasCoSigner', []), help="Does the loan have a co-signer?")

st.markdown("---")
if st.button("Predict Default Risk"):
    raw_input = {
        'Age': age,
        'Income': income,
        'LoanAmount': loan_amount,
        'CreditScore': credit_score,
        'MonthsEmployed': months_employed,
        'NumCreditLines': num_credit_lines,
        'InterestRate': interest_rate,
        'LoanTerm': loan_term,
        'DTIRatio': dti_ratio,
        'Education': education,
        'EmploymentType': employment_type,
        'MaritalStatus': marital_status,
        'HasMortgage': has_mortgage,
        'HasDependents': has_dependents,
        'LoanPurpose': loan_purpose,
        'HasCoSigner': has_cosigner
    }

    try:
        processed_input = preprocess_input(raw_input, feature_names, categorical_info, scaler)
    except Exception as e:
        st.error(f"An error occurred while processing input data. Ensure all fields are filled correctly. Details: {e}")
        st.stop()

    prediction_proba = model.predict_proba(processed_input)[:, 1][0]
    prediction_class = (prediction_proba >= OPTIMAL_THRESHOLD).astype(int)

    st.subheader("Prediction Results:")
    st.write(f"**Probability of Default:** `{prediction_proba:.2%}`")

    if prediction_class == 1:
        st.error(f"**Model Decision:** Customer is **LIKELY TO DEFAULT** (high risk)")
        st.markdown(f"""
        <div style='padding: 10px; border-left: 5px solid red; background-color: #ffe0e0; color: black;'>
            Recommendation for the Bank:
            <ul>
                <li>This customer has a high default probability according to the model.</li>
                <li><strong>Further review</strong> by a senior credit analyst is recommended.</li>
                <li>Consider <strong>rejecting the loan application</strong> or offering <strong>stricter loan terms</strong> (e.g., higher interest rate, smaller loan amount, mandatory co-signer).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"**Model Decision:** Customer is **UNLIKELY TO DEFAULT** (low risk)")
        st.markdown(f"""
        <div style='padding: 10px; border-left: 5px solid green; background-color: #e0ffe0; color: black;'>
            Recommendation for the Bank:
            <ul>
                <li>This customer has a low default probability.</li>
                <li>The loan may likely be <strong>approved</strong> following standard bank procedures.</li>
                <li>Still perform <strong>standard data verification</strong> as per bank policy.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.info(f"*(Threshold used for decision: {OPTIMAL_THRESHOLD:.4f})*")
    st.caption("This prediction is a model-based recommendation. The final decision rests with the bank.")