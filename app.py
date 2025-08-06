import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Get expected features from scaler
expected_features = scaler.feature_names_in_

st.title("🏦 Loan Approval Prediction App")

# Sidebar option
st.sidebar.header("Choose Input Type")
input_type = st.sidebar.radio("", ["Manual Input", "Upload CSV"])

def predict(df):
    try:
        df = df[expected_features]  # Ensure column order matches scaler training
        scaled = scaler.transform(df)
        preds = model.predict(scaled)
        return preds
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

if input_type == "Manual Input":
    st.header("Enter Applicant Details")

    # Input fields
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Amount Term", min_value=0)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    # Raw input DataFrame
    raw_input = pd.DataFrame([{
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }])

    # One-hot encode
    input_encoded = pd.get_dummies(raw_input)
    input_encoded = input_encoded.reindex(columns=expected_features, fill_value=0)

    if st.button("Predict"):
        try:
            prediction = model.predict(input_encoded)
           

            result = "Approved ✅" if prediction[0] == 1 else "Rejected ❌"
            st.success(f"🏁 Loan Prediction: {result}")
            st.info(f"💡 Model confidence (approval): {proba:.2%}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
proba = model.predict_proba(input_encoded)[0][1]
st.info(f"💡 Model confidence (approval): {proba:.2%}")


