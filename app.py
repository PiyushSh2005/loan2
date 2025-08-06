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

# Sidebar option (no upload anymore)
st.sidebar.header("Input Type")
st.sidebar.markdown("Only manual input is available in this version.")

def predict(df):
    try:
        df = df[expected_features]  # Ensure column order matches scaler training
        scaled = scaler.transform(df)
        preds = model.predict(scaled)
        return preds
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# ===========================
# ✅ MANUAL INPUT SECTION
# ===========================

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
        proba = model.predict_proba(input_encoded)[0][1]

        result = "Approved ✅" if prediction[0] == 1 else "Rejected ❌"
        st.success(f"🏁 Loan Prediction: {result}")
        st.info(f"💡 Model confidence (approval): {proba:.2%}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
With this enhanced version including charts:

#✅ Enhanced Prediction Block with Charts

if st.button("Predict"):
    try:
        prediction = model.predict(input_encoded)
        proba = model.predict_proba(input_encoded)[0][1]

        result = "Approved ✅" if prediction[0] == 1 else "Rejected ❌"
        st.success(f"🏁 Loan Prediction: {result}")
        st.info(f"💡 Model confidence (approval): {proba:.2%}")

        # Chart 1: Model Confidence Bar Chart
        st.subheader("📊 Model Confidence")
        fig1, ax1 = plt.subplots()
        ax1.bar(["Rejected", "Approved"], [1 - proba, proba], color=['red', 'green'])
        ax1.set_ylabel("Probability")
        ax1.set_ylim(0, 1)
        st.pyplot(fig1)

        # Chart 2: Loan vs. Income Comparison
        st.subheader("💰 Loan vs. Income Comparison")
        income_total = applicant_income + coapplicant_income
        fig2, ax2 = plt.subplots()
        ax2.bar(["Total Income", "Loan Amount (×1000)"], [income_total, loan_amount], color=['blue', 'orange'])
        ax2.set_ylabel("Amount")
        ax2.set_title("Loan Request vs Total Income")
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

