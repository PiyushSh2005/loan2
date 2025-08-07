import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
expected_features = scaler.feature_names_in_

st.set_page_config(layout="wide")
st.title("🏦 Loan Approval Prediction App")

def predict(df):
    try:
        df = df[expected_features]
        scaled = scaler.transform(df)
        return model.predict(scaled), model.predict_proba(scaled)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# ➡️ Create two columns
left_col, right_col = st.columns(2)

# =======================
# 🔹 LEFT COLUMN: Inputs + Quick Predict
# =======================
with left_col:
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

    # Prepare input
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

    input_encoded = pd.get_dummies(raw_input)
    input_encoded = input_encoded.reindex(columns=expected_features, fill_value=0)

    # 🔘 Left button: Quick Predict (result only)
    if st.button("Predict", key="left_predict"):
        prediction, proba = predict(input_encoded)
        if prediction is not None:
            result = "Approved ✅" if prediction[0] == 1 else "Rejected ❌"
            confidence = proba[0][1]
            st.success(f"🏁 Loan Prediction: {result}")
            st.info(f"💡 Confidence: {confidence:.2%}")

# =========================
# 🔸 RIGHT COLUMN: Charts & Analysis
# =========================
with right_col:
    st.header("📊 Prediction Analysis")

    # 🔘 Right button: Predict + Show Charts
    if st.button("Make Prediction with Charts", key="right_predict"):
        prediction, proba = predict(input_encoded)
        if prediction is not None:
            result = "Approved ✅" if prediction[0] == 1 else "Rejected ❌"
            confidence = proba[0][1]

            st.success(f"🏁 Loan Prediction: {result}")
            st.info(f"💡 Confidence: {confidence:.2%}")

            # Chart 1: Confidence
            st.subheader("📊 Model Confidence")
            fig1, ax1 = plt.subplots()
            ax1.bar(["Rejected", "Approved"], [1 - confidence, confidence], color=["red", "green"])
            ax1.set_ylabel("Probability")
            ax1.set_ylim(0, 1)
            st.pyplot(fig1)

            # Chart 2: Income vs Loan Amount
            st.subheader("💰 Income vs Loan Amount")
            total_income = applicant_income + coapplicant_income
            fig2, ax2 = plt.subplots()
            ax2.bar(["Total Income", "Loan Amount (×1000)"], [total_income, loan_amount], color=["blue", "orange"])
            ax2.set_ylabel("Amount")
            ax2.set_title("Loan vs Income Comparison")
            st.pyplot(fig2)
