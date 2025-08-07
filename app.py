import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import base64

# ✅ Set page config
st.set_page_config(layout="wide")

# ✅ Function to set background image
def set_background(image_file):
    with open(image_file, "rb") as img:
        b64_string = base64.b64encode(img.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{b64_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ✅ Set background
set_background("banking-and-finance-concept-digital-connect-system-financial-and-banking-technology-with-integrated-circles-glowing-line-icons-and-on-blue-background-design-vector.jpg")

# ✅ Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
expected_features = scaler.feature_names_in_

st.title("🏦 Loan Approval Prediction App")

# ✅ Prediction function
def predict(df):
    try:
        df = df[expected_features]
        scaled = scaler.transform(df)
        return model.predict(scaled), model.predict_proba(scaled)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# ✅ Two-column layout
left_col, right_col = st.columns(2)

# 🔹 LEFT COLUMN
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

    # Button: Predict
    if st.button("Predict", key="left_predict"):
        prediction, proba = predict(input_encoded)
        if prediction is not None:
            result = "Approved ✅" if prediction[0] == 1 else "Rejected ❌"
            confidence = proba[0][1]
            st.success(f"🏁 Loan Prediction: {result}")
            st.info(f"💡 Confidence: {confidence:.2%}")

# 🔸 RIGHT COLUMN
with right_col:
    st.header("📊 Prediction Analysis")

    if st.button("Make Prediction with Charts", key="right_predict"):
        prediction, proba = predict(input_encoded)
