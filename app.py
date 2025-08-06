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

    # One-hot encoding to match training
    input_encoded = pd.get_dummies(raw_input)

    # Reindex to expected columns
    input_encoded = input_encoded.reindex(columns=expected_features, fill_value=0)

    if st.button("Predict"):
        prediction = predict(input_encoded)
        if prediction is not None:
            st.success(f"🏁 Loan Prediction: {'Approved ✅' if prediction[0] == 1 else 'Rejected ❌'}")
proba = model.predict_proba(input_encoded)[0][1]
st.info(f"Model confidence (approval): {proba:.2%}")
# CSV Upload
else:
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("📄 Uploaded Data Preview")
            st.dataframe(df)

            df = df[expected_features]
            predictions = predict(df)
            df['Prediction'] = predictions

            st.subheader("🧾 Prediction Results")
            st.dataframe(df)

            # Plot bar chart
            st.subheader("📊 Loan Approval Distribution")
            counts = df['Prediction'].value_counts().rename({0: "Rejected", 1: "Approved"})
            fig, ax = plt.subplots()
            counts.plot(kind='bar', color=['red', 'green'], ax=ax)
            ax.set_ylabel("Count")
            ax.set_title("Loan Approval Results")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Something went wrong: {e}")


