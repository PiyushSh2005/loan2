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

# Manual Input
if input_type == "Manual Input":
    st.header("Enter Applicant Details")

    # Example fields – ADD MORE to match your training data
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

    # Convert to numeric DataFrame based on your model features
    input_dict = {
        'Gender': 1 if gender == "Male" else 0,
        'Married': 1 if married == "Yes" else 0,
        'Dependents': 3 if dependents == "3+" else int(dependents),
        'Education': 1 if education == "Graduate" else 0,
        'Self_Employed': 1 if self_employed == "Yes" else 0,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0}[property_area]
    }

    input_df = pd.DataFrame([input_dict])

    # Match feature columns
    try:
        input_df = input_df[expected_features]
        if st.button("Predict"):
            prediction = predict(input_df)
            if prediction is not None:
                st.success(f"🏁 Loan Prediction: {'Approved ✅' if prediction[0] == 1 else 'Rejected ❌'}")
    except KeyError as e:
        st.error(f"Missing input for: {e}")

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
