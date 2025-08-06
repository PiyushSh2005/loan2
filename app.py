import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Loan Prediction App")

# Upload CSV or manual input
st.sidebar.header("Input Options")
input_option = st.sidebar.radio("Choose input type:", ["Manual Input", "Upload CSV"])

def predict(df):
    scaled_data = scaler.transform(df)
    predictions = model.predict(scaled_data)
    return predictions

if input_option == "Manual Input":
    st.header("Enter Applicant Data")
    # Example input fields – update according to your actual features
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    credit_history = st.selectbox("Credit History", [0, 1])
    
    # Convert to DataFrame (replace with all necessary features)
    input_df = pd.DataFrame({
        "Gender": [1 if gender == "Male" else 0],
        "Married": [1 if married == "Yes" else 0],
        "ApplicantIncome": [applicant_income],
        "LoanAmount": [loan_amount],
        "Credit_History": [credit_history],
    })
    
    if st.button("Predict"):
        result = predict(input_df)
        st.success(f"Loan Prediction: {'Approved' if result[0] == 1 else 'Rejected'}")

elif input_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(df)
        
        predictions = predict(df)
        df['Prediction'] = predictions
        st.subheader("Prediction Results")
        st.write(df)

        # Show charts
        st.subheader("Prediction Distribution")
        result_counts = df['Prediction'].value_counts().rename({0: 'Rejected', 1: 'Approved'})
        
        fig, ax = plt.subplots()
        result_counts.plot(kind='bar', color=['red', 'green'], ax=ax)
        ax.set_ylabel("Count")
        ax.set_title("Loan Approval Results")
        st.pyplot(fig)
