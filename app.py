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

# ===========================
# ✅ MANUAL INPUT SECTION
# ===========================

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
            proba = model.predict_proba(input_encoded)[0][1]

            result = "Approved ✅" if prediction[0] == 1 else "Rejected ❌"
            st.success(f"🏁 Loan Prediction: {result}")
            st.info(f"💡 Model confidence (approval): {proba:.2%}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ===========================
# ✅ CSV UPLOAD SECTION
# ===========================

elif input_type == "Upload CSV":
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("📄 Uploaded Data Preview")
            st.dataframe(df)

            # One-hot encode + align columns
            df_encoded = pd.get_dummies(df)
            df_encoded = df_encoded.reindex(columns=expected_features, fill_value=0)

            # Predictions and probabilities
            predictions = model.predict(df_encoded)
            probabilities = model.predict_proba(df_encoded)[:, 1]

            df['Prediction'] = predictions
            df['Approval Probability'] = probabilities

            st.subheader("🧾 Prediction Results")
            st.dataframe(df)

            # ✅ Chart 1: Bar Chart of Predictions
            st.subheader("📊 Loan Approval Counts")
            counts = df['Prediction'].value_counts().rename({0: "Rejected", 1: "Approved"})
            fig1, ax1 = plt.subplots()
            counts.plot(kind='bar', color=['red', 'green'], ax=ax1)
            ax1.set_ylabel("Number of Applicants")
            ax1.set_title("Loan Approval Distribution")
            st.pyplot(fig1)

            # ✅ Chart 2: Pie Chart
            st.subheader("🥧 Approval Rate Pie Chart")
            fig2, ax2 = plt.subplots()
            ax2.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=['red', 'green'])
            ax2.axis('equal')
            st.pyplot(fig2)

            # ✅ Chart 3: Histogram of Loan Amount by Approval
            st.subheader("💰 Loan Amount Distribution by Approval")
            fig3, ax3 = plt.subplots()
            df[df['Prediction'] == 1]['LoanAmount'].plot(kind='hist', alpha=0.6, bins=20, label='Approved', color='green', ax=ax3)
            df[df['Prediction'] == 0]['LoanAmount'].plot(kind='hist', alpha=0.6, bins=20, label='Rejected', color='red', ax=ax3)
            ax3.set_xlabel("Loan Amount")
            ax3.set_title("Loan Amounts by Approval Status")
            ax3.legend()
            st.pyplot(fig3)

            # ✅ Chart 4: Histogram of Income by Approval
            st.subheader("👤 Applicant Income by Approval")
            fig4, ax4 = plt.subplots()
            df[df['Prediction'] == 1]['ApplicantIncome'].plot(kind='hist', alpha=0.6, bins=20, label='Approved', color='green', ax=ax4)
            df[df['Prediction'] == 0]['ApplicantIncome'].plot(kind='hist', alpha=0.6, bins=20, label='Rejected', color='red', ax=ax4)
            ax4.set_xlabel("Applicant Income")
            ax4.set_title("Income Distribution by Approval")
            ax4.legend()
            st.pyplot(fig4)

        except Exception as e:
            st.error(f"Something went wrong: {e}")
