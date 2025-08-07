import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ✅ Page config
st.set_page_config(layout="wide")

# ✅ Set background
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

set_background("banking-and-finance-concept-digital-connect-system-financial-and-banking-technology-with-integrated-circles-glowing-line-icons-and-on-blue-background-design-vector.jpg")

# ✅ Load models
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

# ✅ PDF generation
def generate_pdf_report(data, prediction, confidence):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    text = c.beginText(50, 750)
    text.setFont("Helvetica", 12)
    text.textLine("Loan Approval Prediction Report")
    text.textLine("-----------------------------------")

    for key, value in data.items():
        text.textLine(f"{key}: {value}")

    text.textLine("")
    text.textLine(f"Prediction Result: {'Approved ✅' if prediction == 1 else 'Rejected ❌'}")
    text.textLine(f"Confidence: {confidence:.2%}")

    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ✅ Two-column layout
left_col, right_col = st.columns(2)

# ========== LEFT COLUMN ==========
with left_col:
    st.header("Enter Applicant Details")

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

    st.session_state.input_encoded = input_encoded
    st.session_state.applicant_income = applicant_income
    st.session_state.coapplicant_income = coapplicant_income
    st.session_state.loan_amount = loan_amount
    st.session_state.form_data = raw_input.iloc[0].to_dict()

    if st.button("Predict", key="left_predict"):
        prediction, proba = predict(input_encoded)
        if prediction is not None:
            result = "Approved ✅" if prediction[0] == 1 else "Rejected ❌"
            confidence = proba[0][1]
            st.success(f"🏁 Loan Prediction: {result}")
            st.info(f"💡 Confidence: {confidence:.2%}")

# ========== RIGHT COLUMN ==========
with right_col:
    st.header("📊 Prediction Analysis")

    if st.button("Make Prediction with Charts", key="right_predict"):
        input_encoded = st.session_state.get("input_encoded")
        applicant_income = st.session_state.get("applicant_income", 0)
        coapplicant_income = st.session_state.get("coapplicant_income", 0)
        loan_amount = st.session_state.get("loan_amount", 0)
        form_data = st.session_state.get("form_data", {})

        if input_encoded is not None:
            prediction, proba = predict(input_encoded)
            if prediction is not None:
                result = "Approved ✅" if prediction[0] == 1 else "Rejected ❌"
                confidence = proba[0][1]

                st.success(f"🏁 Loan Prediction: {result}")
                st.info(f"💡 Confidence: {confidence:.2%}")

                # Chart 1: Model Confidence
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

                # PDF download
                st.subheader("📄 Download Prediction Report")
                pdf_file = generate_pdf_report(form_data, prediction[0], confidence)
                st.download_button(
                    label="📥 Download Report as PDF",
                    data=pdf_file,
                    file_name="loan_prediction_report.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("Please fill out the form on the left first.")












# import streamlit as st
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# import base64

# # ✅ Set page layout
# st.set_page_config(layout="wide")

# # ✅ Function to set background image
# def set_background(image_file):
#     with open(image_file, "rb") as img:
#         b64_string = base64.b64encode(img.read()).decode()
#     css = f"""
#     <style>
#     .stApp {{
#         background-image: url("data:image/jpg;base64,{b64_string}");
#         background-size: cover;
#         background-position: center;
#         background-repeat: no-repeat;
#         background-attachment: fixed;
#     }}
#     </style>
#     """
#     st.markdown(css, unsafe_allow_html=True)

# # ✅ Set background image
# set_background("banking-and-finance-concept-digital-connect-system-financial-and-banking-technology-with-integrated-circles-glowing-line-icons-and-on-blue-background-design-vector.jpg")

# # ✅ Load model and scaler
# model = joblib.load("model.pkl")
# scaler = joblib.load("scaler.pkl")
# expected_features = scaler.feature_names_in_

# st.title("🏦 Loan Approval Prediction App")

# # ✅ Prediction function
# def predict(df):
#     try:
#         df = df[expected_features]
#         scaled = scaler.transform(df)
#         return model.predict(scaled), model.predict_proba(scaled)
#     except Exception as e:
#         st.error(f"Prediction error: {e}")
#         return None, None

# # ✅ Two-column layout
# left_col, right_col = st.columns(2)

# # ========== LEFT COLUMN ==========
# with left_col:
#     st.header("Enter Applicant Details")

#     gender = st.selectbox("Gender", ["Male", "Female"])
#     married = st.selectbox("Married", ["Yes", "No"])
#     dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
#     education = st.selectbox("Education", ["Graduate", "Not Graduate"])
#     self_employed = st.selectbox("Self Employed", ["Yes", "No"])
#     applicant_income = st.number_input("Applicant Income", min_value=0)
#     coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
#     loan_amount = st.number_input("Loan Amount", min_value=0)
#     loan_term = st.number_input("Loan Amount Term", min_value=0)
#     credit_history = st.selectbox("Credit History", [1.0, 0.0])
#     property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

#     # Prepare input
#     raw_input = pd.DataFrame([{
#         'Gender': gender,
#         'Married': married,
#         'Dependents': dependents,
#         'Education': education,
#         'Self_Employed': self_employed,
#         'ApplicantIncome': applicant_income,
#         'CoapplicantIncome': coapplicant_income,
#         'LoanAmount': loan_amount,
#         'Loan_Amount_Term': loan_term,
#         'Credit_History': credit_history,
#         'Property_Area': property_area
#     }])

#     input_encoded = pd.get_dummies(raw_input)
#     input_encoded = input_encoded.reindex(columns=expected_features, fill_value=0)

#     # Store input in session for reuse
#     st.session_state.input_encoded = input_encoded
#     st.session_state.applicant_income = applicant_income
#     st.session_state.coapplicant_income = coapplicant_income
#     st.session_state.loan_amount = loan_amount

#     # Predict button
#     if st.button("Predict", key="left_predict"):
#         prediction, proba = predict(input_encoded)
#         if prediction is not None:
#             result = "Approved ✅" if prediction[0] == 1 else "Rejected ❌"
#             confidence = proba[0][1]
#             st.success(f"🏁 Loan Prediction: {result}")
#             st.info(f"💡 Confidence: {confidence:.2%}")

# # ========== RIGHT COLUMN ==========
# with right_col:
#     st.header("📊 Prediction Analysis")

#     if st.button("Make Prediction with Charts", key="right_predict"):
#         input_encoded = st.session_state.get("input_encoded", None)
#         applicant_income = st.session_state.get("applicant_income", 0)
#         coapplicant_income = st.session_state.get("coapplicant_income", 0)
#         loan_amount = st.session_state.get("loan_amount", 0)

#         if input_encoded is not None:
#             prediction, proba = predict(input_encoded)
#             if prediction is not None:
#                 result = "Approved ✅" if prediction[0] == 1 else "Rejected ❌"
#                 confidence = proba[0][1]

#                 st.success(f"🏁 Loan Prediction: {result}")
#                 st.info(f"💡 Confidence: {confidence:.2%}")

#                 # Chart 1: Model Confidence
#                 st.subheader("📊 Model Confidence")
#                 fig1, ax1 = plt.subplots()
#                 ax1.bar(["Rejected", "Approved"], [1 - confidence, confidence], color=["red", "green"])
#                 ax1.set_ylabel("Probability")
#                 ax1.set_ylim(0, 1)
#                 st.pyplot(fig1)

#                 # Chart 2: Income vs Loan Amount
#                 st.subheader("💰 Income vs Loan Amount")
#                 total_income = applicant_income + coapplicant_income
#                 fig2, ax2 = plt.subplots()
#                 ax2.bar(["Total Income", "Loan Amount (×1000)"], [total_income, loan_amount], color=["blue", "orange"])
#                 ax2.set_ylabel("Amount")
#                 ax2.set_title("Loan vs Income Comparison")
#                 st.pyplot(fig2)
#         else:
#             st.warning("No input data found. Please fill out the form on the left first.")

