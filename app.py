import streamlit as st
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Salary Predictor", page_icon=":money_with_wings:", layout="centered")

# Enhanced, more visible result box style
st.markdown(
    """
    <style>
    .result-box {
        padding: 20px 32px;
        background-color: #007bff; /* bright blue background */
        border-radius: 8px;
        font-size: 24px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 0 12px #0056b3;
        color: white; /* white font for contrast */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="big-font">💰 Salary Prediction App 💰</div>', unsafe_allow_html=True)
st.markdown('<div class="small-font">Predict your estimated salary based on your workplace attributes.</div>', unsafe_allow_html=True)
st.divider()

# Input layout
col1, col2, col3 = st.columns(3)
with col1:
    years_at_company = st.slider(
        "Years at Company",
        min_value=0,
        max_value=20,
        value=2,
        help="Input your years of experience at the company."
    )
with col2:
    satisfaction_level = st.slider(
        "Satisfaction Level (0.0 to 1.0)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Your current job satisfaction level."
    )
with col3:
    average_monthly_hours = st.number_input(
        "Average Monthly Hours",
        min_value=120,
        max_value=320,
        value=160,
        help="Your average working hours per month."
    )

X = [years_at_company, satisfaction_level, average_monthly_hours]

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

predict_button = st.button("🧮 Predict Salary")

st.divider()

if predict_button:
    X1 = np.array(X)
    X_array = scaler.transform([X1])
    prediction = model.predict(X_array)
    # Visible, high-contrast result box (no balloons)
    st.markdown(
        f'<div class="result-box">Estimated Salary: <b>₹ {prediction[0]:,.2f}</b></div>',
        unsafe_allow_html=True
    )
else:
    st.info("Enter your details, then click 'Predict Salary'.")
