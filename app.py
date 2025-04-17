import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('lr_model.pkl')
scaler = joblib.load('scaler_model.pkl')

st.sidebar.title("Diabetes Prediction App")

# Create two columns
left_col, right_col = st.columns(2)

# Input sliders in left column
with left_col:
    Pregnancies = st.sidebar.slider("Pregnancies", min_value=0, max_value=15, value=5)
    Glucose = st.sidebar.slider('Glucose', min_value=0, max_value=199, value=100)
    BloodPressure = st.sidebar.slider('BloodPressure', min_value=0, max_value=122, value=64)
    SkinThickness = st.sidebar.slider('SkinThickness', min_value=0, max_value=99, value=50)
    Insulin = st.sidebar.slider('Insulin', min_value=0, max_value=846, value=400)
    BMI = st.sidebar.slider('BMI', min_value=0.0, max_value=67.1, value=37.0, step=0.1)
    DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction', min_value=0.078, max_value=2.420, step=0.001)
    Age = st.sidebar.slider('Age', min_value=21, max_value=81, value=40, step=1)

# Prediction output in right column
with right_col:
    if st.sidebar.button("Predict"):
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                Insulin, BMI, DiabetesPedigreeFunction, Age]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        result = "⚠️ Take sugar in limit – You have **DIABETES**" if prediction == 1 else "🎉 Congratulations – You're **Diabetes-Free!**"
        st.sidebar.success(result)

