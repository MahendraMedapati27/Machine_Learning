import pickle
import streamlit as st
import numpy as np

scalar = pickle.load(open(r'C:\Users\Hp\OneDrive\Desktop\ML\Diabetics_Prediction\scaler.pkl', 'rb'))
model = pickle.load(open(r'C:\Users\Hp\OneDrive\Desktop\ML\Diabetics_Prediction\best_model.pkl', 'rb'))

st.title('Diabetes Prediction')

Pregnancies = st.slider('Select value for no of Pregnanacies', min_value=0, max_value=100, value=0)
Glucose = st.slider('Select the value of Glucose level', min_value=0, max_value=300, value=0)
BloodPressure = st.slider('Select the value of your Blood Pressure', min_value=0, max_value=200, value=0)
SkinThickness = st.slider('Select the value of your Skin Thickness', min_value=0, max_value=100, value=0)
Insulin = st.slider('Select the value of your Insulin level', min_value=0, max_value=1000, value=0)
BMI = st.slider('Select your BMI value', min_value=0, max_value=100, value=0)
DiabetesPedigreeFunction= st.slider('Select the value of your DiabetesPedigreeFunction', min_value=0, max_value=50, value=0)
Age = st.slider('Select your Age', min_value=0, max_value=100, value=0)
new_glucose = Glucose*Glucose*Glucose

input_data = np.array([[new_glucose, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

if st.button('Predict'):
    # Ensure all values are received (optional check)
    if np.any(input_data == None):
        st.write('Please provide values for all features.')
    else:
        # Scale the input data using the same scaler used for training
        input_data_scaled = scalar.transform(input_data)

        # Predict diabetes using the input data
        prediction = model.predict(input_data_scaled)
        prediction_prob = model.predict_proba(input_data_scaled)

        # Display the prediction result
        st.write('Prediction (0 = No Diabetes, 1 = Diabetes):', prediction[0])
        st.write('Prediction Probability:', prediction_prob[0])
