# frontend

import streamlit as st
from prediction_helper import predict

st.title("Health Insurance Premium Prediction")

# In drop down menu, picking values from this dictionary
categorical_options = {
        "gender" : ["Male", "Female"],
        "region" : ["Northeast", "Northwest", "Southeast", "Southwest"],
        "marital_status" : ["Unmarried", "Married"],
        "bmi_category" : ["Overweight", "Underweight", "Normal", "Obesity"],
        "smoking_status" : ["Regular", "No Smoking", "Occasional"],
        "employment_status"  :  ['Self-Employed', 'Freelancer', 'Salaried'],
        "income_level"  :  ['> 40L', '<10L', '10L - 25L', '25L - 40L'],
        "insurance_plan"  :  ['Silver', 'Bronze', 'Gold']
}

row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)

with row1[0]:
    age = st.number_input('Age', min_value = 18, max_value = 100, step = 1 )
with row1[1]:
    region = st.selectbox('Region', categorical_options["region"])
with row1[2]:
    number_of_dependents = st.number_input('Number of dependents', min_value = 0, max_value = 20, step =1 )

with row2[0]:
    bmi_category = st.selectbox('BMI category', categorical_options["bmi_category"])
with row2[1]:
    income_lakhs = st.number_input('Income in Lakhs', min_value = 0, max_value = 200, step =1 )
with row2[2]:
    insurance_plan = st.selectbox('Insurance Plan', categorical_options["insurance_plan"])

with row3[0]:
    genetical_risk = st.number_input('Genetical Risk', min_value = 0, max_value = 5, step = 1 )
with row3[1]:
    total_risk_score = st.number_input('Total Risk Score', min_value = 0, max_value = 20, step = 1 )
with row3[2]:
    gender = st.selectbox('Gender', categorical_options["gender"])

with row4[0]:
    marital_status = st.selectbox('Marital Status', categorical_options["marital_status"])
with row4[1]:
    smoking_status = st.selectbox('Smoking status', categorical_options["smoking_status"])
with row4[2]:
    employment_status = st.selectbox(' Employment Status', categorical_options["employment_status"])


# Creating a dictionary to store all input values together
input_dict = {
    'age' : age,
    'region' : region,
    'number_of_dependents' : number_of_dependents,
    'bmi_category' : bmi_category,
    'income_lakhs' : income_lakhs,
    'insurance_plan' : insurance_plan,
    'genetical_risk' : genetical_risk,
    'total_risk_score' : total_risk_score,
    'gender' : gender,
    'marital_status' : marital_status,
    'smoking_status' : smoking_status,
    'employment_status' : employment_status
}

if st.button('Predict'):
    prediction = predict(input_dict)
    st.success(f"Predicted Health insurance premium: {prediction}")