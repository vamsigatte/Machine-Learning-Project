# predict function will process input data and return premium predicted value
import pandas as pd
import joblib
import os

# model_young = joblib.load("model_young.joblib")
# model_rest = joblib.load("model_rest.joblib")
# scaler_young = joblib.load("scaler_young.joblib")
# scaler_rest = joblib.load("scaler_rest.joblib")


# Get base directory (Machine-Learning-Project)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

model_young = joblib.load(os.path.join(ARTIFACTS_DIR, "model_young.joblib"))
model_rest = joblib.load(os.path.join(ARTIFACTS_DIR, "model_rest.joblib"))
scaler_young = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler_young.joblib"))
scaler_rest = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler_rest.joblib"))

def preprocess_input(input_dict):
    # columns list for preprocessing
    # These columns were present just before the feature scaling 
    df_columns = [
   'age', 'region', 'number_of_dependants', 'bmi_category', 'income_level',
       'income_lakhs', 'insurance_plan', 'genetical_risk', 'total_risk_score',
       'gender_Male', 'marital_status_Unmarried', 'smoking_status_Occasional',
       'smoking_status_Regular', 'employment_status_Salaried',
       'employment_status_Self-Employed']
    
    # feature scaling

    # creating a DataFrame pre-filled with zeros
    # index = [0] specify how many rows you want, here only 0th row
    df = pd.DataFrame(0, columns = df_columns, index=[0])

    
    # Numerical direct assignments
    df['age'] = input_dict['age']
    df['number_of_dependants'] = input_dict['number_of_dependents']
    df['income_lakhs'] = input_dict['income_lakhs']
    df['genetical_risk'] = input_dict['genetical_risk']
    df['total_risk_score'] = input_dict['total_risk_score']

    
    # Categorical direct assignments (Ordinal and Label encoding)
    region_encoding = {'Northeast': 0,  'Northwest': 1, 'Southeast':2, 'Southwest':3}
    bmi_category_encoding = {'Overweight':0, 'Underweight':1, 'Normal':2, 'Obesity':3}
    insurance_plan_encoding = {'Bronze': 0,'Silver': 1,  'Gold': 2}

    df['region'] = region_encoding.get(input_dict['region'], 1)
    df['bmi_category'] = insurance_plan_encoding.get(input_dict['bmi_category'], 1)
    df['insurance_plan'] = insurance_plan_encoding.get(input_dict['insurance_plan'], 1)
    

    # One-hot encoding assignments (expanded if-else)
    if input_dict['gender'] == 'Male':
        df['gender_Male'] = 1
    if input_dict['marital_status'] == 'Unmarried':
        df['marital_status_Unmarried'] = 1
    if input_dict['smoking_status'] == 'Occasional':
        df['smoking_status_Occasional'] = 1
    if input_dict['smoking_status'] == 'Regular':
        df['smoking_status_Regular'] = 1
    if input_dict['employment_status'] == 'Salaried':
        df['employment_status_Salaried'] = 1
    if input_dict['employment_status'] == 'Self-Employed':
        df['employment_status_Self-Employed'] = 1
    
    
    df = handle_scaling(input_dict['age'], df)

    return df

def handle_scaling(age, df):
    # scale age and income_lakhs column
    if age <= 25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    # Leave income_level blank
    df['income_level'] = None

    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    df.drop('income_level', axis='columns', inplace=True)

    return df

def predict(input_dict):
    processed_data = preprocess_input(input_dict)

    if input_dict['age'] <=25:
        final_premium = model_young.predict(processed_data)
    else:
        final_premium = model_rest.predict(processed_data)

    return final_premium