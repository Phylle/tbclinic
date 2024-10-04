import pandas as pd
import joblib
import streamlit as st

# Load the trained model
model = joblib.load('logistic_regression_model_pipeline.joblib')

st.title('TB Outcome Prediction')
st.header('Enter Patient Information')

# User inputs
hivstatus = st.selectbox('HIV Status', ['POS', 'NEG'])
weightbefore = st.number_input('Weight Before Treatment (kg)', min_value=0.0)
age = st.number_input('Age (years)', min_value=0)
height = st.number_input('Height (cm)', min_value=0)
physicaladdress = st.selectbox('Physical Address', [
    'URUDI', 'NYAKONGO', 'OLUWA PRI', 'ODOWA', 'NYAKWERE', 
    'SANGANYINYA PRI', 'KAWATER SUPPLY', 'KIGADAI', 'SINYOLO', 
    'MARIERA PRI', 'ROTA', 'KANYAWEGI', 'WATHOREGO', 'KUOYO PRI', 
    'ULALO', 'RABUOR', 'LISUKA PRI', 'KONAKAYONA', 'HUMA GIRLS', 
    'NYAHERA', 'NYALENDA', 'VIHIGA', 'KIBOS', 'ELUHOBE'
])
tbtype = st.selectbox('TB Type', ['PTB', 'EPTB'])
typeofpatient = st.selectbox('Type of Patient', ['N', 'R'])
datetreatmentstarted = st.date_input('Date of Treatment Started')
sex = st.selectbox('Sex', ['M', 'F'])
riskfactors = st.selectbox('Risk Factors', [
    'NONE', 'MAM', 'SAM', 'CAVALVA', 'SCHIZOPHRENIA', 
    'ALCOHOL', 'MALNUTRITION', 'BIPOLAR,ALCOHOL', 
    'ALCOHOL/SMOKING', 'LIVER DISEASE', 
    'CRYPTOCOCCALLLL MENINGITIS', 'OVARIAN CANCER'
])

# Create a DataFrame with the user input
new_data = pd.DataFrame({
    'weightbefore': [weightbefore],
    'age': [age],
    'height': [height],
    'physicaladdress': [physicaladdress],
    'hivstatus': [hivstatus],
    'tbtype': [tbtype],
    'typeofpatient': [typeofpatient],
    'datetreatmentstarted': [datetreatmentstarted],
    'sex': [sex],
    'riskfactors': [riskfactors]
})

if st.button('Predict'):
    # Make the prediction
    prediction = model.predict(new_data)
    
    # Output the prediction directly
    st.write(f'Prediction: {prediction[0]}')
