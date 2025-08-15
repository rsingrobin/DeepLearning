import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import warnings
from warnings import filterwarnings

model = tf.keras.models.load_model('model.h5')

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

st.title("Bank Customer Churn Prediction")

st.write("Please enter the following details:")

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0].tolist())
gender = st.selectbox('Gender',label_encoder_gender.classes_.tolist())
age = st.slider('Age', min_value=18, max_value=90, value=30)
balance = st.number_input('Balance', min_value=0.0, max_value=1000000.0, value=50000.0)
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, max_value=150000.0, value=50000.0)
tenure = st.slider('Tenure', min_value=0, max_value=10, value=5)
num_of_products = st.slider('Number of Products', min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    st.write("The customer is likely to leave the bank.")
else:
    st.write("The customer is likely to stay with the bank.")