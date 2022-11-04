import pickle as pkl
import streamlit as st
import pandas as pd

# Take inputs from user
st.image('https://nextommerce.com/wp-content/uploads/2019/04/pasted-image-0.png')
tenure = st.number_input("Tenure", 1, 1000)
monthlycharges = st.slider("Monthly Charges", 10, 1000)
totalcharges = st.slider("Total Charges", monthlycharges, 10000)
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
onlinesecurity = st.selectbox('Online security', ['No', 'Yes', 'No internet service'])
techsupport = st.selectbox('Tech support', ['No', 'Yes', 'No internet service'])
internetservice = st.selectbox('Internet service', ['DSL', 'Fiber optic', 'No'])
onlinebackup = st.selectbox('Online backup', ['No', 'Yes', 'No internet service'])

# Convert inputs to DataFrame
x = {'contract': contract, 'onlinesecurity': onlinesecurity, 'techsupport': techsupport, 'internetservice': internetservice, 'onlinebackup': onlinebackup,
'tenure': tenure,'monthlycharges': monthlycharges, 'totalcharges': totalcharges }

x_df = pd.DataFrame(x, index=[0])

# Load the transformer
transformer = pkl.load(open('transformer.pkl', 'rb'))

# Apply the transformer on the inputs
X_new = transformer.transform(x_df)

# Load the model
model = pkl.load(open('log_reg.pkl', 'rb'))

# Predict the output
churn_prop = model.predict_proba(X_new)[0][1] * 100
st.markdown(f'## Probability of churn: {round(churn_prop, 2)} %')
