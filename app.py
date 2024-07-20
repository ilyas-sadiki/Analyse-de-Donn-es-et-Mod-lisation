import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Charger le modèle de clustering
model = pickle.load(open('model.pkl', 'rb'))

scaler = pickle.load(open('scaler.pkl', 'rb'))  

def predict_cluster(features):
    # Standardiser les features
    features_scaled = scaler.transform([features])
    return model.predict(features_scaled)[0]

def main():
    html_temp = """
    <style>
    .appview-container .main .block-container{{
        max-width: 700px;
    }}
    .reportview-container .main {{
        color: white;
        background-color: grey;
    }}
    </style>
    <div style="background-color:teal ;padding:10px">
    <h3 style="font-size:18pt;color:white;text-align:center;">Customer Clustering</h3>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    st.sidebar.header('Entrer les caractéristiques du client')
    
    BALANCE = st.sidebar.slider('BALANCE', 0.0, 1990.482788)
    PURCHASES = st.sidebar.slider('PURCHASES', 0.0, 843.34)
    ONEOFF_PURCHASES = st.sidebar.slider('ONEOFF_PURCHASES', 0.0, 95.0)
    INSTALLMENTS_PURCHASES = st.sidebar.slider('INSTALLMENTS_PURCHASES', 0.0, 222.44)
    PURCHASES_FREQUENCY = st.sidebar.slider('PURCHASES_FREQUENCY', 0.0, 1.0)
    ONEOFF_PURCHASES_FREQUENCY = st.sidebar.slider('ONEOFF_PURCHASES_FREQUENCY', 0.0, 0.2)
    PURCHASES_INSTALLMENTS_FREQUENCY = st.sidebar.slider('PURCHASES_INSTALLMENTS_FREQUENCY', 0.0, 1.0)
    CASH_ADVANCE_FREQUENCY = st.sidebar.slider('CASH_ADVANCE_FREQUENCY', 0.0, 1.0)
    PURCHASES_TRX = st.sidebar.slider('PURCHASES_TRX', 0.0, 21.0)
    CREDIT_LIMIT = st.sidebar.slider('CREDIT_LIMIT', 0.0, 12500.0)
    PAYMENTS = st.sidebar.slider('PAYMENTS', 0.0, 1567.278435)
    MINIMUM_PAYMENTS = st.sidebar.slider('MINIMUM_PAYMENTS', 0.0, 524.499084)
    
    inputs = np.array([BALANCE, PURCHASES, ONEOFF_PURCHASES, INSTALLMENTS_PURCHASES, 
                       PURCHASES_FREQUENCY, ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY, 
                       CASH_ADVANCE_FREQUENCY, PURCHASES_TRX, CREDIT_LIMIT, PAYMENTS, 
                       MINIMUM_PAYMENTS])
    
    if st.button('Prédire le cluster'):
        # Standardiser les données d'entrée
        inputs_scaled = scaler.transform([inputs])
        cluster = predict_cluster(inputs_scaled[0])
        st.success(f"Le client appartient au cluster : {cluster}")

if __name__ == '__main__':
    main()
