import pandas as pd
import streamlit as st
import pickle
import numpy as np
MODEL_PATH = "final_model.pkl"
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open("normalizer.pkl", "rb") as file:
    normalizer = pickle.load(file)

st.set_option('deprecation.showPyplotGlobalUse', False)
if __name__ == "__main__":

    st.title("Combined-cycle Power Plant(CCPP) \n Energy Production Predictor")
    st.info('This app is created to predict Power Plant Energy Production')
    # Sidebar for user input
    st.sidebar.header('User Input Parameters')

    # Collect user inputs
    temperature = st.sidebar.slider('Temperature (°C)', 1.00, 39.00)
    exhaust_vacuum = st.sidebar.slider('Exhaust Vacuum (cm Hg)', 25.00, 85.00)
    amb_pressure = st.sidebar.slider('Ambient Pressure (millibar)', 950.00, 1100.00)
    r_humidity = st.sidebar.slider('Relative Humidity (%)', 0.00, 100.00)

    input_df=pd.DataFrame(columns=['temperature', 'exhaust_vacuum', 'amb_pressure', 'r_humidity'],dtype=int)
    data1=[temperature, exhaust_vacuum, amb_pressure, r_humidity]
    new_row = pd.DataFrame([data1],columns=['temperature', 'exhaust_vacuum', 'amb_pressure', 'r_humidity'],dtype=int)
    input_df = pd.concat([input_df, new_row], ignore_index=True)
    normalised=np.round(normalizer.transform(input_df),2)
    input_df=pd.DataFrame(normalised,columns=input_df.columns)
    prediction=model.predict(input_df)
# Display prediction
    st.header('Predicting Energy Production')
    st.write(f'The predicted energy production is: {prediction[0]:.2f} MW')
