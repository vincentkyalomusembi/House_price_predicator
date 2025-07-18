import streamlit as st
import numpy as np
import matplotlib as plt
import seaborn as sns
import pickle

# Load the trained model
with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("House Price Predictor")    

# User inputs
area = st.number_input("Area (in square feet)", min_value=300, max_value=10000)
bedroom = st.slider("Number of bedrooms", 1, 10, step = 1)

#Prediction
if st.button("Predict Price"):
    features = np.array([[area, bedroom]])
    prediction = model.predict(features)
    st.success(f"Estimated Price: ${prediction[0]:,.2f}")