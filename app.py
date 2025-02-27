#Importing Necessary libraries
import pandas as pd
import numpy as np
import pickle
import streamlit as st

with open('first_iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Page customization
st.set_page_config(page_title="Iris Flower Prediction App", page_icon="🌸")

#Title
st.title("Iris FLower prediction App")
st.write("This app predictis the **Iris flower** type!")
st.write("Please input the the following parameters:")

sepal_width = st.number_input("Sepal Width", min_value = 0.1, max_value = 10.0, value = 3.4, step=0.1)
sepal_length = st.number_input("Sepal Length", min_value = 0.1, max_value = 10.0, value = 3.4, step=0.1)
petal_width = st.number_input("Petal Width", min_value = 0.1, max_value = 10.0, value = 3.4, step=0.1)
petal_length = st.number_input("Petal Length", min_value = 0.1, max_value = 10.0, value = 3.4, step=0.1)

#Predictions
if st.button("predict"):
    user_input = np.array([[sepal_width, sepal_length, petal_width, petal_length]])
    prediction = model.predict(user_input)
    species_mapping = {0:'setosa', 1:'Versicolor', 2:'Virginica'}
    predicted_species = species_mapping.get(int(prediction[0]), 'unknown')
    st.write(f"The predicted species is {predicted_species}")

#foorter 
st.write("This is a simple Iris Flower prediction app made with streamlit")





