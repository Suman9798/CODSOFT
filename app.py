import streamlit as st

import numpy as np
import pickle 

with open("irisflowerdtabase.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Iris Flower Prediction")

sepal_length = st.slider("sepal length(cm)",4.0,8.0)
sepal_width = st.slider("sepal width(cm)",2.0,5.0)
petal_length = st.slider("petal length(cm)",1.0,7.0)
petal_width = st.slider("petal width(cm)",0.1,2.5)

if st.button("Prediction"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Iris species: {prediction[0]}")
