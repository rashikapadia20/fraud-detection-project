import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("Fraud Detection App")

revenue = st.number_input("Revenue", 0.0)
valuation = st.number_input("Valuation", 0.0)
growth = st.number_input("Growth %", 0.0)
linkedin = st.slider("LinkedIn %", 0, 100)
web = st.slider("Web Reputation", 0, 100)
integrity = st.slider("Data Integrity", 0, 100)

if st.button("Predict"):
    data = np.array([[revenue, valuation, growth, linkedin, web, integrity]])
    result = model.predict(data)[0]

    if result == 1:
        st.error("Fraud ❌")
    else:
        st.success("Not Fraud ✅")
