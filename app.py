import streamlit as st
import numpy as np
import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load or create model
if not os.path.exists("model.pkl"):
    df = pd.read_csv("fraud_dataset_1000.csv")
    X = df.drop("fraud", axis=1)
    y = df["fraud"]

    model = RandomForestClassifier()
    model.fit(X, y)

    pickle.dump(model, open("model.pkl", "wb"))
else:
    model = pickle.load(open("model.pkl", "rb"))

# UI
st.title("🚨 Fraud Detection App")

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
        st.error("⚠️ Fraud Detected")
    else:
        st.success("✅ Not Fraud")
