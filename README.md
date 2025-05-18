import streamlit as st
import pandas as pd
import joblib
from utils import preprocess_input
model = joblib.load("model.joblib")
st.title("🎬 Movie Matchmaker")
st.markdown("Enter user & movie info to get a predicted rating!")
with st.form("input_form"):
    age = st.number_input("User Age", min_value=1, max_value=120, value=30)
    income = st.number_input("User Income", min_value=0, value=50000)
    hours_since_rating = st.number_input("Hours Since Last Rating", min_value=0, value=24)
    submitted = st.form_submit_button("Predict Rating")
if submitted:
    raw = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "hours_since_rating": hours_since_rating,
    }])
    X = preprocess_input(raw)
    pred = model.predict(X)[0]
    st.success(f"🔮 Predicted Rating: **{pred:.2f} / 5**")
# naan-mudhalvan
