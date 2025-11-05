import streamlit as st
import pandas as pd
import numpy as np
from apputil import predict_rating

st.title("☕ Coffee Rating Predictor")

mode = st.radio("Choose mode:", ["Tabular", "Text"], horizontal=True)

if mode == "Tabular":
    df = st.data_editor(
        pd.DataFrame([{"100g_USD": 10.0, "roast": "medium"}]),
        num_rows="dynamic",
        use_container_width=True
    )
    if st.button("Predict"):
        y_pred = predict_rating(df)
        st.write("Predicted Ratings:", y_pred)

else:
    df = st.data_editor(
        pd.DataFrame([{"text": "A delightful coffee with bold flavor."}]),
        num_rows="dynamic",
        use_container_width=True
    )
    if st.button("Predict from Text"):
        y_pred = predict_rating(df, text=True)
        st.write("Predicted Ratings:", y_pred)

