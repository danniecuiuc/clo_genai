from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import MODEL_BUNDLE_PATH, NUMERIC_FEATURES
from src.data.loaders import load_dataframe
from src.pipeline.inference import predict_existing_id, predict_new_record

st.set_page_config(page_title="CLO Tranche Pricing MVP", layout="wide")
st.title("CLO Tranche Pricing MVP")
st.caption("Manual input or Excel upload → preprocessing → comps → pricing → uncertainty")

mode = st.sidebar.radio("Input mode", ["Existing Bloomberg ID", "Manual form", "Excel upload"])
k = st.sidebar.slider("Number of comps", 3, 20, 10)

if mode == "Existing Bloomberg ID":
    target_id = st.text_input("Bloomberg ID", value="GLM 2022-12A ER")
    if st.button("Run pricing"):
        result = predict_existing_id(target_id, MODEL_BUNDLE_PATH, k=k)
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Spread", f"{result.prediction:.2f}")
        c2.metric("Lower Bound", f"{result.interval_low:.2f}")
        c3.metric("Upper Bound", f"{result.interval_high:.2f}")
        st.subheader("Comparable Deals")
        st.dataframe(pd.DataFrame(result.comparable_deals))
        st.subheader("Top Feature Importance")
        st.dataframe(pd.DataFrame(result.top_features))

elif mode == "Manual form":
    payload = {}
    for col in NUMERIC_FEATURES:
        payload[col] = st.number_input(col, value=0.0, format="%.6f")
    payload["Collateral manager"] = st.text_input("Collateral manager", value="Apollo")
    payload["Trade Date"] = st.text_input("Trade Date", value="2026-02-12")
    payload["Closing Date"] = st.text_input("Closing Date", value="2025-03-06")
    if st.button("Run pricing"):
        df = pd.DataFrame([payload])
        result = predict_new_record(df, MODEL_BUNDLE_PATH, k=k)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Spread", f"{result.prediction:.2f}")
        c2.metric("Lower Bound", f"{result.interval_low:.2f}")
        c3.metric("Upper Bound", f"{result.interval_high:.2f}")
        c4.metric("Uncertainty", result.uncertainty)
        st.subheader("Comparable Deals")
        st.dataframe(pd.DataFrame(result.comparable_deals))
        st.subheader("Top Feature Importance")
        st.dataframe(pd.DataFrame(result.top_features))

else:
    uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv", "parquet"])
    if uploaded is not None:
        temp_path = f"/tmp/{uploaded.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        df = load_dataframe(temp_path)
        st.write("Preview")
        st.dataframe(df.head())
        if st.button("Run pricing"):
            result = predict_new_record(df.head(1), MODEL_BUNDLE_PATH, k=k)
            st.metric("Predicted Spread", f"{result.prediction:.2f}")
            st.write(f"Range: [{result.interval_low:.2f}, {result.interval_high:.2f}] | Uncertainty: {result.uncertainty}")
            st.subheader("Comparable Deals")
            st.dataframe(pd.DataFrame(result.comparable_deals))
            st.subheader("Top Feature Importance")
            st.dataframe(pd.DataFrame(result.top_features))
