"""
Streamlit UI backed by pricing/clo_pricing.py.

This file intentionally does not use the older modular pricing/synthetic/similarity
pipeline. `pricing.clo_pricing` is the single source of truth for:
- feature list
- target normalization
- synthetic data generation
- pricing model
- KNN comparable-deal similarity
- model bundle format

Run from project root:
    python -m streamlit run app/ui.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import MODEL_BUNDLE_PATH
from pricing.clo_pricing import (
    ALL_FEATURES,
    TARGET,
    TARGET_FALLBACKS,
    generate_synthetic,
    predict_from_bundle,
    train_and_save_bundle,
)


def read_uploaded_file(uploaded_file, header_row: int) -> pd.DataFrame:
    """Read CSV/Excel and support the CLO workbook pattern with header in row 0."""
    if uploaded_file.name.lower().endswith(".csv"):
        raw = pd.read_csv(uploaded_file, header=header_row)
    else:
        raw = pd.read_excel(uploaded_file, header=header_row)

    raw.columns = [str(c).strip() for c in raw.columns]
    return raw


def normalize_for_clo_pricing(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the minimum normalization expected by pricing.clo_pricing."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    if TARGET not in out.columns:
        for candidate in TARGET_FALLBACKS:
            if candidate in out.columns:
                out[TARGET] = out[candidate]
                break

    if TARGET not in out.columns:
        raise ValueError(f"Missing target column '{TARGET}'. Tried fallbacks: {TARGET_FALLBACKS}")

    missing_features = [col for col in ALL_FEATURES if col not in out.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    for col in ALL_FEATURES + [TARGET]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[out[TARGET].notna()].reset_index(drop=True)
    return out


def display_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Make nested/object columns safe for Streamlit display."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].apply(
                lambda x: json.dumps(x, default=str) if isinstance(x, (list, tuple, dict)) else x
            )
    return out


def preview(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    return df.head(n).copy()


st.set_page_config(page_title="CLO Pricing Demo", layout="wide")
st.title("CLO Pricing + Similarity Demo")
st.caption("Single source of truth: pricing/clo_pricing.py")

with st.sidebar:
    st.header("1. Upload")
    uploaded = st.file_uploader("Excel / CSV", type=["xlsx", "xls", "csv"])
    header_row = st.number_input("Header row index", min_value=0, max_value=10, value=0, step=1)

if uploaded is None:
    st.info("Upload a file from the left sidebar to start.")
    st.stop()

try:
    raw_df = read_uploaded_file(uploaded, int(header_row))
    clean_df = normalize_for_clo_pricing(raw_df)
except Exception as exc:
    st.error(f"Failed to load/normalize data: {exc}")
    st.stop()

if clean_df.empty:
    st.error("No usable rows after normalization.")
    st.stop()

id_candidates = ["Bloomberg ID", "BBG ID", "ID", "Deal ID", "Tranche ID", "Security ID"]
id_col_default = next((c for c in id_candidates if c in clean_df.columns), None)

with st.sidebar:
    st.header("2. Bundle")
    bundle_path = st.text_input("Bundle path", value=str(MODEL_BUNDLE_PATH))
    synth_rows = st.slider("Synthetic rows", min_value=0, max_value=5000, value=500, step=100)
    similarity_real_only = st.checkbox("Similarity on real rows only", value=True)
    train_save = st.button("Train + Save Model Bundle", type="primary", use_container_width=True)

    st.header("3. Query Row")
    query_mode_options = ["By row index"] + (["By ID"] if id_col_default else [])
    query_mode = st.radio("Query mode", query_mode_options)

    if query_mode == "By ID":
        id_values = clean_df[id_col_default].dropna().astype(str).unique().tolist()
        selected_id = st.selectbox("Query ID", id_values)
        query_index = int(clean_df.index[clean_df[id_col_default].astype(str) == str(selected_id)][0])
    else:
        query_index = int(
            st.number_input(
                "Query row index",
                min_value=0,
                max_value=max(len(clean_df) - 1, 0),
                value=0,
            )
        )

    load_price = st.button("Load Bundle + Price Query Row", use_container_width=True)

query_row = clean_df.iloc[[query_index]].copy()

with st.expander("Required model features", expanded=False):
    st.write(ALL_FEATURES)

preview_tab, query_tab, synth_tab = st.tabs(["Preview", "Selected Query", "Synthetic Sample"])
with preview_tab:
    st.subheader("Raw Data")
    st.dataframe(preview(raw_df), use_container_width=True)
    st.subheader("Normalized Data Used by clo_pricing.py")
    st.dataframe(preview(clean_df), use_container_width=True)

with query_tab:
    st.subheader("Selected Query Row")
    st.dataframe(query_row, use_container_width=True)

with synth_tab:
    st.subheader("Synthetic Data Preview")
    try:
        synthetic_preview = generate_synthetic(clean_df, n_rows=min(20, max(int(synth_rows), 0)), method="bootstrap")
        st.dataframe(preview(synthetic_preview), use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not generate synthetic preview: {exc}")

if train_save:
    try:
        result = train_and_save_bundle(
            df=clean_df,
            bundle_path=bundle_path,
            synth_rows=int(synth_rows),
            similarity_on_real_only=similarity_real_only,
        )
        st.success(f"Bundle saved to {result['bundle_path']}")
        st.json(result)
    except Exception as exc:
        st.error(f"Training failed: {exc}")

if load_price:
    bundle_file = Path(bundle_path)
    if not bundle_file.exists():
        st.error(f"Bundle not found at {bundle_path}. Train and save first.")
    else:
        try:
            prediction = predict_from_bundle(query_row=query_row, bundle_path=bundle_path)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Predicted Price", prediction["point_estimate"])
            c2.metric("Lower Bound", prediction["lower_bound"])
            c3.metric("Upper Bound", prediction["upper_bound"])
            c4.metric("Ridge Baseline", prediction.get("baseline_estimate", "n/a"))

            st.subheader("Feature Importance")
            st.dataframe(display_safe(prediction["feature_importance"]), use_container_width=True)

            st.subheader("Comparable Deals")
            comparables = pd.DataFrame(prediction["comparables"])
            st.dataframe(display_safe(comparables), use_container_width=True)

            st.download_button(
                "Download Prediction JSON",
                data=json.dumps(prediction, indent=2, default=str),
                file_name="clo_prediction.json",
                mime="application/json",
            )
        except Exception as exc:
            st.error(f"Pricing failed: {exc}")
