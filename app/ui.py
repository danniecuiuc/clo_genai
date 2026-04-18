from __future__ import annotations

import pandas as pd
import streamlit as st

from common.config import MODEL_BUNDLE_PATH, NUMERIC_COLUMNS
from llm.memo import sample_local_memo
from product_pipeline.predict import predict_for_existing_id, predict_for_new_payload

st.set_page_config(page_title='CLO Tranche Pricing MVP', layout='wide')
st.title('CLO Tranche Pricing MVP')
st.caption('UI layer for pricing range, similar deals, feature importance, and uncertainty signal.')

mode = st.sidebar.radio('Choose workflow', ['Predict by Bloomberg ID', 'Predict by manual input'])
k = st.sidebar.slider('Number of similar deals', min_value=3, max_value=20, value=10)

if mode == 'Predict by Bloomberg ID':
    bloomberg_id = st.text_input('Bloomberg ID', value='RRAM 2025-37A D')
    if st.button('Run prediction by ID'):
        result = predict_for_existing_id(bloomberg_id=bloomberg_id, model_bundle_path=str(MODEL_BUNDLE_PATH), k=k)

        col1, col2, col3 = st.columns(3)
        col1.metric('Predicted Spread', f"{result.prediction:.2f}")
        col2.metric('Lower Bound', f"{result.lower_bound:.2f}")
        col3.metric('Upper Bound', f"{result.upper_bound:.2f}")
        st.write(f"**Uncertainty flag:** {result.uncertainty_flag}")
        st.write(f"**Best model:** {result.best_model_name}")
        st.subheader('Similar Deals')
        st.dataframe(pd.DataFrame(result.similar_deals))
        st.subheader('Top Feature Importance')
        st.dataframe(pd.DataFrame(result.top_feature_importance))
        st.subheader('Optional Memo')
        st.write(sample_local_memo({'Bloomberg ID': bloomberg_id}, result.__dict__, result.similar_deals))

else:
    st.write('Enter a minimal tranche payload. Fields not entered will be handled by the preprocessing pipeline.')
    payload = {}
    for col in NUMERIC_COLUMNS[:12]:
        payload[col] = st.number_input(col, value=0.0)
    payload['Collateral manager'] = st.text_input('Collateral manager', value='Apollo')

    if st.button('Run manual prediction'):
        result = predict_for_new_payload(pd.DataFrame([payload]), model_bundle_path=str(MODEL_BUNDLE_PATH), k=k)
        col1, col2, col3 = st.columns(3)
        col1.metric('Predicted Spread', f"{result.prediction:.2f}")
        col2.metric('Lower Bound', f"{result.lower_bound:.2f}")
        col3.metric('Upper Bound', f"{result.upper_bound:.2f}")
        st.write(f"**Uncertainty flag:** {result.uncertainty_flag}")
        st.write(f"**Best model:** {result.best_model_name}")
        st.subheader('Similar Deals')
        st.dataframe(pd.DataFrame(result.similar_deals))
        st.subheader('Top Feature Importance')
        st.dataframe(pd.DataFrame(result.top_feature_importance))
        st.subheader('Optional Memo')
        st.write(sample_local_memo(payload, result.__dict__, result.similar_deals))
