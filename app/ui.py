"""
Streamlit UI for the CLO Pricing + Similarity MVP.

Integrated from the second project zip, but adapted to this repo's package layout:
- pricing uses synthetic-augmented data and model candidates from pricing.models
- similarity uses the processed real data only
- clo_pricing.py is intentionally not imported or modified

Run from the repo root:
    streamlit run app/ui.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from common.config import ROW_SOURCE_COL
from features.builder import build_feature_frame
from preprocessing.cleaning import clean_raw_dataframe
from preprocessing.pipeline import build_preprocessor, get_transformed_feature_names, prepare_dataset
from pricing.explainability import build_feature_importance_table
from pricing.models import train_candidate_models
from pricing.uncertainty import prediction_interval
from similarity.knn_engine import find_neighbors_for_existing_row
from synthetic.bootstrap_generator import combine_real_and_synthetic

ID_CANDIDATES = ['Bloomberg ID', 'BBG ID', 'ID', 'Deal ID', 'Tranche ID', 'Security ID']
TARGET_CANDIDATES = ['Spread', 'Cover Price', 'Price', 'OAS', 'DM', 'Target']
DEFAULT_PREVIEW_ROWS = 50


def read_uploaded_file(uploaded_file, header_row: int) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith('.csv'):
        return pd.read_csv(uploaded_file, header=header_row)
    return pd.read_excel(uploaded_file, header=header_row)


def guess_column(columns: list[str], candidates: list[str], fallback: int = 0) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return columns[min(fallback, len(columns) - 1)]


def preview(df: pd.DataFrame, n: int = DEFAULT_PREVIEW_ROWS) -> pd.DataFrame:
    return df.head(n).copy()


def display_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert nested objects so Streamlit's dataframe renderer does not fail."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == 'object':
            out[col] = out[col].apply(
                lambda x: json.dumps(x, default=str) if isinstance(x, (list, tuple, dict)) else x
            )
    return out


def train_and_price(
    pricing_train_df: pd.DataFrame,
    query_row: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
) -> dict:
    """Train candidate pricing models and price the selected query row."""
    prepared = prepare_dataset(pricing_train_df, target_col=target_col)
    x_raw = prepared.features.copy()
    y = pd.to_numeric(prepared.target, errors='coerce') if prepared.target is not None else None

    if y is None:
        raise ValueError(f'Target column not found: {target_col}')

    valid_mask = y.notna()
    x_raw = x_raw.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    if len(x_raw) < 5:
        raise ValueError('Not enough rows after cleaning/preprocessing to train a pricing model.')

    x_train_raw, x_valid_raw, y_train, y_valid = train_test_split(
        x_raw,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(x_train_raw)
    x_train = preprocessor.fit_transform(x_train_raw)
    x_valid = preprocessor.transform(x_valid_raw)
    feature_names = get_transformed_feature_names(preprocessor)

    outputs = train_candidate_models(
        X_train=x_train,
        y_train=y_train,
        X_valid=x_valid,
        y_valid=y_valid,
        random_state=random_state,
    )
    best_model = outputs.fitted_models[outputs.best_model_name]

    query_features = build_feature_frame(query_row)
    query_x_raw = query_features.reindex(columns=x_raw.columns, fill_value=np.nan)
    query_x = preprocessor.transform(query_x_raw)

    point_estimate = float(best_model.predict(query_x)[0])
    lower_bound, upper_bound = prediction_interval(point_estimate, outputs.residual_std)
    importance = build_feature_importance_table(best_model, feature_names)

    return {
        'best_model': outputs.best_model_name,
        'point_estimate': point_estimate,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'residual_std': outputs.residual_std,
        'metrics': outputs.metrics,
        'validation_predictions': outputs.validation_predictions,
        'feature_importance': importance,
    }


def similarity_from_processed_real(
    clean_df: pd.DataFrame,
    query_index: int,
    id_col: str,
    k: int,
    target_col: str,
):
    """Find comparable real deals using the same feature pipeline as pricing."""
    prepared_real = prepare_dataset(clean_df, target_col=target_col)
    x_real_raw = prepared_real.features.copy()

    sim_preprocessor = build_preprocessor(x_real_raw)
    x_real = sim_preprocessor.fit_transform(x_real_raw)
    feature_names = get_transformed_feature_names(sim_preprocessor)

    outputs = find_neighbors_for_existing_row(
        reference_df=clean_df,
        transformed_reference=x_real,
        transformed_feature_names=feature_names,
        target_row_idx=query_index,
        k=min(k, len(clean_df)),
    )

    ranked_clean = outputs.ranked_clean.copy()
    if id_col in outputs.ranked_raw.columns and id_col not in ranked_clean.columns:
        ranked_clean.insert(0, id_col, outputs.ranked_raw[id_col].values)

    return outputs.ranked_raw.reset_index(drop=True), ranked_clean.reset_index(drop=True), prepared_real


st.set_page_config(page_title='CLO Pricing + Similarity Demo', layout='wide')
st.title('CLO Pricing + Similarity Demo')
st.caption('Upload data → clean/preprocess → similarity on real rows → pricing with synthetic-augmented training data')

with st.sidebar:
    st.header('1. Upload')
    uploaded = st.file_uploader('Excel / CSV', type=['xlsx', 'xls', 'csv'])
    header_row = st.number_input('Header row index', min_value=0, max_value=10, value=0, step=1)

if uploaded is None:
    st.info('Upload a file from the sidebar to start.')
    st.stop()

try:
    raw_df = read_uploaded_file(uploaded, int(header_row))
    raw_df.columns = [str(col).strip() for col in raw_df.columns]
except Exception as exc:
    st.error(f'Failed to read file: {exc}')
    st.stop()

if raw_df.empty:
    st.error('Uploaded file is empty.')
    st.stop()

columns = raw_df.columns.tolist()

with st.sidebar:
    st.header('2. Columns')
    id_default = guess_column(columns, ID_CANDIDATES, 0)
    target_default = guess_column(columns, TARGET_CANDIDATES, 1 if len(columns) > 1 else 0)

    id_col = st.selectbox('ID column', columns, index=columns.index(id_default))
    target_col = st.selectbox('Target column', columns, index=columns.index(target_default))

try:
    clean_df = clean_raw_dataframe(raw_df)
    if id_col in clean_df.columns:
        clean_df = clean_df[clean_df[id_col].notna()]
    if target_col in clean_df.columns:
        clean_df[target_col] = pd.to_numeric(clean_df[target_col], errors='coerce')
        clean_df = clean_df[clean_df[target_col].notna()]
    clean_df = clean_df.reset_index(drop=True)
except Exception as exc:
    st.error(f'Cleaning failed: {exc}')
    st.stop()

if clean_df.empty:
    st.error('Cleaned data is empty. Check the header row, ID column, and target column.')
    st.stop()

with st.sidebar:
    st.header('3. Query Row')
    query_mode = st.radio('Query mode', ['By ID', 'By row index'])

    if query_mode == 'By ID':
        id_values = clean_df[id_col].dropna().astype(str).unique().tolist()
        selected_id = st.selectbox('Query ID', id_values)
        query_index = int(clean_df.index[clean_df[id_col].astype(str) == str(selected_id)][0])
    else:
        query_index = int(st.number_input('Query row index', 0, max(len(clean_df) - 1, 0), 0))

    st.header('4. Run Settings')
    synthetic_multiplier = st.slider('Synthetic multiplier', 0, 10, 3)
    n_neighbors = st.slider('Similarity neighbors', 1, 20, min(5, len(clean_df)))
    test_size = st.slider('Pricing validation size', 0.10, 0.40, 0.15, 0.05)
    random_state = st.number_input('Random state', value=42, step=1)
    run = st.button('Run Full Pipeline', type='primary', use_container_width=True)

query_row = clean_df.iloc[[query_index]].reset_index(drop=True)

preview_tab, query_tab = st.tabs(['Preview', 'Selected Query'])
with preview_tab:
    st.subheader('Raw Data')
    st.dataframe(preview(raw_df), use_container_width=True)
    st.subheader('Cleaned Data')
    st.dataframe(preview(clean_df), use_container_width=True)

with query_tab:
    st.subheader('Selected Query Row')
    st.dataframe(query_row, use_container_width=True)

if not run:
    st.info('Set options in the sidebar, then click Run Full Pipeline.')
    st.stop()

try:
    similarity_raw, similarity_clean, prepared_real = similarity_from_processed_real(
        clean_df=clean_df,
        query_index=query_index,
        id_col=id_col,
        k=int(n_neighbors),
        target_col=target_col,
    )

    pricing_train_df = combine_real_and_synthetic(
        real_df=clean_df,
        synthetic_df=None,
        multiplier=int(synthetic_multiplier),
        random_state=int(random_state),
    )

    synthetic_only_df = pd.DataFrame()
    if ROW_SOURCE_COL in pricing_train_df.columns:
        synthetic_only_df = pricing_train_df[
            pricing_train_df[ROW_SOURCE_COL].astype(str).str.contains('synthetic', case=False, na=False)
        ].copy()

    pricing_result = train_and_price(
        pricing_train_df=pricing_train_df,
        query_row=query_row,
        target_col=target_col,
        test_size=float(test_size),
        random_state=int(random_state),
    )
except Exception as exc:
    st.error(f'Pipeline failed: {exc}')
    st.stop()

summary_tab, data_tab, pricing_tab, similarity_tab, export_tab = st.tabs(
    ['Summary', 'Data', 'Pricing', 'Similarity', 'Export']
)

with summary_tab:
    st.subheader('Run Summary')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Raw rows', len(raw_df))
    c2.metric('Clean rows', len(clean_df))
    c3.metric('Synthetic rows', len(synthetic_only_df))
    c4.metric('Pricing train rows', len(pricing_train_df))

    st.markdown('**Selected query row**')
    st.dataframe(query_row, use_container_width=True)

    st.markdown('**Flow**')
    st.code(
        'upload -> clean_raw_dataframe -> prepare_dataset\n'
        'similarity: processed real data -> KNN comparable deals\n'
        'pricing: real + synthetic bootstrap data -> train linear/ridge/xgboost -> price selected row',
        language='text',
    )

with data_tab:
    st.subheader('Processed Real Features')
    st.dataframe(preview(prepared_real.features), use_container_width=True)
    st.subheader('Synthetic Rows')
    st.dataframe(preview(synthetic_only_df), use_container_width=True)
    st.subheader('Pricing Training Data: Real + Synthetic')
    st.dataframe(preview(pricing_train_df), use_container_width=True)

with pricing_tab:
    st.subheader('Pricing Result')
    p1, p2, p3, p4 = st.columns(4)
    p1.metric('Best Model', pricing_result['best_model'])
    p2.metric('Point Estimate', round(pricing_result['point_estimate'], 4))
    p3.metric('Lower Bound', round(pricing_result['lower_bound'], 4))
    p4.metric('Upper Bound', round(pricing_result['upper_bound'], 4))

    st.caption(f"Residual std: {pricing_result['residual_std']:.4f}")
    st.markdown('**Validation Metrics**')
    st.dataframe(pd.DataFrame(pricing_result['metrics']).T, use_container_width=True)
    st.markdown('**Validation Predictions**')
    st.dataframe(preview(pricing_result['validation_predictions']), use_container_width=True)
    st.markdown('**Feature Importance**')
    st.dataframe(preview(pricing_result['feature_importance'], 30), use_container_width=True)

with similarity_tab:
    st.subheader('Similarity Results')
    st.markdown('**Clean Ranked Results with Contribution**')
    st.dataframe(display_safe_dataframe(similarity_clean), use_container_width=True)
    st.markdown('**Raw Ranked Results**')
    st.dataframe(display_safe_dataframe(similarity_raw), use_container_width=True)

    if 'top_features' in similarity_clean.columns:
        st.markdown('**Contribution Detail**')
        for _, row in similarity_clean.iterrows():
            label = str(row.get(id_col, 'Comparable'))
            with st.expander(label):
                st.write(json.dumps(row.get('top_features'), indent=2, default=str))

with export_tab:
    st.subheader('Export')
    summary = {
        'id_col': id_col,
        'target_col': target_col,
        'query_row': query_row.to_dict(orient='records'),
        'pricing_result': {
            'best_model': pricing_result['best_model'],
            'point_estimate': pricing_result['point_estimate'],
            'lower_bound': pricing_result['lower_bound'],
            'upper_bound': pricing_result['upper_bound'],
            'residual_std': pricing_result['residual_std'],
            'metrics': pricing_result['metrics'],
        },
        'similarity_clean': similarity_clean.to_dict(orient='records'),
    }

    st.download_button(
        'Download JSON Summary',
        data=json.dumps(summary, indent=2, default=str),
        file_name='clo_demo_summary.json',
        mime='application/json',
    )
    st.download_button(
        'Download Cleaned Data CSV',
        data=clean_df.to_csv(index=False).encode('utf-8'),
        file_name='cleaned_data.csv',
        mime='text/csv',
    )
    st.download_button(
        'Download Similarity CSV',
        data=similarity_clean.to_csv(index=False).encode('utf-8'),
        file_name='similarity_results.csv',
        mime='text/csv',
    )

st.success('Pipeline finished successfully.')
