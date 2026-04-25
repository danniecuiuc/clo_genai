"""
CLO Pricing + Similarity Streamlit Demo

Flow:
Upload file
-> cleaning.clean_raw_dataframe
-> pipeline.prepare_dataset on real cleaned data
-> Branch A: Similarity uses processed real data directly
-> Branch B: Pricing creates synthetic data, then runs pipeline + model training

Run:
    python -m streamlit run app/ui.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Make imports work when running from project root:
#     python -m streamlit run app/ui.py
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from preprocessing import cleaning as cleaning_mod
from preprocessing import pipeline as pipe
from synthetic import bootstrap_generator as bootstrap_mod
from pricing import models as pricing_models
from explainability import explainability as explain_mod
from uncertainty import uncertainty as uncertainty_mod
from similarity import siilarity_V2 as similarity_mod

try:
    import common.config as config
except Exception:
    config = None


ROW_SOURCE_COL = getattr(config, "ROW_SOURCE_COL", "row_source") if config else "row_source"

ID_CANDIDATES = [
    "Bloomberg ID",
    "BBG ID",
    "ID",
    "Deal ID",
    "Tranche ID",
    "Security ID",
]

TARGET_CANDIDATES = [
    "Cover Price",
    "Price",
    "Spread",
    "OAS",
    "DM",
    "Target",
]


# ============================================================
# UI helpers
# ============================================================

def read_file(uploaded_file, header_row: int) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file, header=header_row)
    return pd.read_excel(uploaded_file, header=header_row)


def guess_col(cols: list[str], candidates: list[str], fallback: int = 0) -> str:
    for candidate in candidates:
        if candidate in cols:
            return candidate
    return cols[min(fallback, len(cols) - 1)]


def preview(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    return df.head(n).copy()


def display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Make object columns safe for Streamlit / pyarrow display."""
    out = df.copy()

    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].apply(
                lambda x: json.dumps(x, default=str)
                if isinstance(x, (list, tuple, dict))
                else x
            )

    return out


def to_dense(x):
    """Convert scipy sparse matrix to dense numpy array if needed."""
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


# ============================================================
# Pricing branch
# Synthetic data is generated first, then pipeline preprocessing is fitted.
# ============================================================

def train_and_price(
    pricing_train_df: pd.DataFrame,
    query_row: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
) -> dict:
    prepared = pipe.prepare_dataset(pricing_train_df, target_col=target_col)

    X_raw = prepared.features.copy()
    y = pd.to_numeric(prepared.target, errors="coerce")

    mask = y.notna()
    X_raw = X_raw.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    if len(X_raw) < 5:
        raise ValueError("Not enough rows after cleaning/preprocessing.")

    X_train_raw, X_valid_raw, y_train, y_valid = train_test_split(
        X_raw,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = pipe.build_preprocessor(X_train_raw)

    X_train = preprocessor.fit_transform(X_train_raw)
    X_valid = preprocessor.transform(X_valid_raw)

    feature_names = pipe.get_transformed_feature_names(preprocessor)

    outputs = pricing_models.train_candidate_models(
        X_train,
        y_train,
        X_valid,
        y_valid,
        random_state=random_state,
    )

    best_model = outputs.fitted_models[outputs.best_model_name]

    query_prepared = pipe.prepare_dataset(query_row, target_col=target_col)
    query_X_raw = query_prepared.features.reindex(columns=X_raw.columns, fill_value=np.nan)
    query_X = preprocessor.transform(query_X_raw)

    point = float(best_model.predict(query_X)[0])
    lower, upper = uncertainty_mod.prediction_interval(point, outputs.residual_std)
    importance = explain_mod.build_feature_importance_table(best_model, feature_names)

    return {
        "best_model": outputs.best_model_name,
        "point_estimate": point,
        "lower_bound": lower,
        "upper_bound": upper,
        "residual_std": outputs.residual_std,
        "metrics": outputs.metrics,
        "validation_predictions": outputs.validation_predictions,
        "feature_importance": importance,
    }


# ============================================================
# Similarity branch
# Uses processed real data directly.
# ============================================================

def similarity_from_processed_real(
    clean_df: pd.DataFrame,
    prepared_real,
    query_index: int,
    id_col: str,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_real_raw = prepared_real.features.copy()

    sim_preprocessor = pipe.build_preprocessor(X_real_raw)
    X_real = to_dense(sim_preprocessor.fit_transform(X_real_raw))

    feature_names = pipe.get_transformed_feature_names(sim_preprocessor)

    target_scaled = X_real[[query_index]]
    dist, ind = similarity_mod.knn(X_real, target_scaled, k)

    ranked_raw = clean_df.iloc[ind[0]].copy()
    ranked_raw["distance"] = dist[0]

    contrib = similarity_mod.contribution(
        X_real,
        target_scaled,
        ind[0],
        feature_names,
    )

    keep_cols = [
        c for c in [id_col, "Collateral manager", "distance"]
        if c in ranked_raw.columns
    ]

    ranked_clean = ranked_raw[keep_cols].copy()
    ranked_clean["top_features"] = ranked_clean.index.map(contrib)

    return ranked_raw.reset_index(drop=True), ranked_clean.reset_index(drop=True)


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="CLO Demo", layout="wide")

st.title("CLO Pricing + Similarity Demo")
st.caption(
    "Upload data → cleaning + pipeline preprocess → "
    "similarity on real data + pricing on synthetic-augmented data"
)


with st.sidebar:
    st.header("1. Upload")

    uploaded = st.file_uploader(
        "Excel / CSV",
        type=["xlsx", "xls", "csv"],
    )

    header_row = st.number_input(
        "Header row index",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
    )


if uploaded is None:
    st.info("Upload a file from the left sidebar to start.")
    st.stop()


try:
    raw_df = read_file(uploaded, int(header_row))
    raw_df.columns = [str(c).strip() for c in raw_df.columns]
except Exception as exc:
    st.error(f"Failed to read file: {exc}")
    st.stop()


cols = raw_df.columns.tolist()

with st.sidebar:
    st.header("2. Columns")

    id_default = guess_col(cols, ID_CANDIDATES, fallback=0)
    target_default = guess_col(
        cols,
        TARGET_CANDIDATES,
        fallback=1 if len(cols) > 1 else 0,
    )

    id_col = st.selectbox(
        "ID column",
        cols,
        index=cols.index(id_default),
    )

    target_col = st.selectbox(
        "Target column",
        cols,
        index=cols.index(target_default),
    )


# --------------------------
# Main cleaning trunk
# --------------------------
try:
    clean_df = cleaning_mod.clean_raw_dataframe(raw_df)

    if id_col in clean_df.columns:
        clean_df = clean_df[clean_df[id_col].notna()]

    if target_col in clean_df.columns:
        clean_df[target_col] = pd.to_numeric(clean_df[target_col], errors="coerce")
        clean_df = clean_df[clean_df[target_col].notna()]

    clean_df = clean_df.reset_index(drop=True)

except Exception as exc:
    st.error(f"Cleaning failed: {exc}")
    st.stop()


if clean_df.empty:
    st.error("Cleaned data is empty. Check header row, ID column, or target column.")
    st.stop()


# --------------------------
# Main pipeline trunk on real data
# --------------------------
try:
    prepared_real = pipe.prepare_dataset(clean_df, target_col=target_col)
except Exception as exc:
    st.error(f"Pipeline preprocessing failed: {exc}")
    st.stop()


with st.sidebar:
    st.header("3. Query Row")

    query_mode = st.radio("Query mode", ["By ID", "By row index"])

    if query_mode == "By ID":
        id_values = clean_df[id_col].dropna().astype(str).unique().tolist()

        selected_id = st.selectbox("Query ID", id_values)

        query_index = int(
            clean_df.index[
                clean_df[id_col].astype(str) == str(selected_id)
            ][0]
        )
    else:
        query_index = int(
            st.number_input(
                "Query row index",
                min_value=0,
                max_value=max(len(clean_df) - 1, 0),
                value=0,
            )
        )

    st.header("4. Run Settings")

    synthetic_multiplier = st.slider(
        "Synthetic multiplier",
        min_value=0,
        max_value=10,
        value=3,
    )

    n_neighbors = st.slider(
        "Similarity neighbors",
        min_value=1,
        max_value=20,
        value=5,
    )

    test_size = st.slider(
        "Pricing test size",
        min_value=0.10,
        max_value=0.40,
        value=0.15,
        step=0.05,
    )

    random_state = st.number_input(
        "Random state",
        value=42,
        step=1,
    )

    run = st.button(
        "Run Full Pipeline",
        type="primary",
        use_container_width=True,
    )


query_row = clean_df.iloc[[query_index]].reset_index(drop=True)


preview_tab, query_tab = st.tabs(["Preview", "Selected Query"])

with preview_tab:
    st.subheader("Raw Data")
    st.dataframe(preview(raw_df), use_container_width=True)

    st.subheader("Cleaned Data")
    st.dataframe(preview(clean_df), use_container_width=True)

    st.subheader("Processed Real Features from pipeline.prepare_dataset")
    st.dataframe(preview(prepared_real.features), use_container_width=True)


with query_tab:
    st.subheader("Selected Query Row")
    st.dataframe(query_row, use_container_width=True)


if not run:
    st.info("Set options in the sidebar, then click Run Full Pipeline.")
    st.stop()


try:
    # Branch A: similarity directly from processed real data
    similarity_raw, similarity_clean = similarity_from_processed_real(
        clean_df=clean_df,
        prepared_real=prepared_real,
        query_index=query_index,
        id_col=id_col,
        k=int(n_neighbors),
    )

    # Branch B: pricing uses synthetic-augmented data, then pipeline preprocessing
    pricing_train_df = bootstrap_mod.combine_real_and_synthetic(
        real_df=clean_df,
        synthetic_df=None,
        multiplier=int(synthetic_multiplier),
        random_state=int(random_state),
    )

    synthetic_only_df = pd.DataFrame()

    if ROW_SOURCE_COL in pricing_train_df.columns:
        synthetic_only_df = pricing_train_df[
            pricing_train_df[ROW_SOURCE_COL]
            .astype(str)
            .str.contains("synthetic", case=False, na=False)
        ].copy()

    pricing_result = train_and_price(
        pricing_train_df=pricing_train_df,
        query_row=query_row,
        target_col=target_col,
        test_size=float(test_size),
        random_state=int(random_state),
    )

except Exception as exc:
    st.error(f"Pipeline failed: {exc}")
    st.stop()


summary_tab, data_tab, pricing_tab, similarity_tab, export_tab = st.tabs(
    ["Summary", "Data", "Pricing", "Similarity", "Export"]
)


with summary_tab:
    st.subheader("Run Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Raw rows", len(raw_df))
    c2.metric("Clean rows", len(clean_df))
    c3.metric("Synthetic rows", len(synthetic_only_df))
    c4.metric("Train rows", len(pricing_train_df))

    st.markdown("**Selected query row**")
    st.dataframe(query_row, use_container_width=True)

    st.markdown("**Pipeline**")
    st.code(
        "\n".join(
            [
                "upload",
                "-> cleaning.clean_raw_dataframe",
                "-> pipeline.prepare_dataset on real data",
                "-> branch A similarity: processed real features -> siilarity_V2.knn/contribution",
                "-> branch B pricing: synthetic generation -> pipeline.prepare_dataset/build_preprocessor -> models.train_candidate_models",
            ]
        ),
        language="text",
    )


with data_tab:
    st.subheader("Processed Real Features")
    st.dataframe(preview(prepared_real.features), use_container_width=True)

    st.subheader("Synthetic Rows")
    st.dataframe(preview(synthetic_only_df), use_container_width=True)

    st.subheader("Pricing Training Data: Real + Synthetic")
    st.dataframe(preview(pricing_train_df), use_container_width=True)


with pricing_tab:
    st.subheader("Pricing Result")

    p1, p2, p3, p4 = st.columns(4)

    p1.metric("Best Model", pricing_result["best_model"])
    p2.metric("Point Estimate", round(pricing_result["point_estimate"], 4))
    p3.metric("Lower Bound", round(pricing_result["lower_bound"], 4))
    p4.metric("Upper Bound", round(pricing_result["upper_bound"], 4))

    st.caption(f"Residual std: {pricing_result['residual_std']:.4f}")

    st.markdown("**Validation Metrics**")
    st.dataframe(
        pd.DataFrame(pricing_result["metrics"]).T,
        use_container_width=True,
    )

    st.markdown("**Validation Predictions**")
    st.dataframe(
        preview(pricing_result["validation_predictions"]),
        use_container_width=True,
    )

    st.markdown("**Feature Importance**")
    st.dataframe(
        preview(pricing_result["feature_importance"], 30),
        use_container_width=True,
    )


with similarity_tab:
    st.subheader("Similarity Results")

    st.markdown("**Clean Ranked Results with Contribution**")
    st.dataframe(
        display_df(similarity_clean),
        use_container_width=True,
    )

    st.markdown("**Raw Ranked Results**")
    st.dataframe(
        display_df(similarity_raw),
        use_container_width=True,
    )

    if "top_features" in similarity_clean.columns:
        st.markdown("**Contribution Detail**")

        for _, row in similarity_clean.iterrows():
            label = str(row.get(id_col, "Comparable"))

            with st.expander(label):
                st.write(
                    json.dumps(
                        row.get("top_features"),
                        indent=2,
                        default=str,
                    )
                )


with export_tab:
    st.subheader("Export")

    summary = {
        "id_col": id_col,
        "target_col": target_col,
        "query_row": query_row.to_dict(orient="records"),
        "pricing_result": {
            "best_model": pricing_result["best_model"],
            "point_estimate": pricing_result["point_estimate"],
            "lower_bound": pricing_result["lower_bound"],
            "upper_bound": pricing_result["upper_bound"],
            "residual_std": pricing_result["residual_std"],
            "metrics": pricing_result["metrics"],
        },
        "similarity_clean": similarity_clean.to_dict(orient="records"),
    }

    st.download_button(
        "Download JSON Summary",
        data=json.dumps(summary, indent=2, default=str),
        file_name="clo_demo_summary.json",
        mime="application/json",
    )

    st.download_button(
        "Download Cleaned Data CSV",
        data=clean_df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_data.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download Similarity CSV",
        data=similarity_clean.to_csv(index=False).encode("utf-8"),
        file_name="similarity_results.csv",
        mime="text/csv",
    )


st.success("Pipeline finished successfully.")