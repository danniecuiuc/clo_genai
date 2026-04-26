"""
Professional Streamlit UI backed by pricing/clo_pricing.py.

Two workflows:

1. User Pricing
   - No dataset upload required
   - User manually enters one new security / tranche
   - App loads an existing model bundle and prices the input

2. Research / Training
   - Upload historical CLO data
   - Train / refresh model bundle
   - Select an existing security by security name
   - Validate pricing, feature drivers, and comparable deals

Run from project root:
    python -m streamlit run app/ui.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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


SECURITY_NAME_CANDIDATES = [
    "Security Name",
    "security_name",
    "Security",
    "Security Description",
    "Name",
    "Tranche Name",
    "Tranche",
    "Bloomberg ID",
    "BBG ID",
    "Security ID",
]


FEATURE_DEFAULTS = {
    "Manager Discount": -3.0,
    "Spread": 500.0,
    "NC Yrs left": 1.0,
    "RP Yrs left": 3.0,
    "MVOC": 1.05,
    "MVOC (w/o X)": 1.05,
    "Attach": 0.08,
    "Thickness": 0.04,
    "Diversity": 80.0,
    "WAS %ile": 0.50,
    "% <80": 0.03,
    "% <50": 0.01,
    "% CCC": 0.05,
    "Equity NAV": 0.55,
    "Eq Last Payment": 0.07,
    "Junior OC": 0.04,
    "Excess Spread": 0.015,
    "Deal WACC": 0.018,
    "AAA Coupon": 0.015,
    "Palmer Square DM Index": 650.0,
    "Palmer Square Price Index": 96.0,
    "LSTA Index": 96.0,
    "LSTA 100 Index": 98.0,
    "HY CDX": 300.0,
    "VIX": 20.0,
}


FEATURE_HELP = {
    "Manager Discount": "Manager-level pricing discount / premium signal.",
    "Spread": "Tranche spread, usually in basis points.",
    "NC Yrs left": "Years remaining in non-call period.",
    "RP Yrs left": "Years remaining in reinvestment period.",
    "MVOC": "Market value over-collateralization.",
    "MVOC (w/o X)": "MVOC excluding selected adjustment / excluded assets.",
    "Attach": "Attachment point.",
    "Thickness": "Tranche thickness.",
    "Diversity": "Collateral diversity score.",
    "WAS %ile": "Weighted average spread percentile.",
    "% <80": "Share of assets priced below 80.",
    "% <50": "Share of assets priced below 50.",
    "% CCC": "CCC bucket exposure.",
    "Equity NAV": "Equity NAV.",
    "Eq Last Payment": "Most recent equity payment.",
    "Junior OC": "Junior over-collateralization cushion.",
    "Excess Spread": "Deal excess spread.",
    "Deal WACC": "Deal weighted average cost of capital.",
    "AAA Coupon": "AAA coupon level.",
    "Palmer Square DM Index": "Market index input.",
    "Palmer Square Price Index": "Market price index input.",
    "LSTA Index": "LSTA leveraged loan index.",
    "LSTA 100 Index": "LSTA 100 index.",
    "HY CDX": "High-yield CDX level.",
    "VIX": "Equity volatility index.",
}


def read_uploaded_file(uploaded_file, header_row: int) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        raw = pd.read_csv(uploaded_file, header=header_row)
    else:
        raw = pd.read_excel(uploaded_file, header=header_row)

    raw.columns = [str(c).strip() for c in raw.columns]
    return raw


def normalize_for_clo_pricing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    if TARGET not in out.columns:
        for candidate in TARGET_FALLBACKS:
            if candidate in out.columns:
                out[TARGET] = out[candidate]
                break

    if TARGET not in out.columns:
        raise ValueError(
            f"Missing target column '{TARGET}'. Tried fallbacks: {TARGET_FALLBACKS}"
        )

    missing_features = [col for col in ALL_FEATURES if col not in out.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    for col in ALL_FEATURES + [TARGET]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[out[TARGET].notna()].reset_index(drop=True)
    return out


def guess_security_name_column(df: pd.DataFrame) -> str:
    for col in SECURITY_NAME_CANDIDATES:
        if col in df.columns:
            return col

    object_cols = [
        col for col in df.columns
        if df[col].dtype == "object" and df[col].notna().any()
    ]
    if object_cols:
        return object_cols[0]

    raise ValueError(
        "Could not find a security name column. Please include a column such as "
        "'Security Name', 'Tranche Name', 'Bloomberg ID', or 'Security ID'."
    )


def format_security_option(row: pd.Series, security_col: str) -> str:
    security_name = str(row.get(security_col, "")).strip()
    manager = str(row.get("Collateral manager", "")).strip()
    target_value = row.get(TARGET, None)

    pieces = [security_name]
    if manager and manager.lower() != "nan":
        pieces.append(manager)
    if pd.notna(target_value):
        try:
            pieces.append(f"{TARGET}: {float(target_value):.2f}")
        except Exception:
            pieces.append(f"{TARGET}: {target_value}")

    return "  |  ".join(pieces)


def display_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].apply(
                lambda x: json.dumps(x, default=str)
                if isinstance(x, (list, tuple, dict))
                else x
            )
    return out


def preview(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    return df.head(n).copy()


def feature_default(feature: str) -> float:
    return float(FEATURE_DEFAULTS.get(feature, 0.0))


def feature_step(feature: str) -> float:
    if feature in {"Spread", "Palmer Square DM Index", "HY CDX"}:
        return 5.0
    if feature in {"Diversity"}:
        return 1.0
    if feature in {"VIX"}:
        return 0.5
    if "%" in feature or feature in {
        "MVOC",
        "MVOC (w/o X)",
        "Attach",
        "Thickness",
        "WAS %ile",
        "Equity NAV",
        "Eq Last Payment",
        "Junior OC",
        "Excess Spread",
        "Deal WACC",
        "AAA Coupon",
    }:
        return 0.001
    return 0.1


def build_manual_query_row() -> tuple[pd.DataFrame, str, str]:
    st.markdown("### Security Information")

    info_col_1, info_col_2 = st.columns(2)

    with info_col_1:
        security_name = st.text_input(
            "Security Name",
            value="New CLO Tranche",
            help="User-facing name for the tranche being priced.",
        )

    with info_col_2:
        collateral_manager = st.text_input(
            "Collateral Manager",
            value="",
            help="Optional. Used for display and comparable-deal context if available.",
        )

    st.markdown("### Deal & Market Inputs")

    input_values: dict[str, Any] = {}

    feature_groups = {
        "Tranche Structure": [
            "Spread",
            "NC Yrs left",
            "RP Yrs left",
            "Attach",
            "Thickness",
        ],
        "Collateral Quality": [
            "MVOC",
            "MVOC (w/o X)",
            "Diversity",
            "WAS %ile",
            "% <80",
            "% <50",
            "% CCC",
        ],
        "Deal Economics": [
            "Manager Discount",
            "Equity NAV",
            "Eq Last Payment",
            "Junior OC",
            "Excess Spread",
            "Deal WACC",
            "AAA Coupon",
        ],
        "Market Conditions": [
            "Palmer Square DM Index",
            "Palmer Square Price Index",
            "LSTA Index",
            "LSTA 100 Index",
            "HY CDX",
            "VIX",
        ],
    }

    for group_name, features in feature_groups.items():
        available_features = [f for f in features if f in ALL_FEATURES]
        if not available_features:
            continue

        with st.expander(group_name, expanded=True):
            cols = st.columns(3)
            for i, feature in enumerate(available_features):
                with cols[i % 3]:
                    input_values[feature] = st.number_input(
                        feature,
                        value=feature_default(feature),
                        step=feature_step(feature),
                        format="%.6f",
                        help=FEATURE_HELP.get(feature),
                    )

    missing_features = [f for f in ALL_FEATURES if f not in input_values]
    for feature in missing_features:
        input_values[feature] = feature_default(feature)

    input_values["Security Name"] = security_name
    input_values["Collateral manager"] = collateral_manager

    query_row = pd.DataFrame([input_values])

    for col in ALL_FEATURES:
        query_row[col] = pd.to_numeric(query_row[col], errors="coerce")

    return query_row, security_name, collateral_manager


def render_prediction_result(
    prediction: dict,
    selected_security: str,
    bundle_path: str,
    query_row: pd.DataFrame,
) -> None:
    st.markdown("## Pricing Result")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Price", prediction["point_estimate"])
    c2.metric("Lower Bound", prediction["lower_bound"])
    c3.metric("Upper Bound", prediction["upper_bound"])
    c4.metric("Ridge Baseline", prediction.get("baseline_estimate", "n/a"))

    result_tabs = st.tabs(
        [
            "Model Drivers",
            "Comparable Deals",
            "Input Review",
            "Export",
        ]
    )

    with result_tabs[0]:
        st.markdown("### Feature Importance")
        feature_importance = prediction.get("feature_importance", pd.DataFrame())

        if isinstance(feature_importance, pd.DataFrame) and not feature_importance.empty:
            st.dataframe(
                display_safe(feature_importance),
                use_container_width=True,
                hide_index=True,
            )

            chart_df = feature_importance.copy()
            if {"feature", "importance"}.issubset(chart_df.columns):
                st.bar_chart(chart_df.set_index("feature")["importance"])
        else:
            st.info("No feature importance table returned by the pricing bundle.")

    with result_tabs[1]:
        st.markdown("### Comparable Deals")
        comparables = pd.DataFrame(prediction.get("comparables", []))

        if not comparables.empty:
            st.dataframe(
                display_safe(comparables),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No comparable deals returned.")

    with result_tabs[2]:
        st.markdown("### Submitted Input")
        display_cols = [
            col for col in [
                "Security Name",
                "Collateral manager",
                "Spread",
                "MVOC",
                "Attach",
                "Thickness",
                "NC Yrs left",
                "RP Yrs left",
                "VIX",
            ]
            if col in query_row.columns
        ]

        st.dataframe(
            display_safe(query_row[display_cols] if display_cols else query_row),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("All model features", expanded=False):
            st.dataframe(
                display_safe(query_row[ALL_FEATURES]),
                use_container_width=True,
                hide_index=True,
            )

    with result_tabs[3]:
        st.markdown("### Export")

        export_payload = {
            "selected_security": selected_security,
            "bundle_path": bundle_path,
            "input": query_row.to_dict(orient="records"),
            "prediction": prediction,
        }

        st.download_button(
            "Download Prediction JSON",
            data=json.dumps(export_payload, indent=2, default=str),
            file_name="clo_prediction.json",
            mime="application/json",
            use_container_width=True,
        )

        comparables = pd.DataFrame(prediction.get("comparables", []))
        if not comparables.empty:
            st.download_button(
                "Download Comparable Deals CSV",
                data=comparables.to_csv(index=False).encode("utf-8"),
                file_name="comparable_deals.csv",
                mime="text/csv",
                use_container_width=True,
            )


def render_header() -> None:
    st.markdown(
        """
        <style>
            .main .block-container {
                padding-top: 1.8rem;
                padding-bottom: 2rem;
                max-width: 1280px;
            }
            .hero-card {
                padding: 1.35rem 1.55rem;
                border-radius: 18px;
                border: 1px solid rgba(120, 120, 120, 0.18);
                background: linear-gradient(135deg, rgba(245,247,250,0.95), rgba(255,255,255,0.9));
                box-shadow: 0 8px 24px rgba(0,0,0,0.04);
                margin-bottom: 1.1rem;
            }
            .hero-title {
                font-size: 2.05rem;
                font-weight: 760;
                margin-bottom: 0.25rem;
                letter-spacing: -0.03em;
            }
            .hero-subtitle {
                font-size: 1rem;
                color: #555;
                margin-bottom: 0;
            }
            div[data-testid="stMetric"] {
                background: rgba(255,255,255,0.82);
                border: 1px solid rgba(120, 120, 120, 0.16);
                padding: 0.8rem 0.9rem;
                border-radius: 14px;
                box-shadow: 0 3px 12px rgba(0,0,0,0.025);
            }
        </style>
        <div class="hero-card">
            <div class="hero-title">CLO Tranche Pricing Workbench</div>
            <p class="hero-subtitle">
                Price a new CLO tranche using a saved model bundle, or use the research workflow
                to train, validate, and inspect model behavior on historical data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="CLO Pricing Workbench", layout="wide")
render_header()

with st.sidebar:
    st.markdown("### Shared Settings")
    bundle_path = st.text_input("Model bundle path", value=str(MODEL_BUNDLE_PATH))
    st.caption("Train a bundle in the Research tab, then use it in the User Pricing tab.")

user_tab, research_tab = st.tabs(["User Pricing", "Research / Training"])


# =============================================================================
# USER PRICING TAB
# =============================================================================

with user_tab:
    st.markdown("## Price a New Security")
    st.caption(
        "Use this workflow for an end user: enter the tranche characteristics directly, "
        "then load the saved bundle to produce a price, confidence range, feature drivers, "
        "and comparable deals."
    )

    query_row, security_name, collateral_manager = build_manual_query_row()

    st.markdown("### Input Snapshot")
    snapshot_cols = [
        col for col in [
            "Security Name",
            "Collateral manager",
            "Spread",
            "MVOC",
            "Attach",
            "Thickness",
            "NC Yrs left",
            "RP Yrs left",
            "VIX",
        ]
        if col in query_row.columns
    ]

    st.dataframe(
        display_safe(query_row[snapshot_cols]),
        use_container_width=True,
        hide_index=True,
    )

    price_new_security = st.button(
        "Price New Security",
        type="primary",
        use_container_width=True,
    )

    if price_new_security:
        bundle_file = Path(bundle_path)

        if not bundle_file.exists():
            st.error(
                f"Bundle not found at {bundle_path}. "
                "Please train/save a bundle in the Research / Training tab first."
            )
        else:
            with st.spinner("Loading model bundle and pricing new security..."):
                try:
                    prediction = predict_from_bundle(
                        query_row=query_row,
                        bundle_path=bundle_path,
                    )

                    render_prediction_result(
                        prediction=prediction,
                        selected_security=security_name,
                        bundle_path=bundle_path,
                        query_row=query_row,
                    )

                except Exception as exc:
                    st.error(f"Pricing failed: {exc}")


# =============================================================================
# RESEARCH / TRAINING TAB
# =============================================================================

with research_tab:
    st.markdown("## Research / Training Workflow")
    st.caption(
        "Use this workflow to upload historical data, train or refresh the model bundle, "
        "inspect synthetic data, and validate pricing on an existing security from the dataset."
    )

    with st.sidebar:
        st.divider()
        st.markdown("### Research Data")
        uploaded = st.file_uploader("Historical Excel / CSV file", type=["xlsx", "xls", "csv"])
        header_row = st.number_input(
            "Header row index",
            min_value=0,
            max_value=10,
            value=1,
            step=1,
            help="Default is 1 if the second row contains column names.",
        )

        st.markdown("### Training Options")
        synth_rows = st.slider(
            "Synthetic rows",
            min_value=0,
            max_value=5000,
            value=500,
            step=100,
            help="Used during training. The stable default generator is bootstrap.",
        )

        similarity_real_only = st.checkbox(
            "Similarity on real rows only",
            value=True,
            help="Recommended: comparable deals should come from real historical rows.",
        )

        train_save = st.button(
            "Train + Save Bundle",
            type="primary",
            use_container_width=True,
        )

    if uploaded is None:
        st.info("Upload a historical CLO tranche dataset to use the research workflow.")
        st.stop()

    try:
        raw_df = read_uploaded_file(uploaded, int(header_row))
        clean_df = normalize_for_clo_pricing(raw_df)
        security_name_col = guess_security_name_column(clean_df)
    except Exception as exc:
        st.error(f"Failed to load or normalize data: {exc}")
        st.stop()

    if clean_df.empty:
        st.error("No usable rows after normalization.")
        st.stop()

    top_left, top_mid, top_right = st.columns(3)
    top_left.metric("Raw Rows", f"{len(raw_df):,}")
    top_mid.metric("Usable Rows", f"{len(clean_df):,}")
    top_right.metric("Model Features", f"{len(ALL_FEATURES):,}")

    if train_save:
        with st.spinner("Training pricing bundle and comparable-deal index..."):
            try:
                result = train_and_save_bundle(
                    df=clean_df,
                    bundle_path=bundle_path,
                    synth_rows=int(synth_rows),
                    similarity_on_real_only=similarity_real_only,
                )
                st.success(f"Bundle saved to {result['bundle_path']}")

                metrics_df = pd.DataFrame(result.get("metrics", {})).T
                if not metrics_df.empty:
                    st.markdown("### Training Metrics")
                    st.dataframe(metrics_df, use_container_width=True)

                with st.expander("Training bundle details", expanded=False):
                    st.json(result)

            except Exception as exc:
                st.error(f"Training failed: {exc}")

    st.markdown("### Validate on Existing Security")

    security_options = {
        format_security_option(row, security_name_col): idx
        for idx, row in clean_df.iterrows()
    }

    selector_col, action_col = st.columns([3, 1])

    with selector_col:
        selected_label = st.selectbox(
            "Security name",
            options=list(security_options.keys()),
            help=f"Dropdown is based on column: {security_name_col}",
        )
        query_index = security_options[selected_label]

    with action_col:
        st.write("")
        st.write("")
        load_price = st.button(
            "Price Selected Security",
            use_container_width=True,
        )

    research_query_row = clean_df.iloc[[query_index]].copy()

    st.markdown("### Selected Security Snapshot")
    research_snapshot_cols = [
        col for col in [
            security_name_col,
            "Collateral manager",
            TARGET,
            "Spread",
            "MVOC",
            "Attach",
            "Thickness",
            "NC Yrs left",
            "RP Yrs left",
            "VIX",
        ]
        if col in research_query_row.columns
    ]

    st.dataframe(
        display_safe(
            research_query_row[research_snapshot_cols]
            if research_snapshot_cols
            else research_query_row
        ),
        use_container_width=True,
        hide_index=True,
    )

    if load_price:
        bundle_file = Path(bundle_path)

        if not bundle_file.exists():
            st.error(f"Bundle not found at {bundle_path}. Train and save the bundle first.")
        else:
            with st.spinner("Loading bundle and pricing selected security..."):
                try:
                    prediction = predict_from_bundle(
                        query_row=research_query_row,
                        bundle_path=bundle_path,
                    )

                    render_prediction_result(
                        prediction=prediction,
                        selected_security=selected_label,
                        bundle_path=bundle_path,
                        query_row=research_query_row,
                    )

                except Exception as exc:
                    st.error(f"Pricing failed: {exc}")

    research_tabs = st.tabs(
        [
            "Data Preview",
            "Synthetic Preview",
            "Required Features",
        ]
    )

    with research_tabs[0]:
        data_a, data_b = st.columns(2)
        with data_a:
            st.markdown("#### Raw Data Preview")
            st.dataframe(preview(raw_df), use_container_width=True)
        with data_b:
            st.markdown("#### Normalized Data Used by Model")
            st.dataframe(preview(clean_df), use_container_width=True)

    with research_tabs[1]:
        st.markdown("#### Synthetic Data Preview")
        try:
            synthetic_preview = generate_synthetic(
                clean_df,
                n_rows=min(20, max(int(synth_rows), 0)),
                method="bootstrap",
            )
            st.dataframe(
                preview(synthetic_preview),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.warning(f"Could not generate synthetic preview: {exc}")

    with research_tabs[2]:
        st.markdown("#### Required Model Features")
        st.write(ALL_FEATURES)