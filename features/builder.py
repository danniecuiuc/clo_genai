from __future__ import annotations

import pandas as pd

from common.config import DATE_COLUMNS


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in DATE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors='coerce')
            out[f'{col}_year'] = out[col].dt.year
            out[f'{col}_month'] = out[col].dt.month
            out[f'{col}_dayofyear'] = out[col].dt.dayofyear

    if all(col in out.columns for col in DATE_COLUMNS):
        out['seasoning_days'] = (out['Trade Date'] - out['Closing Date']).dt.days

    return out


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {'Attach', 'Thickness'}.issubset(out.columns):
        denom = out['Thickness'].replace(0, pd.NA)
        out['attach_to_thickness'] = out['Attach'] / denom

    if {'MVOC', 'MVOC (w/o X)'}.issubset(out.columns):
        denom = out['MVOC (w/o X)'].replace(0, pd.NA)
        out['mvoc_gap_ratio'] = (out['MVOC'] - out['MVOC (w/o X)']) / denom

    if {'Cover Price', 'AAA Coupon'}.issubset(out.columns):
        denom = out['AAA Coupon'].replace(0, pd.NA)
        out['cover_price_to_coupon'] = out['Cover Price'] / denom

    if {'Excess Spread', 'Deal WACC'}.issubset(out.columns):
        denom = out['Deal WACC'].replace(0, pd.NA)
        out['excess_spread_minus_wacc'] = out['Excess Spread'] - out['Deal WACC']
        out['excess_spread_to_wacc'] = out['Excess Spread'] / denom

    return out


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = add_date_features(df)
    out = add_ratio_features(out)
    return out
