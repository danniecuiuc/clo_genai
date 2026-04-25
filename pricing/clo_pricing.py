#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
CLO Tranche Pricing Model
=========================
  - Preprocessing pipeline (shared between training and inference)
  - Synthetic data generation via SDV GaussianCopulaSynthesizer
  - Similarity module (KNN-based comparable deal retrieval)
  - Pricing model (Ridge baseline + XGBoost final)
  - Feature importance and SHAP explainability
  - Confidence / uncertainty signal (residual-based)
  - Batch pricing support
  - Audit logging for reproducibility

Target variable: Cover Price
Rating scope:    BB-rated tranches (all rows in the provided dataset)

Synthesizers:   bootstrap | ctgan | glow | ddpm
Toggle RUN_* flags at the bottom to control what runs.
"""

import json
import logging
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

# ── Optional heavy imports (graceful degradation) ──────────────────────────
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed – falling back to Ridge-only mode.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from ctgan import CTGAN
    HAS_SDV = True  # reuse flag name so rest of code is unchanged
except ImportError:
    HAS_SDV = False

# ── PyTorch: auto-install a Python 3.9 compatible version if needed ────────
import sys, subprocess, types

def _stub_torch():
    """Return stub modules so class definitions parse even without torch."""
    _nn = types.ModuleType("nn")
    _nn.Module    = object
    _nn.Linear    = object
    _nn.ReLU      = object
    _nn.SiLU      = object
    _nn.Tanh      = object
    _nn.Sequential  = object
    _nn.ModuleList  = object
    _nn.functional  = types.ModuleType("functional")
    _nn.utils       = types.ModuleType("utils")
    _nn.utils.clip_grad_norm_ = lambda *a, **k: None
    _torch = types.ModuleType("torch")
    _torch.tensor    = None
    _torch.no_grad   = lambda: __import__("contextlib").nullcontext()
    _torch.zeros     = None
    _torch.randn     = None
    _torch.randn_like = None
    _torch.sqrt      = None
    _torch.exp       = None
    _torch.cat       = None
    _torch.full      = None
    _torch.randint   = None
    _torch.cumprod   = None
    _torch.linspace  = None
    _torch.utils     = types.ModuleType("utils")
    _torch.utils.data = types.ModuleType("data")
    _torch.utils.data.TensorDataset = object
    _torch.utils.data.DataLoader    = object
    _optim = types.ModuleType("optim")
    _optim.Adam = object
    return _torch, _nn, _optim

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except Exception:
    # torch ≥2.2 requires TypeIs from typing_extensions, which Python 3.9 lacks.
    # Silently install a compatible version (2.1.x is the last to support 3.9).
    print("[INFO] Installing PyTorch compatible with Python 3.9 …")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "torch==2.1.2", "torchvision==0.16.2",
            "--index-url", "https://download.pytorch.org/whl/cpu",
        ])
        import torch
        import torch.nn as nn
        import torch.optim as optim
        HAS_TORCH = True
        print("[INFO] PyTorch 2.1.2 installed — GLOW/DDPM available.")
    except Exception as e:
        print(f"[WARN] PyTorch install failed ({e}). GLOW/DDPM disabled; bootstrap will be used.")
        HAS_TORCH = False
        torch, nn, optim = _stub_torch()

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH       = "/Users/gbloch/Downloads/CLO tranche pricing/CLO_Tranche_Data.xlsx"
LOG_PATH        = "audit_log.jsonl"
MODEL_VERSION   = "v1.0"
N_COMPS         = 5          # number of comparable deals to return
N_SYNTH_ROWS    = 500        # synthetic rows to generate (set 0 to skip)
# Synthesizer choice: "ctgan" | "glow" | "ddpm" | "bootstrap"
SYNTH_METHOD    = "ddpm"
RANDOM_STATE    = 42
TEST_SIZE       = 0.15

TARGET          = "Cover Price"

NUMERIC_FEATURES = [
    "Manager Discount",
    "Spread",
    "NC Yrs left",
    "RP Yrs left",
    "MVOC",
    "MVOC (w/o X)",
    "Attach",
    "Thickness",
    "Diversity",
    "WAS %ile",
    "% <80",
    "% <50",
    "% CCC",
    "Equity NAV",
    "Eq Last Payment",
    "Junior OC",
    "Excess Spread",
    "Deal WACC",
    "AAA Coupon",
    "Palmer Square DM Index",
    "Palmer Square Price Index",
    "LSTA Index",
    "LSTA 100 Index",
    "HY CDX",
    "VIX",
]

CATEGORICAL_FEATURES = []  # Manager signal captured by Manager Discount % instead

ALL_FEATURES = NUMERIC_FEATURES  # no categoricals — Manager Discount carries manager signal

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def audit(record: dict):
    """Append a JSON record to the audit log for reproducibility."""
    record["timestamp"] = datetime.utcnow().isoformat()
    record["model_version"] = MODEL_VERSION
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load raw Excel, parse headers from row 0, coerce types."""
    log.info("Loading data from %s", path)
    raw = pd.read_excel(path, header=0)
    headers = raw.iloc[0].tolist()
    df = raw.iloc[1:].copy()
    df.columns = headers
    df = df.reset_index(drop=True)

    # Coerce numeric columns
    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse dates (not used as features but kept for audit / filtering)
    for date_col in ["Trade Date", "Closing Date"]:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Drop rows where the target is missing
    before = len(df)
    df = df.dropna(subset=[TARGET])
    log.info("Dropped %d rows with missing target; %d rows remain", before - len(df), len(df))

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

# Leave blank, did not have much success when I tried this. 

# ══════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_preprocessor() -> ColumnTransformer:
    """
    Shared preprocessing pipeline used identically in training and inference.
    Numeric only: median imputation → StandardScaler.
    Manager quality is captured by Manager Discount % (continuous) rather
    than one-hot encoding, which was shown to be noisier in ablation tests.
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# 3. SYNTHETIC DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

# ── Shared tabular preprocessing for deep generative models ───────────────

def _prepare_tensor(df: pd.DataFrame, num_cols: list, cat_cols: list):
    """
    Encode a DataFrame into a float32 tensor for deep generative models.
    Numeric columns are z-scored; categoricals are label-encoded.
    Returns (tensor, scaler_params, cat_mappings, col_order).
    """
    import sklearn.preprocessing as skp
    data = df.copy()

    # Encode categoricals to integers
    cat_mappings = {}
    for col in cat_cols:
        le = skp.LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        cat_mappings[col] = le

    # Scale numerics
    num_data = data[num_cols].astype(float)
    means = num_data.mean()
    stds  = num_data.std().replace(0, 1)
    data[num_cols] = (num_data - means) / stds

    col_order = num_cols + cat_cols
    X = torch.tensor(data[col_order].values, dtype=torch.float32)
    return X, means, stds, cat_mappings, col_order


def _tensor_to_df(
    tensor: "torch.Tensor",
    num_cols: list,
    cat_cols: list,
    cat_mappings: dict,
    means: "pd.Series",
    stds: "pd.Series",
    orig_df: pd.DataFrame,
) -> pd.DataFrame:
    """Inverse-transform a generated tensor back to a DataFrame."""
    arr = tensor.detach().cpu().numpy()
    n_num = len(num_cols)
    df_out = pd.DataFrame(arr, columns=num_cols + cat_cols)

    # Un-scale numerics
    for i, col in enumerate(num_cols):
        df_out[col] = df_out[col] * stds[col] + means[col]

    # Decode categoricals — clamp to valid label range
    for col in cat_cols:
        le = cat_mappings[col]
        idx = df_out[col].round().astype(int).clip(0, len(le.classes_) - 1)
        df_out[col] = le.inverse_transform(idx)

    return df_out


# ── 1. CTGAN ───────────────────────────────────────────────────────────────

def _synth_ctgan(subset: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    log.info("Training CTGAN on %d rows …", len(subset))
    model = CTGAN(epochs=300, verbose=False)
    model.fit(subset, discrete_columns=CATEGORICAL_FEATURES)
    synth = model.sample(n_rows)
    log.info("CTGAN done.")
    return synth


# ── 2. GLOW (RealNVP-style normalizing flow) ───────────────────────────────

class _AffineCouplingLayer(nn.Module):
    """Single affine coupling layer for a RealNVP / GLOW-style flow."""
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        half = dim // 2
        self.net = nn.Sequential(
            nn.Linear(half, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim - half),
        )
        self.scale_net = nn.Sequential(
            nn.Linear(half, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim - half), nn.Tanh(),
        )
        self.half = half

    def forward(self, x, reverse=False):
        x1, x2 = x[:, :self.half], x[:, self.half:]
        t = self.net(x1)
        s = self.scale_net(x1)
        if not reverse:
            y2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=1)
            return torch.cat([x1, y2], dim=1), log_det
        else:
            y2 = (x2 - t) * torch.exp(-s)
            return torch.cat([x1, y2], dim=1)


class _GLOWModel(nn.Module):
    def __init__(self, dim: int, n_layers: int = 8, hidden: int = 128):
        super().__init__()
        self.layers = nn.ModuleList(
            [_AffineCouplingLayer(dim, hidden) for _ in range(n_layers)]
        )

    def forward(self, x):
        log_det_total = torch.zeros(x.size(0))
        for layer in self.layers:
            x, ld = layer(x)
            log_det_total += ld
        return x, log_det_total

    def sample(self, n: int, dim: int):
        z = torch.randn(n, dim)
        x = z
        for layer in reversed(self.layers):
            x = layer(x, reverse=True)
        return x


def _synth_glow(subset: pd.DataFrame, n_rows: int, epochs: int = 500) -> pd.DataFrame:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for GLOW synthesizer.")
    log.info("Training GLOW normalizing flow on %d rows …", len(subset))
    num_cols = NUMERIC_FEATURES + [TARGET]
    cat_cols = CATEGORICAL_FEATURES

    X, means, stds, cat_mappings, col_order = _prepare_tensor(subset, num_cols, cat_cols)
    dim = X.shape[1]

    model = _GLOWModel(dim=dim, n_layers=8, hidden=128)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    dataset = torch.utils.data.TensorDataset(X)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            z, log_det = model(batch)
            # Negative log-likelihood under standard normal prior
            log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=1)
            loss = -(log_pz + log_det).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if (epoch + 1) % 100 == 0:
            log.info("  GLOW epoch %d/%d  loss=%.4f", epoch + 1, epochs, loss.item())

    model.eval()
    with torch.no_grad():
        samples = model.sample(n_rows, dim)

    synth = _tensor_to_df(samples, num_cols, cat_cols, cat_mappings, means, stds, subset)
    log.info("GLOW done.")
    return synth


# ── 3. DDPM (Denoising Diffusion Probabilistic Model) ─────────────────────

class _DDPMDenoiser(nn.Module):
    """Simple MLP denoiser for tabular DDPM."""
    def __init__(self, dim: int, hidden: int = 256, time_emb: int = 64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_emb), nn.SiLU(),
            nn.Linear(time_emb, time_emb),
        )
        self.net = nn.Sequential(
            nn.Linear(dim + time_emb, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: "torch.Tensor", t: "torch.Tensor") -> "torch.Tensor":
        t_emb = self.time_embed(t.float().unsqueeze(-1) / 1000.0)
        return self.net(torch.cat([x, t_emb], dim=-1))


def _ddpm_schedule(T: int = 1000):
    """Linear beta schedule."""
    betas  = torch.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


def _synth_ddpm(
    subset: pd.DataFrame,
    n_rows: int,
    epochs: int = 1000,
    T: int = 1000,
) -> pd.DataFrame:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for DDPM synthesizer.")
    log.info("Training TabDDPM on %d rows …", len(subset))
    num_cols = NUMERIC_FEATURES + [TARGET]
    cat_cols = CATEGORICAL_FEATURES

    X, means, stds, cat_mappings, col_order = _prepare_tensor(subset, num_cols, cat_cols)
    dim = X.shape[1]

    betas, alphas, alpha_bar = _ddpm_schedule(T)
    model     = _DDPMDenoiser(dim=dim, hidden=256, time_emb=64)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset   = torch.utils.data.TensorDataset(X)
    loader    = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for (x0,) in loader:
            optimizer.zero_grad()
            # Sample random timestep
            t = torch.randint(0, T, (x0.size(0),))
            # Forward diffusion: add noise
            ab   = alpha_bar[t].unsqueeze(1)
            eps  = torch.randn_like(x0)
            xt   = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps
            # Predict noise
            eps_pred = model(xt, t)
            loss = nn.functional.mse_loss(eps_pred, eps)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if (epoch + 1) % 200 == 0:
            log.info("  DDPM epoch %d/%d  loss=%.4f", epoch + 1, epochs, loss.item())

    # Reverse diffusion sampling
    model.eval()
    with torch.no_grad():
        x = torch.randn(n_rows, dim)
        for t_idx in reversed(range(T)):
            t_batch = torch.full((n_rows,), t_idx, dtype=torch.long)
            eps_pred = model(x, t_batch)
            beta_t   = betas[t_idx]
            alpha_t  = alphas[t_idx]
            ab_t     = alpha_bar[t_idx]
            # DDPM reverse step
            x = (1 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - ab_t)) * eps_pred
            )
            if t_idx > 0:
                x += torch.sqrt(beta_t) * torch.randn_like(x)

    synth = _tensor_to_df(x, num_cols, cat_cols, cat_mappings, means, stds, subset)
    log.info("DDPM done.")
    return synth


# ── 4. Bootstrap fallback ──────────────────────────────────────────────────

def _synth_bootstrap(subset: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    log.info("Bootstrap resampling %d synthetic rows …", n_rows)
    synth = subset.sample(n=n_rows, replace=True, random_state=RANDOM_STATE)
    for col in NUMERIC_FEATURES + [TARGET]:
        noise = np.random.normal(0, subset[col].std() * 0.05, size=n_rows)
        synth[col] = synth[col].values + noise
    return synth.reset_index(drop=True)


# ── Comparison across all synthesizers ────────────────────────────────────

def compare_synthesizers(
    full_df: pd.DataFrame,
    n_rows: int = N_SYNTH_ROWS,
    methods: list = ["bootstrap", "ctgan", "glow", "ddpm"],
    n_folds: int = 5,
) -> pd.DataFrame:
    """
    5-fold cross-validation comparison of each synthesizer × model combination.

    For each fold:
      1. Split full_df into train/val
      2. Generate synthetic data from the training fold only (no leakage)
      3. Fit Ridge and XGBoost on augmented training fold
      4. Evaluate on the held-out validation fold

    Reports mean ± std of MAE, RMSE, R² across folds for each combination.
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_records = []  # one entry per (method, model, fold)

    all_methods = ["none (baseline)"] + methods

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(full_df)):
        log.info("─── Fold %d / %d ───", fold_idx + 1, n_folds)
        fold_train = full_df.iloc[train_idx].reset_index(drop=True)
        fold_val   = full_df.iloc[val_idx].reset_index(drop=True)
        X_val = fold_val[ALL_FEATURES]
        y_val = fold_val[TARGET].astype(float)

        for method in all_methods:
            log.info("  Synthesizer: %s", method)
            try:
                if method == "none (baseline)":
                    aug = fold_train
                else:
                    synth = generate_synthetic(fold_train, n_rows=n_rows, method=method)
                    aug   = pd.concat([fold_train, synth], ignore_index=True)

                m = PricingModel()
                m.fit(aug[ALL_FEATURES], aug[TARGET].astype(float))
                metrics = m.evaluate(X_val, y_val)

                for model_name, vals in metrics.items():
                    fold_records.append({
                        "Synthesizer": method,
                        "Model":       model_name,
                        "Fold":        fold_idx + 1,
                        **vals,
                    })

            except Exception as e:
                log.warning("  Synthesizer %s failed on fold %d: %s", method, fold_idx + 1, e)
                for model_name in ["Ridge baseline", "XGBoost"]:
                    fold_records.append({
                        "Synthesizer": method,
                        "Model":       model_name,
                        "Fold":        fold_idx + 1,
                        "MAE": None, "RMSE": None, "R2": None,
                    })

    raw = pd.DataFrame(fold_records)

    # Aggregate: mean ± std across folds
    agg = (
        raw.groupby(["Synthesizer", "Model"])[["MAE", "MedAE", "RMSE", "R2"]]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.columns = [f"{m} ({s})" for m, s in agg.columns]
    agg = agg.reset_index()

    # Pivot so Ridge and XGBoost are side by side
    ridge_rows = agg[agg["Model"] == "Ridge baseline"].drop(columns="Model").add_suffix(" [Ridge]").rename(columns={"Synthesizer [Ridge]": "Synthesizer"})
    xgb_rows   = agg[agg["Model"] == "XGBoost"].drop(columns="Model").add_suffix(" [XGB]").rename(columns={"Synthesizer [XGB]": "Synthesizer"})
    summary = ridge_rows.merge(xgb_rows, on="Synthesizer")

    # Sort by mean XGBoost R²
    xgb_r2_col = [c for c in summary.columns if "R2 (mean)" in c and "XGB" in c]
    if xgb_r2_col:
        summary = summary.sort_values(xgb_r2_col[0], ascending=False)

    print("\n" + "═"*80)
    print(f"  SYNTHESIZER COMPARISON — {n_folds}-Fold Cross-Validation")
    print("═"*80)
    print("\n  Ridge Results (mean ± std across folds):")
    ridge_cols = ["Synthesizer"] + [c for c in summary.columns if "[Ridge]" in c]
    print(summary[ridge_cols].to_string(index=False))
    print("\n  XGBoost Results (mean ± std across folds):")
    xgb_cols = ["Synthesizer"] + [c for c in summary.columns if "[XGB]" in c]
    print(summary[xgb_cols].to_string(index=False))
    print("═"*80)

    summary.to_csv("synthesizer_comparison_cv.csv", index=False)
    raw.to_csv("synthesizer_comparison_cv_raw.csv", index=False)
    log.info("Saved synthesizer_comparison_cv.csv and synthesizer_comparison_cv_raw.csv")
    return summary


# ── Feature Ablation ──────────────────────────────────────────────────────

def run_ablation(
    train_df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_folds: int = 5,
) -> pd.DataFrame:
    """
    5-fold CV ablation experiment testing four feature configurations:
      1. Full model (all features including both manager signals)
      2. Categorical only (drop Manager Discount)
      3. Manager Discount only (drop categorical)
      4. Neither (drop both manager signals entirely)

    Reports mean ± std of R², MAE, RMSE for Ridge and XGBoost.
    """
    from sklearn.model_selection import KFold
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.linear_model import Ridge

    full_df = pd.concat([train_df, X_test.assign(**{TARGET: y_test})], ignore_index=True)

    configs = {
        "Full (both signals)":        (NUMERIC_FEATURES,                                    CATEGORICAL_FEATURES),
        "Categorical only":           ([f for f in NUMERIC_FEATURES if f != "Manager Discount"], CATEGORICAL_FEATURES),
        "Manager Discount only":      (NUMERIC_FEATURES,                                    []),
        "Neither (no manager signal)":([ f for f in NUMERIC_FEATURES if f != "Manager Discount"], []),
    }

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    records = []

    for config_name, (num_feats, cat_feats) in configs.items():
        log.info("Ablation config: %s", config_name)
        fold_metrics = {"Ridge baseline": [], "XGBoost": []}

        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(full_df)):
            fold_train = full_df.iloc[tr_idx].reset_index(drop=True)
            fold_val   = full_df.iloc[val_idx].reset_index(drop=True)

            y_tr  = fold_train[TARGET].astype(float)
            y_val = fold_val[TARGET].astype(float)

            # Build a preprocessor for this config
            transformers = []
            if num_feats:
                transformers.append(("num", Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc",  StandardScaler()),
                ]), num_feats))
            if cat_feats:
                transformers.append(("cat", Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]), cat_feats))

            pre = ColumnTransformer(transformers)
            all_feats = num_feats + cat_feats

            X_tr_t  = pre.fit_transform(fold_train[all_feats])
            X_val_t = pre.transform(fold_val[all_feats])

            # Ridge
            ridge = Ridge(alpha=10.0)
            ridge.fit(X_tr_t, y_tr)
            r_preds = ridge.predict(X_val_t)
            fold_metrics["Ridge baseline"].append({
                "MAE":  mean_absolute_error(y_val, r_preds),
                "RMSE": np.sqrt(mean_squared_error(y_val, r_preds)),
                "R2":   r2_score(y_val, r_preds),
            })

            # XGBoost
            if HAS_XGB:
                xgb_m = xgb.XGBRegressor(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=RANDOM_STATE, verbosity=0,
                )
                xgb_m.fit(X_tr_t, y_tr)
                x_preds = xgb_m.predict(X_val_t)
                fold_metrics["XGBoost"].append({
                    "MAE":  mean_absolute_error(y_val, x_preds),
                    "RMSE": np.sqrt(mean_squared_error(y_val, x_preds)),
                    "R2":   r2_score(y_val, x_preds),
                })

        for model_name, folds in fold_metrics.items():
            if not folds:
                continue
            arr = pd.DataFrame(folds)
            records.append({
                "Config":        config_name,
                "Model":         model_name,
                "R2 mean":       round(arr["R2"].mean(), 4),
                "R2 std":        round(arr["R2"].std(), 4),
                "MAE mean":      round(arr["MAE"].mean(), 4),
                "MAE std":       round(arr["MAE"].std(), 4),
                "RMSE mean":     round(arr["RMSE"].mean(), 4),
                "RMSE std":      round(arr["RMSE"].std(), 4),
            })

    results = pd.DataFrame(records)

    print("\n" + "═"*80)
    print("  FEATURE ABLATION — Manager Signal Experiment (5-Fold CV)")
    print("═"*80)
    for model_name in ["Ridge baseline", "XGBoost"]:
        subset = results[results["Model"] == model_name].drop(columns="Model")
        subset = subset.sort_values("R2 mean", ascending=False).reset_index(drop=True)
        print(f"\n  {model_name}:")
        print(subset.to_string(index=False))
    print("═"*80)

    results.to_csv("ablation_manager_signal.csv", index=False)
    log.info("Saved ablation_manager_signal.csv")
    return results


# ── Dispatcher ─────────────────────────────────────────────────────────────

def generate_synthetic(
    df: pd.DataFrame,
    n_rows: int = N_SYNTH_ROWS,
    method: str = SYNTH_METHOD,
) -> pd.DataFrame:
    """
    Generate synthetic tabular data using one of:
      "ctgan"     – CTGAN GAN (best for mixed tabular data)
      "glow"      – RealNVP normalizing flow (exact likelihood)
      "ddpm"      – Denoising diffusion probabilistic model (state of the art)
      "bootstrap" – Gaussian jitter resampling (fast fallback)
    """
    cols   = ALL_FEATURES + [TARGET]
    subset = df[cols].dropna()

    if n_rows == 0:
        return pd.DataFrame(columns=cols)

    if method == "ctgan":
        if not HAS_SDV:
            log.warning("CTGAN not installed – falling back to bootstrap.")
            return _synth_bootstrap(subset, n_rows)
        return _synth_ctgan(subset, n_rows)

    elif method == "glow":
        if not HAS_TORCH:
            log.warning("PyTorch not installed – falling back to bootstrap.")
            return _synth_bootstrap(subset, n_rows)
        return _synth_glow(subset, n_rows)

    elif method == "ddpm":
        if not HAS_TORCH:
            log.warning("PyTorch not installed – falling back to bootstrap.")
            return _synth_bootstrap(subset, n_rows)
        return _synth_ddpm(subset, n_rows)

    else:
        return _synth_bootstrap(subset, n_rows)


# ══════════════════════════════════════════════════════════════════════════════
# 4. SIMILARITY MODULE  (comparable deal retrieval)
# ══════════════════════════════════════════════════════════════════════════════

class SimilarityModule:
    """
    KNN-based comparable deal retrieval.
    Returns the top-N most similar historical tranches with similarity scores
    and a feature-level explanation of which attributes drove the match.
    """

    def __init__(self, n_neighbors: int = N_COMPS):
        self.n_neighbors = n_neighbors
        self.preprocessor = build_preprocessor()
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        self.df_ref: pd.DataFrame | None = None  # original rows for display

    def fit(self, df: pd.DataFrame):
        self.df_ref = df.reset_index(drop=True)
        X = self.preprocessor.fit_transform(df[ALL_FEATURES])
        self.nn.fit(X)
        return self

    def query(self, row: pd.DataFrame) -> pd.DataFrame:
        """
        Given a single-row DataFrame with the same feature columns,
        return a DataFrame of comparable deals with similarity scores.
        """
        X_q = self.preprocessor.transform(row[ALL_FEATURES])
        distances, indices = self.nn.kneighbors(X_q)

        comps = self.df_ref.iloc[indices[0]].copy()
        comps["similarity_score"] = np.round(1 / (1 + distances[0]), 4)
        comps["euclidean_distance"] = np.round(distances[0], 4)

        display_cols = (
            ["Bloomberg ID", "Collateral manager", "Manager Discount", TARGET,
             "MVOC", "Attach", "Thickness", "NC Yrs left", "RP Yrs left", "VIX",
             "similarity_score", "euclidean_distance"]
        )
        existing = [c for c in display_cols if c in comps.columns]
        return comps[existing].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. PRICING MODEL
# ══════════════════════════════════════════════════════════════════════════════

class PricingModel:
    """
    Two-stage pricing model:
      Stage 1 – Ridge regression baseline (transparent, always present)
      Stage 2 – XGBoost final model       (stronger; used when available)

    Outputs:
      • point estimate (OAS / Spread)
      • confidence band (±1 std of training residuals)
      • feature importance chart
      • optional SHAP values
    """

    def __init__(self):
        self.preprocessor = build_preprocessor()
        self.baseline = Ridge(alpha=10.0)
        self.final    = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=0,
        ) if HAS_XGB else None

        self.residual_std: float = 0.0          # fallback fixed confidence band
        self.feature_names: list[str] = []
        # Quantile models for adaptive confidence bands
        self.q_low = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=0.025,
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0,
        ) if HAS_XGB else None
        self.q_high = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=0.975,
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0,
        ) if HAS_XGB else None

    # ── training ──────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series):
        log.info("Fitting preprocessor …")
        X_t = self.preprocessor.fit_transform(X[ALL_FEATURES])

        self.feature_names = NUMERIC_FEATURES

        log.info("Training Ridge baseline …")
        self.baseline.fit(X_t, y)
        baseline_resid = y - self.baseline.predict(X_t)

        if self.final is not None:
            log.info("Training XGBoost final model …")
            self.final.fit(X_t, y)
            final_resid = y - self.final.predict(X_t)
            self.residual_std = float(final_resid.std())
            log.info("Training quantile models for adaptive confidence bands …")
            self.q_low.fit(X_t, y)
            self.q_high.fit(X_t, y)
        else:
            self.residual_std = float(baseline_resid.std())

        return self

    # ── inference ─────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> dict:
        """
        Return a dict with keys:
          point_estimate, lower_bound, upper_bound,
          baseline_estimate, feature_importance (DataFrame)
        """
        X_t = self.preprocessor.transform(X[ALL_FEATURES])

        baseline_pred = float(self.baseline.predict(X_t)[0])

        if self.final is not None:
            point = float(self.final.predict(X_t)[0])
            fi = pd.DataFrame({
                "feature": self.feature_names,
                "importance": self.final.feature_importances_,
            }).sort_values("importance", ascending=False).head(15)
        else:
            point = baseline_pred
            fi = pd.DataFrame({
                "feature": self.feature_names,
                "importance": np.abs(self.baseline.coef_),
            }).sort_values("importance", ascending=False).head(15)

        if self.q_low is not None and self.q_high is not None:
            lower = float(self.q_low.predict(X_t)[0])
            upper = float(self.q_high.predict(X_t)[0])
        else:
            lower = point - 1.96 * self.residual_std
            upper = point + 1.96 * self.residual_std

        return {
            "point_estimate":    round(point, 2),
            "lower_bound":       round(lower, 2),
            "upper_bound":       round(upper, 2),
            "baseline_estimate": round(baseline_pred, 2),
            "feature_importance": fi,
        }

    # ── evaluation ────────────────────────────────────────────────────────

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        X_t = self.preprocessor.transform(X[ALL_FEATURES])

        results = {}
        for name, mdl in [("Ridge baseline", self.baseline),
                           ("XGBoost",        self.final)]:
            if mdl is None:
                continue
            preds = mdl.predict(X_t)
            results[name] = {
                "MAE":   round(mean_absolute_error(y, preds), 4),
                "MedAE": round(median_absolute_error(y, preds), 4),
                "RMSE":  round(np.sqrt(mean_squared_error(y, preds)), 4),
                "R2":    round(r2_score(y, preds), 4),
            }
        return results

    # ── SHAP (optional) ───────────────────────────────────────────────────

    def shap_explain(self, X: pd.DataFrame, max_display: int = 15):
        if not HAS_SHAP or self.final is None:
            log.warning("SHAP unavailable or XGBoost not trained.")
            return
        X_t = self.preprocessor.transform(X[ALL_FEATURES])
        explainer = shap.TreeExplainer(self.final)
        shap_values = explainer.shap_values(X_t)
        if HAS_PLOT:
            shap.summary_plot(
                shap_values, X_t,
                feature_names=self.feature_names,
                max_display=max_display,
                show=True,
            )
        return shap_values

    # ── plots ─────────────────────────────────────────────────────────────

    def plot_feature_importance(self, top_n: int = 15):
        """
        Side-by-side feature importance:
          Left  — Ridge: absolute coefficient magnitude (normalized to sum=1)
          Right — XGBoost: gain-based feature importance (normalized to sum=1)
        This directly supports the sell-side interpretability use case:
        Ridge coefficients are linear and auditable; XGBoost shows nonlinear signal.
        """
        if not HAS_PLOT:
            return

        # Ridge importance: absolute coefficients, normalized
        ridge_raw = np.abs(self.baseline.coef_)
        ridge_norm = ridge_raw / ridge_raw.sum()
        ridge_fi = pd.DataFrame({
            "feature":    self.feature_names,
            "importance": ridge_norm,
        }).sort_values("importance", ascending=False).head(top_n)

        # XGBoost importance: gain, normalized
        if self.final is not None:
            xgb_raw  = self.final.feature_importances_
            xgb_norm = xgb_raw / xgb_raw.sum() if xgb_raw.sum() > 0 else xgb_raw
            xgb_fi = pd.DataFrame({
                "feature":    self.feature_names,
                "importance": xgb_norm,
            }).sort_values("importance", ascending=False).head(top_n)
        else:
            xgb_fi = ridge_fi.copy()

        # Print comparison table
        merged = ridge_fi.rename(columns={"importance": "Ridge (norm coef)"}).merge(
            xgb_fi.rename(columns={"importance": "XGBoost (gain)"}),
            on="feature", how="outer"
        ).fillna(0).sort_values("XGBoost (gain)", ascending=False)
        merged["Ridge (norm coef)"]  = merged["Ridge (norm coef)"].round(4)
        merged["XGBoost (gain)"]     = merged["XGBoost (gain)"].round(4)
        merged["Agreement"] = merged.apply(
            lambda r: "✓" if abs(
                merged["Ridge (norm coef)"].rank(ascending=False)[r.name] -
                merged["XGBoost (gain)"].rank(ascending=False)[r.name]
            ) <= 3 else "~", axis=1
        )
        print("\n── Feature Importance: Ridge vs XGBoost ──")
        print("  Interpretation guide:")
        print("  Ridge coef  — linear marginal effect; suitable for model validation")
        print("  XGBoost gain — nonlinear split contribution; stronger predictive signal")
        print("  Agreement ✓ — feature ranks similarly in both models (robust signal)")
        print("  Agreement ~ — feature rank differs (nonlinear / interaction effect)\n")
        print(merged.to_string(index=False))
        merged.to_csv("feature_importance_comparison.csv", index=False)
        log.info("Saved feature_importance_comparison.csv")

        # Side-by-side bar chart
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Ridge plot (sorted by Ridge importance)
        ridge_plot = ridge_fi.sort_values("importance")
        axes[0].barh(ridge_plot["feature"], ridge_plot["importance"], color="steelblue")
        axes[0].set_title("Ridge — Normalized |Coefficient|\n(Interpretable, linear, audit-ready)", fontsize=11)
        axes[0].set_xlabel("Normalized Importance")

        # XGBoost plot (sorted by XGBoost importance)
        xgb_plot = xgb_fi.sort_values("importance")
        axes[1].barh(xgb_plot["feature"], xgb_plot["importance"], color="darkorange")
        axes[1].set_title("XGBoost — Normalized Gain\n(Nonlinear, higher accuracy)", fontsize=11)
        axes[1].set_xlabel("Normalized Importance")

        plt.suptitle("Feature Importance Comparison: Ridge vs XGBoost", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig("feature_importance_comparison.png", dpi=150)
        plt.show()
        log.info("Feature importance comparison chart saved to feature_importance_comparison.png")

    def plot_actuals_vs_predicted(self, X: pd.DataFrame, y: pd.Series):
        if not HAS_PLOT:
            return
        X_t = self.preprocessor.transform(X[ALL_FEATURES])
        mdl = self.final if self.final is not None else self.baseline
        preds = mdl.predict(X_t)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y, preds, alpha=0.5, s=20)
        mn, mx = min(y.min(), preds.min()), max(y.max(), preds.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1)
        ax.set_xlabel(f"Actual {TARGET}")
        ax.set_ylabel(f"Predicted {TARGET}")
        ax.set_title("Actuals vs Predicted – Test Set")
        plt.tight_layout()
        plt.savefig("actuals_vs_predicted.png", dpi=150)
        plt.show()
        log.info("Actuals vs Predicted chart saved to actuals_vs_predicted.png")



# ══════════════════════════════════════════════════════════════════════════════
# 6. BATCH PRICING
# ══════════════════════════════════════════════════════════════════════════════

def batch_price(
    df_batch: pd.DataFrame,
    pricing_model: PricingModel,
    similarity_module: SimilarityModule,
    df_ref: pd.DataFrame,
) -> pd.DataFrame:
    """
    Price every row in df_batch.
    Returns a DataFrame with Bloomberg ID, point estimate, bounds, and top comp.
    """
    records = []
    for _, row in df_batch.iterrows():
        row_df = row.to_frame().T.reset_index(drop=True)
        result  = pricing_model.predict(row_df)
        comps   = similarity_module.query(row_df)
        top_comp = comps.iloc[0]["Bloomberg ID"] if len(comps) > 0 else "N/A"

        records.append({
            "Bloomberg ID":      row.get("Bloomberg ID", "N/A"),
            "Actual Spread":     row.get(TARGET, np.nan),
            "Predicted Spread":  result["point_estimate"],
            "Lower (95%)":       result["lower_bound"],
            "Upper (95%)":       result["upper_bound"],
            "Baseline (Ridge)":  result["baseline_estimate"],
            "Top Comp":          top_comp,
        })

        audit({
            "event":         "batch_price",
            "bloomberg_id":  row.get("Bloomberg ID", "N/A"),
            "predicted":     result["point_estimate"],
            "lower":         result["lower_bound"],
            "upper":         result["upper_bound"],
        })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# 7. SINGLE QUERY HELPER
# ══════════════════════════════════════════════════════════════════════════════

# ── Diagnostic: high coupon + high MVOC + short NC → should price ~100 ──────
DIAGNOSTIC_QUERY = {
    "Collateral manager":       "GoldenTree",   # tier-1 manager
    "Manager Discount":         -3.0,           # tight — good manager
    "Spread":                   450.0,          # relatively tight BB spread
    "NC Yrs left":              0.25,           # almost out of non-call — near call
    "RP Yrs left":              2.5,
    "MVOC":                     1.085,          # high MVOC — strong OC cushion
    "MVOC (w/o X)":             1.085,
    "Attach":                   0.082,
    "Thickness":                0.040,
    "Diversity":                85,
    "WAS %ile":                 0.60,
    "% <80":                    0.01,           # clean collateral
    "% <50":                    0.00,
    "% CCC":                    0.03,
    "Equity NAV":               0.65,           # healthy equity
    "Eq Last Payment":          0.08,
    "Junior OC":                0.045,
    "Excess Spread":            0.018,
    "Deal WACC":                0.017,
    "AAA Coupon":               0.016,          # high AAA coupon
    "Palmer Square DM Index":   650.0,
    "Palmer Square Price Index":96.0,
    "LSTA Index":               96.0,
    "LSTA 100 Index":           98.0,
    "HY CDX":                   300.0,
    "VIX":                      18.0,
}

EXAMPLE_QUERY = {
    "Collateral manager":       "GoldenTree",
    "Manager Discount":         -5.0,
    "Spread":                   570.0,
    "NC Yrs left":              1.0,
    "RP Yrs left":              3.5,
    "MVOC":                     1.055,
    "MVOC (w/o X)":             1.055,
    "Attach":                   0.078,
    "Thickness":                0.040,
    "Diversity":                80,
    "WAS %ile":                 0.50,
    "% <80":                    0.03,
    "% <50":                    0.01,
    "% CCC":                    0.06,
    "Equity NAV":               0.55,
    "Eq Last Payment":          0.07,
    "Junior OC":                0.040,
    "Excess Spread":            0.015,
    "Deal WACC":                0.019,
    "AAA Coupon":               0.013,
    "Palmer Square DM Index":   680.0,
    "Palmer Square Price Index":95.5,
    "LSTA Index":               95.5,
    "LSTA 100 Index":           97.8,
    "HY CDX":                   310.0,
    "VIX":                      21.0,
}


def price_single_query(
    query: dict,
    pricing_model: PricingModel,
    similarity_module: SimilarityModule,
) -> None:
    """Price one tranche and print a human-readable summary."""
    row_df = pd.DataFrame([query])
    result = pricing_model.predict(row_df)
    comps  = similarity_module.query(row_df)

    print("\n" + "═" * 60)
    print("  CLO TRANCHE PRICING RESULT")
    print("═" * 60)
    print(f"  Model:              {MODEL_VERSION}")
    print(f"  Manager:            {query.get('Collateral manager', 'N/A')}")
    print(f"  MVOC:               {query.get('MVOC', 'N/A'):.4f}")
    print(f"  Attach:             {query.get('Attach', 'N/A'):.4f}")
    print()
    print(f"  Predicted Cover Price:  {result['point_estimate']}")
    print(f"  95% Conf Band:          [{result['lower_bound']}, {result['upper_bound']}]")
    print(f"  Ridge Baseline:         {result['baseline_estimate']}")
    print()
    print("  Top Feature Drivers:")
    fi = result["feature_importance"].head(8)
    for _, r in fi.iterrows():
        bar = "█" * int(r["importance"] / fi["importance"].max() * 20)
        print(f"    {r['feature']:<35} {bar}")
    print()
    print(f"  Comparable Deals (top {N_COMPS}):")
    print(comps.to_string(index=False))
    print("═" * 60)

    audit({
        "event":    "single_query",
        "inputs":   query,
        "predicted": result["point_estimate"],
        "lower":    result["lower_bound"],
        "upper":    result["upper_bound"],
    })


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main(run_batch: bool = False, run_query: bool = False, run_compare: bool = False, run_ablation_flag: bool = False):
    # ── Load ────────────────────────────────────────────────────────────
    df = load_data()
    audit({"event": "data_loaded", "rows": len(df), "cols": len(df.columns)})

    # ── Train / test split ──────────────────────────────────────────────
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    log.info("Train: %d  Test: %d", len(train_df), len(test_df))

    # ── Synthetic data augmentation ─────────────────────────────────────
    if N_SYNTH_ROWS > 0:
        synth_df = generate_synthetic(train_df, n_rows=N_SYNTH_ROWS)
        train_aug = pd.concat([train_df, synth_df], ignore_index=True)
        log.info("Augmented training set: %d rows", len(train_aug))
        audit({"event": "synthetic_generated", "rows": len(synth_df)})
    else:
        train_aug = train_df
        log.info("Skipping synthetic data generation.")

    X_train = train_aug[ALL_FEATURES]
    y_train = train_aug[TARGET].astype(float)
    X_test  = test_df[ALL_FEATURES]
    y_test  = test_df[TARGET].astype(float)

    # ── Fit pricing model ───────────────────────────────────────────────
    pricing_model = PricingModel()
    pricing_model.fit(X_train, y_train)

    # ── Evaluate ────────────────────────────────────────────────────────
    metrics = pricing_model.evaluate(X_test, y_test)

    print("\n" + "═"*70)
    print("  MODEL EVALUATION (Test Set)")
    print("  Both models available — institutions can select based on validation needs")
    print("═"*70)
    print(f"  {'Model':<12} {'MAE':>6} {'MedAE':>7} {'RMSE':>7} {'R²':>8}   Notes")
    print(f"  {'-'*65}")
    notes = {
        "Ridge baseline": "Linear, fully auditable coefficients — recommended for strict model validation",
        "XGBoost": "Nonlinear, higher accuracy — recommended for best pricing estimates",
    }
    for model_name, m in metrics.items():
        note = notes.get(model_name, "")
        print(f"  {model_name:<12} {m['MAE']:>6.3f} {m['MedAE']:>7.3f} {m['RMSE']:>7.3f} {m['R2']:>8.4f}   {note}")
    print("═"*70)
    audit({"event": "evaluation", "metrics": metrics})

    # ── Feature ablation ─────────────────────────────────────────────────
    if run_ablation_flag:
        run_ablation(train_df, X_test, y_test)

    # ── Synthesizer comparison ───────────────────────────────────────────
    if run_compare:
        print("\n" + "═"*70)
        print("  SYNTHESIZER COMPARISON")
        print("  Training 5 configurations — this will take 15–20 min if GLOW/DDPM active")
        print("═"*70)
        compare_synthesizers(df)  # uses full dataset with CV — no leakage

    # ── Fit similarity module ────────────────────────────────────────────
    sim_module = SimilarityModule(n_neighbors=N_COMPS)
    sim_module.fit(train_aug)

    # ── Plots ────────────────────────────────────────────────────────────
    pricing_model.plot_feature_importance()
    pricing_model.plot_actuals_vs_predicted(X_test, y_test)

    # ── Batch pricing (test set) ─────────────────────────────────────────
    if run_batch:
        log.info("Running batch pricing on test set …")
        batch_results = batch_price(test_df, pricing_model, sim_module, train_aug)
        batch_results.to_csv("batch_results.csv", index=False)
        print("\n── Batch Pricing Results (first 10) ──")
        print(batch_results.head(10).to_string(index=False))
        log.info("Full results saved to batch_results.csv")

    # ── Single query ─────────────────────────────────────────────────────
    if run_query:
        price_single_query(EXAMPLE_QUERY, pricing_model, sim_module)

    # ── Diagnostic: high coupon + high MVOC + short NC → expect ~100 ─────
    print("\n" + "═"*60)
    print("  DIAGNOSTIC: High Coupon + High MVOC + Short NC")
    print("  Expected: cover price close to 100")
    print("═"*60)
    diag_df = pd.DataFrame([DIAGNOSTIC_QUERY])
    diag_result = pricing_model.predict(diag_df)
    print(f"  AAA Coupon:      {DIAGNOSTIC_QUERY['AAA Coupon']:.3f}  (high)")
    print(f"  MVOC:            {DIAGNOSTIC_QUERY['MVOC']:.3f}   (high)")
    print(f"  NC Yrs left:     {DIAGNOSTIC_QUERY['NC Yrs left']:.2f}   (short)")
    print(f"  Spread:          {DIAGNOSTIC_QUERY['Spread']:.0f} bps (tight)")
    print()
    print(f"  Predicted Cover: {diag_result['point_estimate']}")
    print(f"  95% Band:        [{diag_result['lower_bound']}, {diag_result['upper_bound']}]")
    print(f"  Ridge Baseline:  {diag_result['baseline_estimate']}")
    verdict = "✓ PASS" if 99.0 <= diag_result['point_estimate'] <= 101.5 else "✗ FAIL — model not capturing this regime"
    print(f"  Verdict:         {verdict}")
    print("═"*60)

    # ── Predicted vs Actual — one XGBoost model per synthesizer ─────────
    X_t = pricing_model.preprocessor.transform(X_test[ALL_FEATURES])

    # Train one XGBoost per synthesizer on the same train/test split
    synth_methods = ["none (baseline)", "bootstrap", "ctgan", "glow", "ddpm"]
    synth_preds   = {}

    for method in synth_methods:
        try:
            if method == "none (baseline)":
                aug = train_df
            else:
                aug = pd.concat(
                    [train_df, generate_synthetic(train_df, n_rows=N_SYNTH_ROWS, method=method)],
                    ignore_index=True,
                )
            m = PricingModel()
            m.fit(aug[ALL_FEATURES], aug[TARGET].astype(float))
            mdl = m.final if m.final is not None else m.baseline
            X_t_m = m.preprocessor.transform(X_test[ALL_FEATURES])
            synth_preds[method] = mdl.predict(X_t_m).round(3)
            log.info("Predictions generated for synthesizer: %s", method)
        except Exception as e:
            log.warning("Skipping %s: %s", method, e)
            synth_preds[method] = np.full(len(y_test), np.nan)

    results_df = pd.DataFrame({"Bloomberg ID": test_df["Bloomberg ID"].values,
                                "Actual Cover": y_test.values.round(3)})
    for method, preds in synth_preds.items():
        col = method.replace("none (baseline)", "No Synth")
        results_df[f"{col} Pred"]  = preds
        results_df[f"{col} Error"] = (preds - y_test.values).round(3)

    results_df = results_df.sort_values("No Synth Error", key=abs, ascending=False).reset_index(drop=True)

    print("\n── Predicted vs Actual Cover Price — XGBoost by Synthesizer (Test Set) ──")
    print(results_df.to_string(index=False))
    results_df.to_csv("predicted_vs_actual.csv", index=False)
    log.info("Saved predicted_vs_actual.csv")

    # ── SHAP (if available) ───────────────────────────────────────────────
    if HAS_SHAP and HAS_XGB:
        pricing_model.shap_explain(X_test.head(200))

    return pricing_model, sim_module, metrics


# ══════════════════════════════════════════════════════════════════════════════
# Entry point — works in both Jupyter and terminal
# Toggle these flags to control what runs:
#   RUN_BATCH : price the full test set and save batch_results.csv
#   RUN_QUERY : price the single example tranche in EXAMPLE_QUERY
# ══════════════════════════════════════════════════════════════════════════════

RUN_BATCH    = False
RUN_QUERY    = True
RUN_COMPARE  = True    # Benchmark all synthesizers vs Ridge and XGBoost (5-fold CV)
RUN_ABLATION = False   # Manager signal ablation (already done — categorical dropped)

main(run_batch=RUN_BATCH, run_query=RUN_QUERY, run_compare=RUN_COMPARE, run_ablation_flag=RUN_ABLATION)


# In[ ]:




