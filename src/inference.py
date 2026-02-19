import json
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb

from src.feature_engineering import create_features
from preprocessing import Preprocessor


# --------------------------------------------------
# Load artifacts
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
META_PATH = MODEL_DIR / "model_metadata.json"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"

if not META_PATH.exists():
    raise FileNotFoundError(f"Missing metadata: {META_PATH}")

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

production_version = int(meta["production_version"])
MODEL_PATH = MODEL_DIR / f"xgb_fraud_v{production_version}.pkl"
PRE_PATH = MODEL_DIR / f"preprocessor_v{production_version}.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model artifact: {MODEL_PATH}")
if not PRE_PATH.exists():
    raise FileNotFoundError(f"Missing preprocessor artifact: {PRE_PATH}")
if not FEATURES_PATH.exists():
    raise FileNotFoundError(f"Missing feature columns file: {FEATURES_PATH}")

model = joblib.load(MODEL_PATH)
pre: Preprocessor = joblib.load(PRE_PATH)

with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    FEATURE_COLS = json.load(f)

EXPECTED_MODEL_FEATURES = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "TransactionAmt_log",
    "ProductCD_freq",
    "card4_freq",
    "card6_freq",
    "DeviceType_freq",
    "P_emaildomain_freq",
    "R_emaildomain_freq",
]


# --------------------------------------------------
# Prediction function
# --------------------------------------------------
def predict_transaction(txn: dict) -> float:
    """
    Takes raw transaction dictionary and returns fraud probability.
    """

    if not isinstance(txn, dict):
        raise TypeError("txn must be a dictionary of transaction fields.")

    # 1. Convert request into DataFrame
    df = pd.DataFrame([txn])

    # 2. Apply feature engineering
    df = create_features(df)

    # 3. Ensure all expected raw request fields exist (backward compatibility)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = None

    # 4. Apply preprocessing (encoding + imputation)
    X_pre = pre.transform(df)

    # 5. Force preprocessing output to DataFrame with names
    if isinstance(X_pre, pd.DataFrame):
        X_df = X_pre.copy()
    else:
        pre_feature_cols = getattr(pre, "feature_columns_", None)
        if not pre_feature_cols:
            raise ValueError(
                "Preprocessor returned a non-DataFrame output without feature names. "
                "Expected a pandas DataFrame or a fitted preprocessor with feature_columns_."
            )
        if len(pre_feature_cols) != X_pre.shape[1]:
            raise ValueError(
                "Preprocessor output width does not match feature_columns_. "
                f"output_columns={X_pre.shape[1]}, feature_columns_={len(pre_feature_cols)}"
            )
        X_df = pd.DataFrame(X_pre, columns=pre_feature_cols)

    # 6. Validate model feature schema explicitly
    expected_cols = list(EXPECTED_MODEL_FEATURES)
    current_cols = list(X_df.columns)
    missing = [c for c in expected_cols if c not in current_cols]
    extra = [c for c in current_cols if c not in expected_cols]
    if missing or extra:
        raise ValueError(
            "Feature schema mismatch before prediction. "
            f"missing_columns={missing}, extra_columns={extra}"
        )

    # 7. Enforce strict model input order
    X_df = X_df[expected_cols]
    if list(X_df.columns) != expected_cols:
        raise ValueError("Feature ordering mismatch after reindexing model input DataFrame.")

    # 8. Cross-check booster feature names when available
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None and list(model_features) != expected_cols:
        raise ValueError(
            "Model training feature names do not match expected schema. "
            f"model_features={list(model_features)}, expected_features={expected_cols}"
        )

    # 9. Ensure model input is numeric and has explicit string column names
    X_df.columns = [str(c) for c in X_df.columns]
    try:
        X_df = X_df.astype("float64")
    except Exception as exc:
        raise ValueError(f"Failed to cast model input to numeric DataFrame: {exc}") from exc

    # 10. Predict probability with strict feature validation
    try:
        prob = model.predict_proba(X_df)[0][1]
    except ValueError as exc:
        # Some Docker/runtime combos can drop DataFrame feature names internally.
        # Fallback uses DMatrix with explicit feature names while keeping validation strict.
        if "data did not contain feature names" not in str(exc):
            raise
        dmatrix = xgb.DMatrix(
            X_df,
            feature_names=expected_cols,
            nthread=1,
        )
        prob = float(model.get_booster().predict(dmatrix)[0])

    return float(prob)
