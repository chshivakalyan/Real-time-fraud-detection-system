import os
import json
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from preprocessing import Preprocessor
from feature_engineering import create_features

# --------------------------------------------------
# 0. Paths & directories
# --------------------------------------------------
MODEL_DIR = "models"
META_PATH = os.path.join(MODEL_DIR, "model_metadata.json")
os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------
# 1. Determine model version
# --------------------------------------------------
if os.path.exists(META_PATH):
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    version = meta["latest_version"] + 1
else:
    version = 1

print(f"🚀 Training model version v{version}")

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------
train_trans = pd.read_csv(
    r"C:\Users\Lenovoideapad\Desktop\ML\DATA\train_transaction.csv"
)
train_id = pd.read_csv(
    r"C:\Users\Lenovoideapad\Desktop\ML\DATA\train_identity.csv"
)

df = train_trans.merge(train_id, on="TransactionID", how="left")
# Keep only selected columns

# --------------------------------------------------
# 3. Drop high-missing columns
# --------------------------------------------------
missing = df.isnull().mean()
df = df.drop(columns=missing[missing > 0.9].index)

# --------------------------------------------------
# 4. Drop ID column
# --------------------------------------------------
df = df.drop(columns=["TransactionID"])

# --------------------------------------------------
# 5. Feature engineering
# --------------------------------------------------
keep_cols = [
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "DeviceType",
    "P_emaildomain",
    "R_emaildomain",
    "isFraud"
]

df = df[[col for col in keep_cols if col in df.columns]]

df = create_features(df)

# --------------------------------------------------
# 6. Split target and features
# --------------------------------------------------
y = df["isFraud"]
X = df.drop(columns=["isFraud"])

# --------------------------------------------------
# 7. Train–validation split
# --------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# --------------------------------------------------
# 8. Save feature schema (CRITICAL)
# --------------------------------------------------
feature_columns = X_train.columns.tolist()

with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
    json.dump(feature_columns, f, indent=2)

# --------------------------------------------------
# 9. Preprocessing
# --------------------------------------------------
pre = Preprocessor()
pre.fit(X_train)

X_train_p = pre.transform(X_train)
X_val_p = pre.transform(X_val)

# --------------------------------------------------
# 10. Handle class imbalance
# --------------------------------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# --------------------------------------------------
# 11. XGBoost model (CPU optimized)
# --------------------------------------------------
model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

model.fit(X_train_p, y_train)

# --------------------------------------------------
# 12. Save versioned artifacts
# --------------------------------------------------
model_path = os.path.join(MODEL_DIR, f"xgb_fraud_v{version}.pkl")
pre_path = os.path.join(MODEL_DIR, f"preprocessor_v{version}.pkl")

joblib.dump(model, model_path)
joblib.dump(pre, pre_path)

# --------------------------------------------------
# 13. Update metadata (production-safe)
# --------------------------------------------------
meta = {
    "latest_version": version,
    "production_version": version
}

with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

# --------------------------------------------------
# 14. Done
# --------------------------------------------------
print(" Training completed")
print(f" Model saved as v{version}")
print(" Model marked as PRODUCTION")
