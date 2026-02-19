from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import csv
import os
import json
from pathlib import Path

from src.inference import predict_transaction


app = FastAPI(title="Real-Time Fraud Detection API")


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
META_PATH = MODEL_DIR / "model_metadata.json"

if not os.path.exists(META_PATH):
    raise FileNotFoundError("model_metadata.json not found. Train a model first.")

with open(META_PATH, "r") as f:
    meta = json.load(f)

PROD_VERSION = meta["production_version"]
print(f"Loading production model v{PROD_VERSION}")


LOG_DIR = ROOT_DIR / "logs"
LOG_FILE = LOG_DIR / "predictions.csv"
os.makedirs(LOG_DIR, exist_ok=True)

def log_prediction(features, prob, decision, model_version):
    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "fraud_probability",
                "decision",
                "model_version",
                "features"
            ])

        writer.writerow([
            datetime.utcnow().isoformat(),
            round(float(prob), 6),
            decision,
            model_version,
            str(features)
        ])

# Input schema

class Transaction(BaseModel):
    TransactionAmt: float
    ProductCD: str | None = None
    card1: float | None = None
    card2: float | None = None
    card3: float | None = None
    card4: str | None = None
    card5: float | None = None
    card6: str | None = None
    DeviceType: str | None = None
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None


# Prediction endpoint

@app.post("/predict")
def predict(txn: Transaction):
    features = txn.dict()

    # ML prediction
    prob = predict_transaction(features)

    # Decision policy
    if prob < 0.3:
        decision = "allow"
    elif prob < 0.6:
        decision = "challenge"
    else:
        decision = "block"

    
    log_prediction(features, prob, decision, PROD_VERSION)

    return {
        "fraud_probability": float(prob),
        "decision": decision,
        "model_version": f"v{PROD_VERSION}"
    }

