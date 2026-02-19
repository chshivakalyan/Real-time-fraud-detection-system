import pandas as pd
from scipy.stats import ks_2samp


# Load logs with tolerant parsing

logs = pd.read_csv(
    "logs/predictions.csv",
    engine="python",
    on_bad_lines="skip"  
)

print(f" Loaded {len(logs)} prediction rows")


# Load training data

train = pd.read_csv("DATA/train_transaction.csv")


# Drift check on TransactionAmt

if "TransactionAmt" not in train.columns:
    raise ValueError("TransactionAmt not found in training data")

# Extract TransactionAmt from feature string
def extract_amt(feature_str):
    try:
        return float(feature_str.split("TransactionAmt':")[1].split(",")[0])
    except Exception:
        return None

logs["TransactionAmt"] = logs["features"].apply(extract_amt)

train_amt = train["TransactionAmt"].dropna()
live_amt = logs["TransactionAmt"].dropna()

if len(live_amt) < 20:
    print(" Not enough live data for drift detection")
else:
    stat, p_value = ks_2samp(train_amt, live_amt)

    print(f"KS statistic: {stat:.4f}")
    print(f"P-value: {p_value:.6f}")

    if p_value < 0.05:
        print(" DRIFT DETECTED in TransactionAmt")
    else:
        print(" No drift detected")
