import pandas as pd


# Load prediction logs safely

logs = pd.read_csv(
    "logs/predictions.csv",
    engine="python",
    on_bad_lines="skip"
)

print(f"Loaded {len(logs)} prediction records")


# Performance monitoring (label-free system)
MIN_SAMPLES = 20

if len(logs) < MIN_SAMPLES:
    print("Not enough data for performance monitoring")
    print("Performance monitoring skipped (no labels available)")
    exit(0)


# Placeholder for delayed labels
print("Sufficient data volume detected")
print("Waiting for delayed fraud labels to compute metrics")
