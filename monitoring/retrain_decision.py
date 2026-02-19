import os

# These values normally come from monitoring scripts
# For now, we simulate them (or import functions later)

def should_retrain(drift_detected: bool, auc: float) -> bool:
    """
    Central retraining decision logic (Module 7.4)
    """
    if drift_detected:
        print(" Retrain triggered due to data drift")
        return True

    if auc < 0.80:
        print("Retrain triggered due to performance drop")
        return True

    print(" No retraining required")
    return False


if __name__ == "__main__":
    # Example inputs (replace with real outputs)
    drift_detected = False
    auc = 0.82

    if should_retrain(drift_detected, auc):
        os.system("python pipelines/retrain.py")
