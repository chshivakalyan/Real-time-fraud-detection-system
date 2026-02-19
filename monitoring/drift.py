import pandas as pd
from scipy.stats import ks_2samp

def check_numeric_drift(train_col, live_col, alpha=0.05):
    stat, p_value = ks_2samp(
        train_col.dropna(),
        live_col.dropna()
    )
    return p_value < alpha
