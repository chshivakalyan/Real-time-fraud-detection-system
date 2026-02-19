import numpy as np
import pandas as pd

def create_features(df: pd.DataFrame):
    df = df.copy()
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
    return df
