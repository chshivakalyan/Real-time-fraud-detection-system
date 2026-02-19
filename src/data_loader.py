import pandas as pd

def load_train_data(data_path):
    trans = pd.read_csv(f"{data_path}/train_transaction.csv")
    ident = pd.read_csv(f"{data_path}/train_identity.csv")
    return trans.merge(ident, on="TransactionID", how="left")
