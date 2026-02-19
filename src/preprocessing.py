import pandas as pd
from sklearn.impute import SimpleImputer

class Preprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.cat_cols = []
        self.freq_maps = {}
        self.feature_columns_ = None   # 🔥 IMPORTANT

    def fit(self, X: pd.DataFrame):
        # identify categorical columns
        self.cat_cols = X.select_dtypes(include="object").columns.tolist()

        # build frequency maps
        for col in self.cat_cols:
            self.freq_maps[col] = X[col].value_counts(normalize=True)

        # encode safely
        X_enc = self._encode_safe(X)

        # 🔥 STORE FINAL FEATURE SCHEMA
        self.feature_columns_ = X_enc.columns.tolist()

        # fit imputer
        self.imputer.fit(X_enc)

    def _encode_safe(self, X: pd.DataFrame):
        X = X.copy()

        # frequency encoding
        for col in self.cat_cols:
            if col in X.columns:
                X[col + "_freq"] = X[col].map(self.freq_maps[col])
            else:
                X[col + "_freq"] = None

        # drop raw categoricals if present
        X = X.drop(columns=[c for c in self.cat_cols if c in X.columns])

        return X

    def transform(self, X: pd.DataFrame):
        X_enc = self._encode_safe(X)

        # 🔥 CRITICAL FIX: ALIGN FEATURE SCHEMA
        X_enc = X_enc.reindex(columns=self.feature_columns_, fill_value=None)

        X_out = pd.DataFrame(
            self.imputer.transform(X_enc),
            columns=self.feature_columns_
        )
        return X_out
