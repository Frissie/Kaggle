import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Features(BaseEstimator, TransformerMixin):
    def __init__(self, model=None, n_quantiles=5, smoothing=10.0):
        self.n_quantiles = n_quantiles
        self.model = model
        self.smoothing = smoothing

        self.suffix_category = "_Quartile"
        self.suffix_number = "_risk"

        self.bin_edges_ = {}
        self.target_mean_ = {}
        self.global_mean_ = None

        self.number_columns_ = None
        self.category_columns_ = None

        self.rounding = {
            # "1s": 0,
            "10s": -1,
            "100s": -2,
            "1000s": -3,
        }

    def fit(self, df, y=None):
        df = df.copy()

        if y is None:
            raise ValueError("y must be provided to compute target-based encodings.")
        y = pd.Series(y, index=df.index, name="target")

        self.number_columns_ = df.select_dtypes(exclude="category").columns
        self.category_columns_ = df.select_dtypes(include="category").columns

        self.global_mean_ = y.mean()

        self.target_mean_ = {}
        for col in self.category_columns_:
            stats = pd.DataFrame({"target": y, col: df[col]}).groupby(col)["target"].agg(["mean", "count"])
            smooth = (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing) / (stats["count"] + self.smoothing)
            self.target_mean_[col] = smooth

        self.bin_edges_ = {}
        quantiles = np.linspace(0, 1, self.n_quantiles + 1)
        for col in self.number_columns_:
            series = df[col].dropna()
            if series.empty:
                self.bin_edges_[col] = None
                continue
            try:
                edges = series.quantile(quantiles).values
                if len(np.unique(edges)) < 2:
                    self.bin_edges_[col] = None
                else:
                    self.bin_edges_[col] = edges
            except Exception as e:
                print(f"Warning: Could not compute quantiles for {col}: {e}")
                self.bin_edges_[col] = None

        return self

    def transform(self, df):
        df = df.copy()

        if self.model == "category":
            for col in self.number_columns_:
                edges = self.bin_edges_.get(col)
                if edges is None:
                    continue
                labels = [f"Q{i + 1}" for i in range(self.n_quantiles)]
                df[f"{col}{self.suffix_category}"] = pd.cut(
                    df[col],
                    bins=edges,
                    labels=labels,
                    include_lowest=True,
                    duplicates="drop",
                ).astype("category")

            df = df.select_dtypes(include="category")
            df = df.astype(str)  # CatBoost-friendly
            df = df.fillna("Unknown")  # No missing

            return df

        elif self.model == "linear":
            for col in self.category_columns_:
                mapping = self.target_mean_.get(col)
                if mapping is None:
                    continue
                df[f"{col}{self.suffix_number}"] = df[col].map(mapping).astype(float).fillna(self.global_mean_)

            df = df.select_dtypes(exclude="category")
            return df

        else:
            for col in self.number_columns_:
                edges = self.bin_edges_.get(col)
                if edges is None:
                    continue
                labels = [f"Q{i + 1}" for i in range(self.n_quantiles)]
                df[f"{col}{self.suffix_category}"] = pd.cut(
                    df[col],
                    bins=edges,
                    labels=labels,
                    include_lowest=True,
                    duplicates="drop",
                ).astype("category")

            for col in self.category_columns_:
                mapping = self.target_mean_.get(col)
                if mapping is None:
                    continue
                df[f"{col}{self.suffix_number}"] = df[col].map(mapping).astype(float).fillna(self.global_mean_)

            return df
