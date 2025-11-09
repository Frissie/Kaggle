import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder, StandardScaler


class Features(BaseEstimator, TransformerMixin):
    def __init__(self):
        self

    def fit(self, df, y=None):
        df = df.copy()
        if isinstance(y, pd.Series):
            y = y.copy()
        else:
            y = pd.Series(data=y, index=df.index, name="y")

        if y is not None:
            df["loan_paid_back"] = y.values
            self.risk_employment = df.groupby("employment_status", observed=True)["loan_paid_back"].mean()
            self.risk_subgrade = df.groupby("grade_subgrade", observed=True)["loan_paid_back"].mean()

        self.debt_to_income_ratio_mean = df["debt_to_income_ratio"].mean()

        return self

    def transform(self, df):
        df = df.copy()

        df["annual_incomeXcredit_score"] = df["annual_income"] * df["credit_score"]
        df["loan_to_income"] = df["loan_amount"] / (df["annual_income"] + 1)
        df["interest_burden"] = (df["loan_amount"] * df["interest_rate"]) / (df["annual_income"] + 1)
        df["log_income"] = np.log1p(df["annual_income"])
        df["log_loan_amount"] = np.log1p(df["loan_amount"])

        df["debt_to_income_ratio_diff"] = df["debt_to_income_ratio"] - self.debt_to_income_ratio_mean
        df["debt_to_income_ratio_norm"] = df["debt_to_income_ratio"] / self.debt_to_income_ratio_mean

        if self.risk_employment is not None:
            df["risk_employment_map"] = df["employment_status"].map(self.risk_employment).astype(float)
        if self.risk_subgrade is not None:
            df["risk_grade_map"] = df["grade_subgrade"].map(self.risk_subgrade).astype(float)

        df["credit_dti_interaction"] = df["credit_score"] / (df["debt_to_income_ratio"] + 1)
        df["income_dti_interaction"] = df["annual_income"] / (df["debt_to_income_ratio"] + 1)

        df = df.drop(
            [
                "loan_purpose",
                "gender",
                "education_level",
                "marital_status",
            ],
            axis=1,
        )

        return df


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if self.columns is None:
            return X

        drop = X.drop(columns=self.columns, errors="ignore").columns.to_list()

        return X.drop(columns=drop, errors="ignore")


columnTransformerSelector = ColumnTransformer(
    [
        (
            "int",
            Pipeline([("scaler", MinMaxScaler())]),
            make_column_selector(dtype_exclude="category"),
        ),
        (
            "cat",
            Pipeline(
                [
                    (
                        "encoder",
                        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    )
                ]
            ),
            make_column_selector(dtype_include="category"),
        ),
    ]
).set_output(transform="pandas")

columnTransformerLinear = ColumnTransformer(
    [
        (
            "int",
            Pipeline([("scaler", StandardScaler())]),
            make_column_selector(dtype_exclude="category"),
        ),
        (
            "cat",
            Pipeline(
                [
                    (
                        "encoder",
                        OneHotEncoder(drop="first", sparse_output=False),
                    )
                ]
            ),
            make_column_selector(dtype_include="category"),
        ),
    ]
).set_output(transform="pandas")


FeaturesToNumericalPipeline = Pipeline(
    [
        ("feature", Features()),
        ("transformer", columnTransformerSelector),
        ("selector", ColumnSelector(columns=None)),
    ]
)
FeaturesToLinearPipeline = Pipeline(
    [
        ("feature", Features()),
        ("transformer", columnTransformerLinear),
        ("selector", ColumnSelector(columns=None)),
    ]
)

xgb_params_search = {
    "xgb__booster": ["gbtree", "dart"],
    "xgb__n_estimators": range(500, 1501, 500),
    "xgb__learning_rate": [0.1, 0.01, 0.001, 0.3, 0.03, 0.003],
    "xgb__max_depth": range(3, 12, 2),
    "xgb__min_child_weight": range(1, 10, 2),
    "xgb__gamma": range(0, 10, 1),
    "xgb__subsample": [0.5, 0.7, 0.9],
    "xgb__colsample_bytree": [0.5, 0.7, 0.9],
}
cat_params_search = {
    "cat__n_estimators": range(500, 1501, 500),
    "cat__learning_rate": [0.1, 0.01, 0.001, 0.3, 0.03, 0.003],
    "cat__depth": range(4, 11, 2),
    "cat__l2_leaf_reg": range(1, 10, 2),
    # "cat__bagging_temperature": [0, 1],
    # "cat__rsm": [0.5, 0.7, 0.9],
    "cat__bootstrap_type": ["Bayesian", "Bernoulli", "MVS"],
}
lgb_params_search = {
    "lgb__n_estimators": range(500, 1501, 500),
    "lgb__learning_rate": [0.1, 0.01, 0.001, 0.3, 0.03, 0.003],
    "lgb__num_leaves": range(16, 257, 16),
    "lgb__max_depth": range(3, 15, 2),
    "lgb__min_child_samples": range(5, 26, 10),
    "lgb__feature_fraction": [0.6, 0.8, 1.0],
    "lgb__bagging_fraction": [0.6, 0.8, 1.0],
    "lgb__bagging_freq": range(0, 10, 3),
    "lgb__min_split_gain": range(0, 10, 3),
    "lgb__lambda_l1": [0, 0.1, 0.5, 1, 5],
    "lgb__lambda_l2": [0, 0.1, 0.5, 1, 5],
    "lgb__boosting_type": ["gbdt", "goss", "dart"],
}
hgb_params_search = {
    "hgb__max_iter": range(500, 1501, 500),
    "hgb__learning_rate": [0.1, 0.01, 0.001, 0.3, 0.03, 0.003],
    "hgb__max_depth": range(3, 15, 2),
    "hgb__min_samples_leaf": range(5, 100, 30),
    "hgb__l2_regularization": range(1, 10, 2),
}
