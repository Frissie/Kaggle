#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import pickle
import os
import warnings

from iDirectory import data_dir, sub_dir, model_dir

from scipy.optimize import nnls
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import set_random_seed
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Disable warnings
warnings.filterwarnings("ignore")


folds = 5
cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

train = pd.read_csv(data_dir + "train.csv", index_col=0)
test = pd.read_csv(data_dir + "test.csv", index_col=0)
orig = pd.read_csv(data_dir + "loan_dataset_20000.csv")

for col in train.select_dtypes(include="object").columns:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")
for col in orig.select_dtypes(include="object").columns:
    orig[col] = orig[col].astype("category")

TARGET_COL = "loan_paid_back"
y = train[TARGET_COL]
X = train.drop(columns=TARGET_COL)
X_pred = test

y_orig = orig[TARGET_COL]
X_orig = orig[X.columns]  # align columns


# In[32]:


class Features(BaseEstimator, TransformerMixin):
    """
    Feature engineering transformer with three modes:

    - model=None (default):
        * Adds numeric quantile bins as categorical features: <col>_Quartile
        * Adds target-mean encodings for categoricals: <col>_risk
        * Keeps original columns as well.

    - model="category":
        * Returns only categorical features: original categoricals + <col>_Quartile

    - model="linear":
        * Returns only numeric features: original numerics + <col>_risk

    Notes:
    - No target column is added into df.
    - Target encoding is done per fold via CV (fit on train fold, transform on val).
    """

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
        df['subgrade'] = df['grade_subgrade'].str[1:].astype(int)
        df['grade'] = df['grade_subgrade'].str[0].astype("category")
        df['total_debt_burden'] = (df['loan_amount'] * df['interest_rate'] / 100) / (df['annual_income'] + 1) 

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


# In[33]:


X_inspection = Features().fit_transform(X, y)


# In[34]:


categorical_cols = X.select_dtypes(include="category").columns.tolist()
numeric_cols = X.select_dtypes(exclude="category").columns.tolist()

# Sparse View: numerics + one-hot for categoricals
SparseView = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)


# In[35]:


class LinearThenPoly(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2, interaction_only=True):
        self.degree = degree
        self.interaction_only = interaction_only
        self.linear_features_ = None
        self.poly_ = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=False,
        )

    def fit(self, X, y=None):
        # First transform with linear Features
        linear = Features(model="linear")
        linear.fit(X, y)
        X_lin = linear.transform(X)
        self.linear_features_ = linear
        self.poly_.fit(X_lin)
        return self

    def transform(self, X):
        X_lin = self.linear_features_.transform(X)
        X_poly = self.poly_.transform(X_lin)
        return X_poly


# In[36]:


class CatBoostWrapper(BaseEstimator):
    def __init__(self, **params):
        self.params = params
        self.model = None

    def fit(self, X, y):
        cat_idx = list(range(X.shape[1]))  # all columns are categorical strings
        self.model = CatBoostClassifier(**self.params, cat_features=cat_idx, verbose=False)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, prettified=True):
        return self.model.get_feature_importance(prettified=prettified)


# In[37]:


class KerasLinear(BaseEstimator):
    """
    Neural network base learner for the LINEAR view.
    ✔ Works with Features(model="linear")
    ✔ Automatically scales numeric inputs
    ✔ OOF safe (no leakage)
    ✔ predict_proba() returns 2-cols like sklearn
    """

    def __init__(
        self,
        hidden_units=128,
        dropout=0.20,
        lr=1e-3,
        batch_size=256,
        epochs=200,
        random_state=42,
    ):
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = None

    def _build_model(self, input_dim):
        model = Sequential(
            [
                Input(shape=(input_dim,)),
                Dense(self.hidden_units, activation="relu"),
                BatchNormalization(),
                Dropout(self.dropout),
                Dense(self.hidden_units, activation="relu"),
                BatchNormalization(),
                Dropout(self.dropout),
                Dense(self.hidden_units // 2, activation="relu"),
                BatchNormalization(),
                Dropout(self.dropout),
                Dense(self.hidden_units // 2, activation="relu"),
                BatchNormalization(),
                Dropout(self.dropout),
                Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(loss="binary_crossentropy", optimizer=Adam(self.lr), metrics=["AUC"])
        return model

    def fit(self, X, y):
        set_random_seed(self.random_state)

        X_scaled = self.scaler.fit_transform(X)

        self.model = self._build_model(X_scaled.shape[1])

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=0),
        ]

        self.model.fit(
            X_scaled,
            y,
            validation_split=0.15,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=0,
            callbacks=callbacks,
        )
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        p = self.model.predict(X_scaled, verbose=0).reshape(-1)
        return np.column_stack([1 - p, p])


# In[38]:


def pipeline_hash(pipeline):
    params = pipeline.get_params(deep=False)

    safe = {}
    for k, v in params.items():
        if isinstance(v, (int, float, str, bool, tuple)):
            safe[k] = v
        else:
            # Only keep the CLASS NAME, not the object
            safe[k] = v.__class__.__name__

    return hashlib.md5(pickle.dumps(safe)).hexdigest()


# In[39]:


def fit_with_oof_cached(name, pipeline, X, y, cv, cache_dir=model_dir):
    """
    Train model sequentially with OOF predictions.
    Uses caching: loads saved results when hash matches.
    """

    # Create directory for this model
    new_model_dir = os.path.join(cache_dir, name)
    os.makedirs(new_model_dir, exist_ok=True)

    # Compute hash
    current_hash = pipeline_hash(pipeline)
    hash_path = os.path.join(new_model_dir, "hash.txt")
    oof_path = os.path.join(new_model_dir, "oof.npy")
    model_path = os.path.join(new_model_dir, "model.pkl")

    # ----------------------
    # LOAD FROM CACHE
    # ----------------------
    if os.path.exists(hash_path) and os.path.exists(oof_path) and os.path.exists(model_path):
        saved_hash = open(hash_path).read().strip()

        if saved_hash == current_hash:
            print(f"\n[LOAD] {name}: Cached model found. Skipping training.")
            oof = np.load(oof_path)
            final_model = pickle.load(open(model_path, "rb"))
            return oof, final_model

        else:
            print(f"\n[INFO] {name}: Pipeline changed → retraining.")
    else:
        print(f"\n[TRAIN] {name}: No cache → training model.")

    # ----------------------
    # TRAIN SEQUENTIALLY
    # ----------------------
    oof = np.zeros(len(X))

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"{name} – Fold {fold + 1}/{cv.n_splits}")

        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = clone(pipeline)
        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict_proba(X_val)[:, 1]

    # Train final on full data
    final_model = clone(pipeline)
    final_model.fit(X, y)

    # ----------------------
    # SAVE CACHE
    # ----------------------
    np.save(oof_path, oof)
    pickle.dump(final_model, open(model_path, "wb"))
    open(hash_path, "w").write(current_hash)

    print(f"[SAVED] {name}: Model + OOF cached.\n")

    return oof, final_model


# In[40]:


def train_all_cached(base_models, X, y, cv):
    oof_dict = {}
    final_dict = {}

    for name, pipe in base_models.items():
        oof, final_model = fit_with_oof_cached(name, pipe, X, y, cv)
        oof_dict[name] = oof
        final_dict[name] = final_model

    print("\nAll models trained (cached when possible).")
    return oof_dict, final_dict


# In[41]:


# Base View: Features() -> XGB
xgb_base = Pipeline(
    [
        ("features", Features()),
        (
            "model",
            XGBClassifier(
                random_state=42,
                booster="dart",
                n_estimators=2500,
                learning_rate=0.03,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                enable_categorical=True,
                device="gpu",
                n_jobs=7,
            ),
        ),
    ]
)

# Interaction View: Linear -> Poly -> XGB
xgb_inter = Pipeline(
    [
        ("features_poly", LinearThenPoly(degree=2, interaction_only=True)),
        (
            "model",
            XGBClassifier(
                random_state=42,
                n_estimators=2000,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                enable_categorical=True,
                tree_method="hist",
                device="gpu",
                n_jobs=7,
            ),
        ),
    ]
)

cat_default = Pipeline(
    [
        ("features", Features(model="category")),
        (
            "model",
            CatBoostWrapper(
                random_state=42,
                n_estimators=1500,
                learning_rate=0.05,
                depth=8,
                thread_count=7,
            ),
        ),
    ]
)

# Linear View: Features(linear) -> HistGB
hgb_linear = Pipeline(
    [
        ("features", Features(model="linear")),
        (
            "model",
            HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.05,
                max_iter=700,
                random_state=42,
                l2_regularization=1.0,
            ),
        ),
    ]
)

# Linear View: Features(linear) -> LogisticRegression
lr_linear = Pipeline(
    [
        ("features", Features(model="linear")),
        (
            "model",
            LogisticRegression(
                random_state=42,
                penalty="l2",
                solver="lbfgs",
                class_weight="balanced",
                max_iter=3000,
                n_jobs=7,
            ),
        ),
    ]
)

# Sparse View: ColumnTransformer (OHE) -> LogisticRegression (L1)
lr_sparse = Pipeline(
    [
        ("sparse", SparseView),
        (
            "model",
            LogisticRegression(
                random_state=42,
                penalty="l1",
                solver="saga",
                class_weight="balanced",
                max_iter=3000,
                n_jobs=7,
            ),
        ),
    ]
)

# linear View: Features() -> ExtraTrees (robust bagging)
et_base = Pipeline(
    [
        ("features", Features(model="linear")),
        (
            "model",
            ExtraTreesClassifier(
                n_estimators=1000,
                max_depth=None,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=7,
            ),
        ),
    ]
)

keras_linear = Pipeline(
    [
        ("features", Features(model="linear")),
        (
            "keras",
            KerasLinear(
                hidden_units=128,
                dropout=0.1,
                lr=1e-3,
                batch_size=256,
                epochs=300,
            ),
        ),
    ]
)

cat_linear = Pipeline(
    [
        ("features", Features(model="linear")),
        (
            "model",
            CatBoostClassifier(
                depth=6,
                learning_rate=0.03,
                n_estimators=2000,
                l2_leaf_reg=3.0,
                thread_count=7,
                random_state=42,
                verbose=False,
            ),
        ),
    ]
)

lgb_base = Pipeline(
    [
        ("features", Features()),
        (
            "model",
            LGBMClassifier(
                n_estimators=2500,
                learning_rate=0.03,
                num_leaves=64,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=2,
                reg_lambda=2,
                categorical_feature="auto",
                random_state=42,
                n_jobs=7,
                verbose=-1,
            ),
        ),
    ]
)

base_models = {
    # "xgb_base": xgb_base,
    "lgb_base": lgb_base,
    "xgb_inter": xgb_inter,
    # "cat_default": cat_default,
    # "hgb_linear": hgb_linear,
    # "lr_linear": lr_linear,
    # "lr_sparse": lr_sparse,
    "et_base": et_base,
    "keras_linear": keras_linear,
    "cat_linear": cat_linear,
}


# In[42]:


oof_dict, final_models = train_all_cached(base_models, X, y, cv)


print("\nAll base models trained successfully.\n")

# Level-1 OOF matrix
predictionMatrix = pd.DataFrame(oof_dict, index=X.index)

print("\nBase model performance (OOF AUC & Accuracy):")
for name in predictionMatrix.columns:
    auc = roc_auc_score(y, predictionMatrix[name])
    acc = accuracy_score(y, (predictionMatrix[name] > 0.5).astype(int))
    print(f"{name:12} - ROC_AUC: {auc:.5f} - Acc: {acc:.5f}")

plt.figure(figsize=(10, 8))
sns.heatmap(
    data=pd.concat([predictionMatrix, y.rename(TARGET_COL)], axis=1).corr().abs(),
    cmap="icefire",
    annot=True,
    fmt=".2f",
)
plt.title("Correlation between base model OOF preds and target")
plt.show()


# In[43]:


# Meta model 1: Logistic Regression
meta_lr = LogisticRegression(
    random_state=42,
    max_iter=3000,
    C=3.0,
    penalty="elasticnet",
    solver="saga",
    class_weight="balanced",
    l1_ratio=0.3,
)
lr_trans = Features(model="linear")
lr_pred_matrix = pd.concat([lr_trans.fit_transform(X,y), predictionMatrix], axis=1)

meta_lr.fit(predictionMatrix, y)
meta_lr_oof = meta_lr.predict_proba(predictionMatrix)[:, 1]
meta_lr_auc = roc_auc_score(y, meta_lr_oof)
meta_lr_acc = accuracy_score(y, (meta_lr_oof > 0.5).astype(int))
print(f"\nMETA_LR  - ROC_AUC: {meta_lr_auc:.5f} - Acc: {meta_lr_acc:.5f}")

# Meta model 2: XGB on OOF predictions
meta_xgb = XGBClassifier(
    random_state=42,
    n_estimators=800,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.9,
    enable_categorical=True,
    tree_method="hist",
    device="gpu",
    n_jobs=7,
)
weights, _ = nnls(predictionMatrix.values, y)
w = weights / weights.sum()

meta_xgb.fit(lr_pred_matrix, y)
meta_xgb_oof = meta_xgb.predict_proba(lr_pred_matrix)[:, 1]
meta_xgb_auc = roc_auc_score(y, meta_xgb_oof)
meta_xgb_acc = accuracy_score(y, (meta_xgb_oof > 0.5).astype(int))
print(f"META_XGB - ROC_AUC: {meta_xgb_auc:.5f} - Acc: {meta_xgb_acc:.5f}")

# Simple blender on train: average of meta models
meta_blend_oof = (predictionMatrix.values * w).sum(axis=1)
meta_blend_auc = roc_auc_score(y, meta_blend_oof)
meta_blend_acc = accuracy_score(y, (meta_blend_oof > 0.5).astype(int))
print(f"BLEND    - ROC_AUC: {meta_blend_auc:.5f} - Acc: {meta_blend_acc:.5f}")


# In[44]:


# Level-1 predictions on test
PredProbMatrix = pd.DataFrame(
    {name: model.predict_proba(X_pred)[:, 1] for name, model in final_models.items()},
    index=X_pred.index,
)

lr_pred_Prob_matrix = pd.concat([lr_trans.transform(X_pred), PredProbMatrix], axis=1)


# Level-2 predictions on test
meta_lr_test = meta_lr.predict_proba(PredProbMatrix)[:, 1]
meta_xgb_test = meta_xgb.predict_proba(lr_pred_Prob_matrix)[:, 1]
meta_blend_test = (PredProbMatrix.values * w).sum(axis=1)

# Final submission (you can also submit meta_lr_test or meta_xgb_test separately)
submission = pd.DataFrame(
    {TARGET_COL: meta_blend_test},
    index=X_pred.index,
)
submission.reset_index().to_parquet(sub_dir + "submission_elite_stack.parquet")

submission_xgb = pd.DataFrame(
    {TARGET_COL: meta_xgb_test},
    index=X_pred.index,
)
submission_xgb.reset_index().to_parquet(sub_dir + "submission_elite_stack_xgb.parquet")

submission_lr = pd.DataFrame(
    {TARGET_COL: meta_lr_test},
    index=X_pred.index,
)
submission_lr.reset_index().to_parquet(sub_dir + "submission_elite_stack_lr.parquet")
print("\nSaved submission to:", sub_dir + "submission_elite_stack.parquet")


# In[45]:


print("======================")
print(" DIAGNOSTICS MODULE ")
print("======================\n")


print("1. BASE MODEL OOF PERFORMANCE\n")
auc_scores = {name: roc_auc_score(y, predictionMatrix[name]) for name in predictionMatrix.columns}
auc_df = pd.DataFrame.from_dict(auc_scores, orient="index", columns=["AUC"])
auc_df = auc_df.sort_values("AUC", ascending=False)
print(auc_df, "\n")


print("2. CORRELATION BETWEEN BASE MODELS\n")

plt.figure(figsize=(10, 8))
sns.heatmap(predictionMatrix.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("OOF Prediction Correlation (Base Models)")
plt.show()

print("\nLower correlation = better stacking diversity.\n")


print("3. MUTUAL INFORMATION WITH TARGET\n")

mi_scores = mutual_info_classif(predictionMatrix, y, discrete_features=False, random_state=42)

mi_df = pd.DataFrame(mi_scores, index=predictionMatrix.columns, columns=["MutualInformation"]).sort_values("MutualInformation", ascending=False)

print(mi_df, "\n")



print("4. MARGINAL CONTRIBUTION OF EACH MODEL (DROP TEST)\n")


def stack_auc(models):
    meta_temp = LogisticRegression(max_iter=2000, penalty="l2", solver="lbfgs")
    meta_temp.fit(predictionMatrix[models], y)
    return roc_auc_score(y, meta_temp.predict_proba(predictionMatrix[models])[:, 1])


base_models_list = list(predictionMatrix.columns)
full_auc = stack_auc(base_models_list)

drop_results = {}
for m in base_models_list:
    reduced = [x for x in base_models_list if x != m]
    auc_drop = stack_auc(reduced)
    drop_results[m] = full_auc - auc_drop  # positive = helpful

drop_df = pd.DataFrame.from_dict(drop_results, orient="index", columns=["MarginalContribution"]).sort_values("MarginalContribution", ascending=False)

print(drop_df, "\n")
print("Interpretation: High-positive means the model is valuable.\n")


print("5. STACKING GAIN OVER BEST SINGLE MODEL\n")

best_single = auc_df.iloc[0]["AUC"]
gain = meta_blend_auc - best_single
print(f"Best single model: {best_single:.5f}")
print(f"Stack (blend) AUC: {meta_blend_auc:.5f}")
print(f"Gain from stacking: {gain:.5f}\n")


print("6. OPTIMAL BLENDING WEIGHTS (NNLS)\n")


X_mat = predictionMatrix.values
y_vec = y.values

weights, _ = nnls(X_mat, y_vec)

blend_df = pd.DataFrame(weights / weights.sum(), index=predictionMatrix.columns, columns=["OptimalWeight"]).sort_values(
    "OptimalWeight", ascending=False
)

print(blend_df, "\n")


print("7. VIEW EFFECTIVENESS\n")

view_map = {
    "xgb_base": "BaseView",
    "xgb_inter": "InteractionView",
    "cat_default": "CatView",
    "hgb_linear": "LinearView",
    "lr_linear": "LinearView",
    "lr_sparse": "SparseView",
    "et_base": "BaseView",
    "keras_linear": "LinearView",
}

# Use the AUCs you already computed above
view_df = pd.DataFrame({
    "model": predictionMatrix.columns,
    "view": predictionMatrix.columns.map(lambda c: view_map.get(c, "Unknown")),
    "AUC": [auc_scores[m] for m in predictionMatrix.columns],
})

print(view_df, "\n")
print("Mean AUC per view:\n", view_df.groupby("view")["AUC"].mean(), "\n")



print("8. MODEL DIVERSITY SCORE (1 - |correlation| means diversity)\n")

corr = predictionMatrix.corr().abs()
diversity = 1 - corr
np.fill_diagonal(diversity.values, np.nan)

div_df = pd.DataFrame(diversity.mean(), columns=["DiversityScore"]).sort_values("DiversityScore", ascending=False)

print(div_df, "\n")
print("Higher score = more diverse = better stacking synergy.\n")

print("=== DIAGNOSTICS DONE ===")

