"""Model builders shared by the runtime and failure scripts. Every learned preprocessing step is fit on the
rows passed to .fit() (the fold's fit set, or all development rows for the final model), never on validation/test."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from config import SEED
from features import CATEGORICAL, HEAVY, TOP_N, CategoryEncoder, safe_log1p


class TreeModel:
    """CategoryEncoder (train-only top-N + OTHER) followed by a HistGradientBoosting model with native categoricals."""

    def __init__(self, cols, kind="reg", **params):
        self.cols, self.kind, self.params = list(cols), kind, params

    def fit(self, df, y, sample_weight=None):
        cats = [c for c in self.cols if c in CATEGORICAL]
        self.enc = CategoryEncoder().fit(df, cats)
        X = self.enc.transform(df[self.cols])
        Est = HistGradientBoostingRegressor if self.kind == "reg" else HistGradientBoostingClassifier
        self.model = Est(categorical_features="from_dtype", random_state=SEED, early_stopping=False, **self.params)
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, df):
        X = self.enc.transform(df[self.cols])
        if self.kind == "reg":
            return self.model.predict(X)
        return self.model.predict_proba(X)[:, 1]


def linear_preprocessor(cols):
    """log1p on heavy-tailed counts, median imputation with missing indicators, standard scaling; one-hot categoricals.
    Categorical levels: top-N by train frequency (handled by OneHotEncoder max_categories), unseen -> ignored."""
    cats = [c for c in cols if c in CATEGORICAL]
    nums = [c for c in cols if c not in CATEGORICAL]
    heavy = [c for c in nums if c in HEAVY]
    light = [c for c in nums if c not in heavy]
    parts = []
    if heavy:
        parts.append(("heavy", make_pipeline(FunctionTransformer(safe_log1p, feature_names_out="one-to-one"),
                                             SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), heavy))
    if light:
        parts.append(("light", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), light))
    if cats:
        parts.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=TOP_N + 1,
                                           min_frequency=None), cats))
    return ColumnTransformer(parts, sparse_threshold=0.3)


class LinearModel:
    def __init__(self, cols, kind="reg", **params):
        self.cols, self.kind, self.params = list(cols), kind, params

    def fit(self, df, y, sample_weight=None):
        X = df[self.cols].copy()
        for c in self.cols:
            if c in CATEGORICAL:
                X[c] = X[c].astype(str)
        est = Ridge(**self.params) if self.kind == "reg" else LogisticRegression(max_iter=500, random_state=SEED, **self.params)
        self.pipe = Pipeline([("prep", linear_preprocessor(self.cols)), ("est", est)])
        self.pipe.fit(X, y, est__sample_weight=sample_weight)
        return self

    def predict(self, df):
        X = df[self.cols].copy()
        for c in self.cols:
            if c in CATEGORICAL:
                X[c] = X[c].astype(str)
        if self.kind == "reg":
            return self.pipe.predict(X)
        return self.pipe.predict_proba(X)[:, 1]


def clip_prob(p):
    return np.clip(p, 1e-6, 1 - 1e-6)
