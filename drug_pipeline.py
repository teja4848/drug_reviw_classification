# housing_pipeline.py
"""
Shared ML pipeline components for the housing project.

This module holds all custom transformers and helper functions that are used
both in training and in inference (FastAPI app), so that joblib pickles
refer to a stable module path: `housing_pipeline.<name>`.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# =============================================================================
# Custom transformer and helper functions
# =============================================================================

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(
            self.n_clusters,
            n_init=10,
            random_state=self.random_state
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def fit_transform(self, X, y=None, sample_weight=None):
        self.fit(X, y, sample_weight)
        return self.transform(X)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


def column_ratio(X):
    """
    Calculate ratio of first column to second column.
    Works for numpy arrays and pandas DataFrames.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    return X[:, [0]] / X[:, [1]]


def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]


def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler(),
    )


# =============================================================================
# Building blocks for preprocessing
# =============================================================================

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler(),
)

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)

default_num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
)


def build_preprocessing():
    """
    Return the ColumnTransformer preprocessing used in the housing models.
    """
    preprocessing = ColumnTransformer(
        [
            ("bedrooms",        ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
            ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
            ("people_per_house",ratio_pipeline(), ["population", "households"]),
            ("log",             log_pipeline,
                ["total_bedrooms", "total_rooms", "population",
                 "households", "median_income"]),
            ("geo",             cluster_simil, ["latitude", "longitude"]),
            ("cat",             cat_pipeline, make_column_selector(dtype_include=object)),
        ],
        remainder=default_num_pipeline,
    )
    return preprocessing


# =============================================================================
# Estimator factory used by both non-Optuna and Optuna code
# =============================================================================

def make_estimator_for_name(name: str):
    """
    Given a model name, return an unconfigured estimator instance.
    Used in PCA variants and (optionally) elsewhere.
    """
    if name == "ridge":
        return Ridge()
    elif name == "histgradientboosting":
        return HistGradientBoostingRegressor(random_state=42)
    elif name == "xgboost":
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
        )
    elif name == "lightgbm":
        return LGBMRegressor(
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")