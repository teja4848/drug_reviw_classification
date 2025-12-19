"""
Drug Review Classification Pipeline
====================================
Shared ML pipeline components for drug effectiveness classification.

Target: effectiveness (5 classes)
    - Highly Effective
    - Considerably Effective  
    - Moderately Effective
    - Marginally Effective
    - Ineffective
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sqlite3

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Classification models
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier

# ==============================================================================
# Paths and Constants
# ==============================================================================

BASE_DIR = Path(__file__).parent if "__file__" in dir() else Path(".")
DB_PATH = BASE_DIR / "data" / "drug_reviews.db"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Target classes (ordered)
EFFECTIVENESS_CLASSES = [
    "Ineffective",
    "Marginally Effective",
    "Moderately Effective",
    "Considerably Effective",
    "Highly Effective"
]

# Feature column groups
TEXT_COLUMNS = ['benefits_review', 'side_effects_review', 'comments_review']
TARGET_COLUMN = 'effectiveness'


# ==============================================================================
# Custom Transformers (Module-level for pickle compatibility)
# ==============================================================================

class SimpleTextPipeline(BaseEstimator, TransformerMixin):
    """
    Simple TF-IDF pipeline for text features.
    Combines text columns and applies TF-IDF vectorization.
    """
    def __init__(self, text_columns=None, max_features=2000):
        self.text_columns = text_columns if text_columns else TEXT_COLUMNS
        self.max_features = max_features
        self.tfidf = None
    
    def fit(self, X, y=None):
        self.tfidf = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        if isinstance(X, pd.DataFrame):
            combined = X[self.text_columns].fillna("").agg(" ".join, axis=1)
        else:
            combined = [" ".join(str(x) for x in row) for row in X]
        self.tfidf.fit(combined)
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            combined = X[self.text_columns].fillna("").agg(" ".join, axis=1)
        else:
            combined = [" ".join(str(x) for x in row) for row in X]
        return self.tfidf.transform(combined)
    
    def get_feature_names_out(self, names=None):
        return self.tfidf.get_feature_names_out() if self.tfidf else None


# ==============================================================================
# Database Functions
# ==============================================================================

def get_dataframe_from_db(split: str = None, db_path: Path = None) -> pd.DataFrame:
    """Load data from normalized SQLite database."""
    if db_path is None:
        db_path = DB_PATH
    
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
        r.review_id,
        d.drug_name,
        c.condition_name,
        e.level_name as effectiveness,
        s.category_name as side_effects,
        s.severity_order as side_effects_severity,
        r.rating,
        r.benefits_review,
        r.side_effects_review,
        r.comments_review,
        r.split
    FROM reviews r
    JOIN drugs d ON r.drug_id = d.drug_id
    JOIN conditions c ON r.condition_id = c.condition_id
    JOIN effectiveness_levels e ON r.effectiveness_id = e.effectiveness_id
    JOIN side_effect_categories s ON r.side_effect_id = s.side_effect_id
    """
    
    if split:
        query += f" WHERE r.split = '{split}'"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df


def prepare_train_test_data(df: pd.DataFrame = None):
    """Prepare train and test DataFrames from database."""
    if df is None:
        df_train = get_dataframe_from_db('train')
        df_test = get_dataframe_from_db('test')
    else:
        df_train = df[df['split'] == 'train'].copy()
        df_test = df[df['split'] == 'test'].copy()
    
    feature_cols = TEXT_COLUMNS + ['drug_name', 'condition_name', 'side_effects', 
                                    'rating', 'side_effects_severity']
    
    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]
    
    y_train = df_train[TARGET_COLUMN]
    y_test = df_test[TARGET_COLUMN]
    
    return X_train, X_test, y_train, y_test


# ==============================================================================
# Preprocessing
# ==============================================================================

def build_preprocessing_simple(max_text_features: int = 2000):
    """Return SimpleTextPipeline for text preprocessing."""
    return SimpleTextPipeline(text_columns=TEXT_COLUMNS, max_features=max_text_features)


def encode_target(y_train, y_test=None):
    """Encode target labels."""
    le = LabelEncoder()
    le.fit(EFFECTIVENESS_CLASSES)
    y_train_encoded = le.transform(y_train)
    if y_test is not None:
        y_test_encoded = le.transform(y_test)
        return y_train_encoded, y_test_encoded, le
    return y_train_encoded, le


# ==============================================================================
# Model Factory
# ==============================================================================

def make_classifier(name: str, **kwargs):
    """Create a classifier by name."""
    name = name.lower()
    
    if name == "logistic":
        defaults = {"max_iter": 1000, "random_state": 42, "n_jobs": -1}
        defaults.update(kwargs)
        return LogisticRegression(**defaults)
    elif name == "ridge":
        defaults = {"random_state": 42}
        defaults.update(kwargs)
        return RidgeClassifier(**defaults)
    elif name == "histgradientboosting":
        defaults = {"random_state": 42}
        defaults.update(kwargs)
        return HistGradientBoostingClassifier(**defaults)
    elif name == "xgboost":
        defaults = {
            "objective": "multi:softprob",
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": 'mlogloss',
            "n_jobs": -1
        }
        defaults.update(kwargs)
        return XGBClassifier(**defaults)
    else:
        raise ValueError(f"Unknown classifier: {name}")


def get_model_names():
    """Return list of available model names."""
    return ["logistic", "ridge", "histgradientboosting", "xgboost"]


# ==============================================================================
# Optuna Search Spaces
# ==============================================================================

def get_optuna_search_space(model_name: str, trial, uses_pca: bool = False):
    """Define Optuna search spaces for each model."""
    model_name = model_name.lower()
    params = {}
    
    if model_name == "logistic":
        params = {
            "C": trial.suggest_float("C", 0.01, 10.0, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])
        }
    
    elif model_name == "ridge":
        params = {
            "alpha": trial.suggest_float("alpha", 0.01, 10.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])
        }
    
    elif model_name == "histgradientboosting":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
            "max_iter": trial.suggest_int("max_iter", 50, 200)
        }
    
    elif model_name == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0)
        }
    
    return params


if __name__ == "__main__":
    # Quick test
    X_train, X_test, y_train, y_test = prepare_train_test_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Target classes: {set(y_train)}")
    
    # Test preprocessing
    preprocessor = build_preprocessing_simple()
    X_train_transformed = preprocessor.fit_transform(X_train)
    print(f"Transformed shape: {X_train_transformed.shape}")
