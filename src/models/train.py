"""Train and evaluate NHL value models.

Models mirrored from the R project:
  - ridge     : baseline linear model (replaces bare lm)
  - xgb       : XGBoost with randomized hyperparameter search
                (mirrors the tidymodels tune_grid in Final_Project.Rmd)
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.features.build import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODELS_DIR = Path(__file__).parents[2] / "models"
CV = KFold(n_splits=5, shuffle=True, random_state=42)


# ── Preprocessor ──────────────────────────────────────────────────────────────
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")


# ── Model pipelines ───────────────────────────────────────────────────────────
def build_pipelines(X: pd.DataFrame) -> dict[str, Pipeline]:
    pre = build_preprocessor(X)
    return {
        "ridge": Pipeline([
            ("pre", pre),
            ("model", Ridge()),
        ]),
        "xgb": Pipeline([
            ("pre", build_preprocessor(X)),   # fresh clone per pipeline
            ("model", XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.75,
                colsample_bytree=0.8,
                min_child_weight=20,
                random_state=42,
                verbosity=0,
            )),
        ]),
    }


# ── XGBoost hyperparameter search (mirrors R grid_space_filling) ──────────────
XGB_PARAM_GRID = {
    "model__n_estimators":       [200, 300, 400, 500, 600],
    "model__learning_rate":      [0.01, 0.03, 0.05, 0.07, 0.10],
    "model__max_depth":          [3, 4, 5, 6],
    "model__subsample":          [0.6, 0.7, 0.75, 0.8],
    "model__colsample_bytree":   [0.6, 0.7, 0.8, 1.0],
    "model__min_child_weight":   [10, 20, 30, 50],
    "model__gamma":              [0, 5, 10, 20, 30],
}


def tune_xgb(X: pd.DataFrame, y: pd.Series, n_iter: int = 30) -> Pipeline:
    """Randomized search over XGB hyperparameters — mirrors R's tune_grid(size=30)."""
    pipe = build_pipelines(X)["xgb"]
    search = RandomizedSearchCV(
        pipe,
        param_distributions=XGB_PARAM_GRID,
        n_iter=n_iter,
        cv=CV,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        verbose=1,
        n_jobs=-1,
    )
    search.fit(X, y)
    print(f"Best XGB RMSE (CV): ${-search.best_score_:,.0f}")
    print(f"Best params: {search.best_params_}")
    return search.best_estimator_


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(pipelines: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for name, pipe in pipelines.items():
        cv_results = cross_validate(
            pipe, X, y, cv=CV,
            scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"},
        )
        rows.append({
            "model":     name,
            "r2_mean":   cv_results["test_r2"].mean(),
            "r2_std":    cv_results["test_r2"].std(),
            "rmse_mean": -cv_results["test_rmse"].mean(),
            "rmse_std":  cv_results["test_rmse"].std(),
        })
    return pd.DataFrame(rows).sort_values("r2_mean", ascending=False)


# ── Persist ───────────────────────────────────────────────────────────────────
def train_and_save(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, name: str) -> Pipeline:
    pipeline.fit(X, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(pipeline, path)
    print(f"Saved {path}")
    return pipeline


def load_model(name: str) -> Pipeline:
    return joblib.load(MODELS_DIR / f"{name}.pkl")
