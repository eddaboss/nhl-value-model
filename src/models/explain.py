"""SHAP-based model explanations for the NHL value model."""
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
from pathlib import Path

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"


def _transform_and_names(pipeline: Pipeline, X: pd.DataFrame):
    """Return (X_transformed_array, feature_names_list)."""
    pre = pipeline.named_steps["pre"]
    X_t = pre.transform(X)
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        names = [f"f{i}" for i in range(X_t.shape[1])]
    # Clean up sklearn's prefixes: "num__age" → "age", "cat__pos_C" → "pos_C"
    names = [n.split("__", 1)[-1] for n in names]
    return X_t, names


def compute_shap(pipeline: Pipeline, X: pd.DataFrame
                 ) -> tuple[np.ndarray, list[str]]:
    """
    Compute SHAP values using TreeExplainer (fast for XGBoost).
    Returns (shap_values array [n_players x n_features], feature_names).
    """
    model = pipeline.named_steps["model"]
    X_t, feature_names = _transform_and_names(pipeline, X)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_t)
    return shap_values, feature_names


def save_shap_artifacts(pipeline: Pipeline, X: pd.DataFrame,
                        player_names: pd.Series) -> None:
    """
    Compute SHAP values and save two files:
      - shap_values.csv   : one row per player, one col per feature
      - shap_summary.csv  : mean |SHAP| per feature, sorted
    """
    print("Computing SHAP values...")
    shap_values, feature_names = compute_shap(pipeline, X)

    # Per-player SHAP values
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.insert(0, "name", player_names.values)
    shap_df.to_csv(PROCESSED_DIR / "shap_values.csv", index=False)

    # Global importance summary
    mean_abs = np.abs(shap_values).mean(axis=0)
    summary = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    summary.to_csv(PROCESSED_DIR / "shap_summary.csv", index=False)
    print(f"  Saved shap_values.csv and shap_summary.csv")
    print("\nTop 10 features by mean |SHAP|:")
    print(summary.head(10).to_string(index=False))


def shap_for_player(pipeline: Pipeline, X: pd.DataFrame,
                    player_idx: int) -> pd.DataFrame:
    """Return a tidy DataFrame of (feature, value, shap_value) for one player."""
    shap_values, feature_names = compute_shap(pipeline, X)
    X_t, _ = _transform_and_names(pipeline, X)
    return pd.DataFrame({
        "feature":    feature_names,
        "value":      X_t[player_idx],
        "shap_value": shap_values[player_idx],
    }).sort_values("shap_value", key=abs, ascending=False)
