"""End-to-end training pipeline (fully live — no CSV files).

Data sources:
  - NHL API:       rosters, per-player stats (current + prior season)
  - PuckPedia:     cap hit, contract length, expiry year/status (scraped, cached 24 h)

Usage:
    python pipeline.py                 # quick run (XGB defaults, use cached data)
    python pipeline.py --tune          # XGB hyperparameter search (~5 min)
    python pipeline.py --refresh       # force re-fetch stats from NHL API
    python pipeline.py --refresh-all   # force re-fetch stats + contracts (slow)
    python pipeline.py --tune --refresh
"""
import argparse
import json
import warnings
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from src.data.load import load_and_merge, save_processed, CAP_CEILING
from src.features.build import build_features, get_feature_matrix, resign_label
from src.models.train import build_pipelines, evaluate, train_and_save, tune_xgb
from src.models.explain import save_shap_artifacts

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"


def main(tune: bool = False, refresh: bool = False, refresh_all: bool = False):
    # ── 1. Load & merge ────────────────────────────────────────────────────────
    # refresh=True  → force re-fetch NHL API stats (not contracts)
    # refresh_all   → force re-fetch both stats AND contracts
    df_raw, ctx = load_and_merge(
        force_refresh=refresh or refresh_all,
        force_refresh_contracts=refresh_all,
    )

    # If refresh_all, also bust the contract cache (handled in load_and_merge
    # via the puckpedia_scraper's force_refresh parameter)

    # ── 2. Feature engineering ─────────────────────────────────────────────────
    df = build_features(df_raw)
    X, y_full = get_feature_matrix(df)

    # ── Train ONLY on players with contract data ───────────────────────────────
    train_mask = df["has_contract_data"].fillna(False).astype(bool)
    X_train    = X[train_mask].copy()
    y_train    = y_full[train_mask].copy()

    assert len(y_train) > 0, "No players with contract data found — check scraper"

    # Normalise target to fraction of cap ceiling
    y_norm = y_train / CAP_CEILING

    print(f"\nFeature matrix: {X.shape[0]} total players × {X.shape[1]} features")
    print(f"Training set:   {len(X_train)} players with contract data")
    print(f"Cap hit range:  ${y_train.min():,.0f}  –  ${y_train.max():,.0f}")

    # ── 3. CV comparison ───────────────────────────────────────────────────────
    pipes = build_pipelines(X_train)
    print("\nRunning 5-fold cross-validation (normalised cap %)...")
    results = evaluate(pipes, X_train, y_norm)
    results["rmse_dollars"] = (results["rmse_mean"] * CAP_CEILING).round(0).astype(int)
    print("\n-- CV Results --")
    print(results[["model", "r2_mean", "r2_std", "rmse_dollars"]].to_string(index=False))

    # ── 4. Train final model ───────────────────────────────────────────────────
    if tune:
        print("\nRunning XGB hyperparameter search (n_iter=30)...")
        final_pipe = tune_xgb(X_train, y_norm, n_iter=30)
        model_name = "xgb_tuned"
        train_and_save(final_pipe, X_train, y_norm, model_name)
    else:
        best_name  = results.iloc[0]["model"]
        print(f"\nTraining final model on full data: {best_name}")
        final_pipe = train_and_save(pipes[best_name], X_train, y_norm, best_name)
        model_name = best_name

    # ── 5. Predict for ALL players (including those without contracts) ──────────
    df["predicted_value"] = final_pipe.predict(X) * CAP_CEILING
    df["value_delta"]     = df.apply(
        lambda r: r["predicted_value"] - r["cap_hit"]
        if r.get("has_contract_data") else None, axis=1
    )
    df["resign_signal"] = df.apply(resign_label, axis=1)

    keep = [
        "name", "team", "pos", "age",
        "cap_hit", "predicted_value", "value_delta",
        "expiry_status", "expiry_year", "years_left", "length_of_contract",
        "gp", "g", "a", "p", "ppg",
        "toi_per_g", "plus_minus", "pim",
        "g60", "p60", "pp_pts", "shots", "shooting_pct",
        "resign_signal", "player_id",
        "has_contract_data", "has_prior_market_data", "is_estimated",
    ]
    out_df = df[[c for c in keep if c in df.columns]].copy()

    for col in ["cap_hit", "predicted_value", "value_delta"]:
        if col in out_df.columns:
            out_df[col] = pd.to_numeric(out_df[col], errors="coerce").round(0)
            out_df[col] = out_df[col].astype("Int64")  # nullable int (handles NaN)

    save_processed(out_df, "predictions.csv")

    # ── 6. Save season context for app status indicator ───────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ctx_out = {**ctx, "cap_ceiling": CAP_CEILING}   # single source of truth for app
    with open(PROCESSED_DIR / "season_context.json", "w") as f:
        json.dump(ctx_out, f, indent=2)
    print(f"Saved season_context.json  ({ctx['mode']})")

    # ── 7. SHAP artifacts ──────────────────────────────────────────────────────
    save_shap_artifacts(final_pipe, X_train, df[train_mask]["name"])

    # ── 8. Console summary ─────────────────────────────────────────────────────
    ranked = out_df[out_df["has_contract_data"].fillna(False)]

    print("\n-- Top 10 Underpaid --")
    print(ranked.nlargest(10, "value_delta")[
        ["name", "team", "cap_hit", "predicted_value", "value_delta"]
    ].to_string(index=False))

    print("\n-- Top 10 Overpaid --")
    print(ranked.nsmallest(10, "value_delta")[
        ["name", "team", "cap_hit", "predicted_value", "value_delta"]
    ].to_string(index=False))

    print("\n-- LA Kings Roster --")
    kings = out_df[out_df["team"] == "LAK"].sort_values(
        "value_delta", ascending=False, na_position="last"
    )
    print(kings[
        ["name", "pos", "age", "cap_hit", "predicted_value", "value_delta", "resign_signal"]
    ].to_string(index=False))

    print(f"\nDone. Run:  py -3 -m streamlit run src/app/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune",        action="store_true", help="XGB hyperparameter search")
    parser.add_argument("--refresh",     action="store_true", help="Force re-fetch NHL API stats")
    parser.add_argument("--refresh-all", action="store_true", dest="refresh_all",
                        help="Force re-fetch stats + contracts (~16 min)")
    args = parser.parse_args()
    main(tune=args.tune, refresh=args.refresh, refresh_all=args.refresh_all)
