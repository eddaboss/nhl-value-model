"""End-to-end training pipeline (fully live — no CSV files).

Data sources:
  - NHL API:       rosters, per-player stats (current + prior season)
  - PuckPedia:     cap hit, contract length, expiry year/status (scraped, cached 24 h)

Prediction methodology:
  PRIMARY  — UFA comps model (clustering + performance score + weighted avg AAV)
  BENCHMARK — XGBoost (kept for CV metric comparison)

Usage:
    python pipeline.py                 # quick run (use cached data)
    python pipeline.py --tune          # XGB hyperparameter search (~5 min)
    python pipeline.py --refresh       # force re-fetch stats from NHL API
    python pipeline.py --refresh-all   # force re-fetch stats + contracts (slow)
"""
import argparse
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from src.data.load import load_and_merge, save_processed, CAP_CEILING
from src.features.build import build_features, get_feature_matrix, resign_label
from src.models.train import build_pipelines, evaluate, train_and_save, tune_xgb
from src.models.comps import run_comps_model, find_comps
from src.models.explain import save_shap_artifacts

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"


# ── Test helpers ───────────────────────────────────────────────────────────────

def _find_player(df: pd.DataFrame, *name_fragments: str) -> pd.Series | None:
    """Return first row whose name contains any of the fragments (case-insensitive)."""
    for frag in name_fragments:
        mask = df["name"].str.lower().str.contains(frag.lower(), na=False)
        if mask.any():
            return df[mask].iloc[0]
    return None


def _run_tests(df: pd.DataFrame, comp_pool: pd.DataFrame,
               xgb_rmse: float) -> None:
    """Print all 5 diagnostic tests to console."""
    sep = "=" * 60

    # ── Test 1: Cluster distribution ──────────────────────────────
    print(f"\n{sep}")
    print("TEST 1 — Cluster Distribution")
    print(sep)
    dist = df.groupby("cluster_label").size().sort_values(ascending=False)
    print(dist.to_string())
    for label, count in dist.items():
        if count < 30:
            print(f"  !! WARNING: '{label}' has only {count} players (<30 threshold)")
        if count > 200:
            print(f"  !! WARNING: '{label}' has {count} players (>200 threshold)")

    # ── Test 2: Score distribution ────────────────────────────────
    print(f"\n{sep}")
    print("TEST 2 — Performance Score Distribution")
    print(sep)
    scores = df["performance_score"]
    print(f"  Overall  mean={scores.mean():.1f}  std={scores.std():.1f}  "
          f"min={scores.min():.1f}  max={scores.max():.1f}")
    print()
    for label, grp in df.groupby("cluster_label"):
        s = grp["performance_score"]
        print(f"  {label:<22s}  n={len(grp):3d}  "
              f"mean={s.mean():+6.1f}  std={s.std():5.1f}  "
              f"min={s.min():+7.1f}  max={s.max():+7.1f}")
    if scores.std() < 10:
        print("  !! WARNING: overall std < 10 — scoring may be broken")
    if scores.std() > 45:
        print("  !! WARNING: overall std > 45 — scoring may be broken")

    # ── Test 3: Comp quality for 5 named players ──────────────────
    print(f"\n{sep}")
    print("TEST 3 — Comp Quality (5 Named Players)")
    print(sep)
    test_cases = [
        ("Connor McDavid",       ["mcdavid"]),
        ("Cale Makar",           ["makar"]),
        ("Checking C (Kopitar)", ["bergeron", "kopitar", "barkov"]),
        ("4th-Line F (Reaves)",  ["reaves", "maroon", "jankowski", "gauthier"]),
        ("Shutdown D (Schenn)",  ["schenn", "gudas", "hague", "deslauriers"]),
    ]
    for display_name, fragments in test_cases:
        player = _find_player(df, *fragments)
        if player is None:
            print(f"\n  {display_name}: NOT FOUND in dataset")
            continue
        comps = find_comps(player, comp_pool)
        pred  = player.get("predicted_value")
        cap   = player.get("cap_hit")
        delta = player.get("value_delta")
        print(f"\n  {display_name}  ->  '{player['name']}'")
        print(f"    Cluster:  {player.get('cluster_label')}  |  "
              f"Score: {player.get('performance_score'):+.1f}")
        print(f"    Cap Hit: ${cap:,.0f}  |  Predicted: "
              f"{'$'+f'{pred:,.0f}' if pred is not None else 'N/A'}  |  "
              f"Delta: {'$'+f'{delta:,.0f}' if delta is not None else 'N/A'}")
        if comps.empty:
            print("    Comps: none found")
        else:
            print("    Comps:")
            for _, c in comps.iterrows():
                print(f"      {c.get('name','?'):<25s}  "
                      f"score={c.get('performance_score', float('nan')):+6.1f}  "
                      f"AAV=${c.get('cap_hit', 0):,.0f}  "
                      f"wt={c.get('_weight', float('nan')):.3f}")

    # ── Test 4: UFA comp pool size per cluster ────────────────────
    print(f"\n{sep}")
    print("TEST 4 — UFA Comp Pool Size Per Cluster")
    print(sep)
    print(f"  Total UFA comp pool: {len(comp_pool)} players")
    print()
    pool_dist = comp_pool.groupby("cluster_label").size().sort_values(ascending=False)
    for label, count in pool_dist.items():
        flag = "  !! FLAGGED — may need to relax same-cluster constraint" if count < 8 else ""
        print(f"  {label:<22s}  {count:3d} UFAs{flag}")
    # Warn about any cluster in the full df that has zero UFAs in pool
    all_labels = set(df["cluster_label"].unique())
    pool_labels = set(comp_pool["cluster_label"].unique()) if not comp_pool.empty else set()
    missing = all_labels - pool_labels
    for lbl in missing:
        print(f"  !! WARNING: '{lbl}' has 0 UFAs in comp pool")

    # ── Test 5: RMSE vs XGBoost benchmark ─────────────────────────
    print(f"\n{sep}")
    print("TEST 5 — RMSE vs XGBoost Benchmark")
    print(sep)
    scored = df[
        df["has_contract_data"].fillna(False).astype(bool) &
        df["predicted_value"].notna() &
        df["cap_hit"].notna()
    ]
    if len(scored) > 0:
        comps_rmse = float(np.sqrt(
            ((scored["predicted_value"] - scored["cap_hit"]) ** 2).mean()
        ))
        print(f"  Comps model RMSE:  ${comps_rmse:,.0f}  (n={len(scored)} players)")
        print(f"  XGBoost CV RMSE:   ${xgb_rmse:,.0f}")
        if comps_rmse > 2_500_000:
            print("  !! WARNING: comps RMSE > $2.5M — something may be structurally wrong")
        pct_diff = (comps_rmse - xgb_rmse) / xgb_rmse * 100
        print(f"  Comps is {pct_diff:+.1f}% vs XGBoost CV RMSE "
              f"({'higher' if pct_diff > 0 else 'lower'} — {'expected' if pct_diff > 0 else 'better than benchmark'})")
    else:
        print("  No players with both predicted_value and cap_hit — cannot compute RMSE")

    print(f"\n{sep}")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main(tune: bool = False, refresh: bool = False, refresh_all: bool = False):
    # ── 1. Load & merge ────────────────────────────────────────────────────────
    df_raw, ctx = load_and_merge(
        force_refresh=refresh or refresh_all,
        force_refresh_contracts=refresh_all,
    )

    # ── 2. Feature engineering (incl. clustering + performance scores) ─────────
    df = build_features(df_raw)
    X, y_full = get_feature_matrix(df)

    train_mask = df["has_contract_data"].fillna(False).astype(bool)
    X_train    = X[train_mask].copy()
    y_train    = y_full[train_mask].copy()

    assert len(y_train) > 0, "No players with contract data found — check scraper"

    y_norm = y_train / CAP_CEILING

    print(f"\nFeature matrix: {X.shape[0]} total players × {X.shape[1]} features")
    print(f"Training set:   {len(X_train)} players with contract data")
    print(f"Cap hit range:  ${y_train.min():,.0f}  –  ${y_train.max():,.0f}")
    print(f"Clusters:       {df['cluster_label'].value_counts().to_dict()}")

    # ── 3. XGBoost CV (benchmark) ──────────────────────────────────────────────
    pipes = build_pipelines(X_train)
    print("\nRunning 5-fold CV (XGBoost benchmark, normalised cap %)...")
    xgb_results = evaluate({"xgb": pipes["xgb"]}, X_train, y_norm)
    xgb_results["rmse_dollars"] = (xgb_results["rmse_mean"] * CAP_CEILING).round(0).astype(int)
    xgb_rmse = float(xgb_results.iloc[0]["rmse_dollars"])
    print("\n-- XGBoost CV Results (benchmark) --")
    print(xgb_results[["model", "r2_mean", "r2_std", "rmse_dollars"]].to_string(index=False))

    # ── 4. Train & save XGBoost (for SHAP + reference) ────────────────────────
    if tune:
        print("\nRunning XGB hyperparameter search (n_iter=30)...")
        xgb_pipe = tune_xgb(X_train, y_norm, n_iter=30)
        train_and_save(xgb_pipe, X_train, y_norm, "xgb_tuned")
    else:
        print("\nTraining final XGBoost model on full data...")
        xgb_pipe = train_and_save(pipes["xgb"], X_train, y_norm, "xgb")

    # ── 5. PRIMARY: comps model predictions ───────────────────────────────────
    print("\nRunning comps model (primary predictions)...")
    df, comp_pool = run_comps_model(df)
    print(f"  UFA comp pool: {len(comp_pool)} players")
    scored = df["predicted_value"].notna().sum()
    print(f"  Players with predictions: {scored}/{len(df)}")

    # ── 6. Resign signals ──────────────────────────────────────────────────────
    df["resign_signal"] = df.apply(resign_label, axis=1)

    # ── 7. Build output CSV ────────────────────────────────────────────────────
    keep = [
        "name", "team", "pos", "age",
        "cap_hit", "predicted_value", "value_delta",
        "expiry_status", "expiry_year", "years_left", "length_of_contract",
        "gp", "g", "a", "p", "ppg",
        "toi_per_g", "plus_minus", "pim",
        "g60", "p60", "pp_pts", "shots", "shooting_pct",
        "cluster_id", "cluster_label", "performance_score",
        "faceoff_pct",
        "resign_signal", "player_id",
        "has_contract_data", "has_prior_market_data", "is_estimated",
    ]
    out_df = df[[c for c in keep if c in df.columns]].copy()

    for col in ["cap_hit", "predicted_value", "value_delta"]:
        if col in out_df.columns:
            out_df[col] = pd.to_numeric(out_df[col], errors="coerce").round(0)
            out_df[col] = out_df[col].astype("Int64")

    save_processed(out_df, "predictions.csv")

    # ── 8. Season context ──────────────────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ctx_out = {**ctx, "cap_ceiling": CAP_CEILING}
    with open(PROCESSED_DIR / "season_context.json", "w") as f:
        json.dump(ctx_out, f, indent=2)
    print(f"Saved season_context.json  ({ctx['mode']})")

    # ── 9. SHAP artifacts (XGBoost) ───────────────────────────────────────────
    save_shap_artifacts(xgb_pipe, X_train, df[train_mask]["name"])

    # ── 10. Diagnostic tests ───────────────────────────────────────────────────
    _run_tests(df, comp_pool, xgb_rmse)

    # ── 11. Console summary ────────────────────────────────────────────────────
    ranked = out_df[out_df["has_contract_data"].fillna(False)]

    print("\n-- Top 10 Underpaid (comps model) --")
    print(ranked.nlargest(10, "value_delta")[
        ["name", "team", "cap_hit", "predicted_value", "value_delta"]
    ].to_string(index=False))

    print("\n-- Top 10 Overpaid (comps model) --")
    print(ranked.nsmallest(10, "value_delta")[
        ["name", "team", "cap_hit", "predicted_value", "value_delta"]
    ].to_string(index=False))

    print("\n-- LA Kings Roster --")
    kings = out_df[out_df["team"] == "LAK"].sort_values(
        "value_delta", ascending=False, na_position="last"
    )
    print(kings[
        ["name", "pos", "age", "cap_hit", "predicted_value", "value_delta",
         "cluster_label", "performance_score", "resign_signal"]
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
