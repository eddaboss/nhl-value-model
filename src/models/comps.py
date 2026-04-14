"""
Step 3 — UFA Comp Finder
Step 4 — Value Delta

For each player, find the 5 closest UFA comps (same cluster first,
then by performance_score distance). Predicted value = weighted average
of comp AAVs, weighted by 1 / (1 + abs(score_diff)).
"""
import numpy as np
import pandas as pd


# ── Comp pool ──────────────────────────────────────────────────────────────────

def build_ufa_comp_pool(df: pd.DataFrame) -> pd.DataFrame:
    """
    UFA players with verified (non-estimated) contract data are the comp pool.
    These are freely-negotiated market contracts — the honest price signal.
    """
    mask = (
        (df["expiry_status"].astype(str).str.upper() == "UFA") &
        (df["has_contract_data"].fillna(False).astype(bool)) &
        (df["cap_hit"].notna()) &
        (df["performance_score"].notna())
    )
    return df[mask].copy()


# ── Comp finding ───────────────────────────────────────────────────────────────

def find_comps(
    player: pd.Series,
    comp_pool: pd.DataFrame,
    n: int = 5,
) -> pd.DataFrame:
    """
    Find n closest comps for a player.

    Priority:
      1. Same cluster (cluster_id match)
      2. Closest performance_score (absolute difference)

    If fewer than n comps exist in same cluster, fill from other clusters
    ordered by score distance.

    Weight = 1 / (1 + abs(score_diff))  — closer score → higher weight.
    """
    # Exclude the player themselves
    pid  = player.get("player_id")
    name = player.get("name")
    pool = comp_pool.copy()
    if pid is not None:
        pool = pool[pool["player_id"] != pid]
    if name is not None:
        pool = pool[pool["name"] != name]

    if pool.empty:
        return pd.DataFrame()

    p_cid   = player.get("cluster_id")
    p_score = float(player.get("performance_score") or 0)

    pool["_diff"]   = (pd.to_numeric(pool["performance_score"], errors="coerce") - p_score).abs()
    pool["_same"]   = (pool["cluster_id"] == p_cid).astype(int)
    pool["_weight"] = 1.0 / (1.0 + pool["_diff"])

    same  = pool[pool["_same"] == 1].nsmallest(n, "_diff")
    other = pool[pool["_same"] == 0].nsmallest(n, "_diff")

    if len(same) >= n:
        return same.head(n)
    return pd.concat([same, other.head(n - len(same))])


def predict_value(player: pd.Series, comp_pool: pd.DataFrame) -> float | None:
    """Weighted average cap hit of 5 closest comps."""
    comps = find_comps(player, comp_pool)
    if comps.empty:
        return None
    hits    = pd.to_numeric(comps["cap_hit"],  errors="coerce")
    weights = comps["_weight"]
    valid   = hits.notna() & weights.notna()
    if valid.sum() == 0:
        return None
    return float((hits[valid] * weights[valid]).sum() / weights[valid].sum())


# ── Public API ─────────────────────────────────────────────────────────────────

def run_comps_model(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply comps model to all players.
    Adds predicted_value and value_delta columns.
    Returns (df_with_predictions, comp_pool).
    """
    comp_pool = build_ufa_comp_pool(df)

    df = df.copy()
    df["predicted_value"] = df.apply(lambda r: predict_value(r, comp_pool), axis=1)
    df["value_delta"] = df.apply(
        lambda r: (
            r["predicted_value"] - r["cap_hit"]
            if (
                r.get("has_contract_data")
                and pd.notna(r.get("predicted_value"))
                and pd.notna(r.get("cap_hit"))
            )
            else None
        ),
        axis=1,
    )
    return df, comp_pool
