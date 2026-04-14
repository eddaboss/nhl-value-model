"""
Step 3 — Comp Finder
Step 4 — Value Delta

For each player, find the n closest comps using a multi-dimensional distance:
  - Performance score proximity (within-cluster rank)
  - Age proximity  (don't match 22-year-olds to 34-year-olds)
  - p60 proximity  (match similar production rates)

Comp pool = ALL players with verified contract data (not UFA-only).
UFA contracts are weighted 1.5× since they are freely-negotiated market signals.

Predicted value = weighted average of comp AAVs.
"""
import numpy as np
import pandas as pd


# ── Comp pool ──────────────────────────────────────────────────────────────────

def build_ufa_comp_pool(df: pd.DataFrame) -> pd.DataFrame:
    """
    All players with verified contract data form the comp pool.

    We include RFAs and players on existing deals — their contracts are still
    market signals (GMs pay what they think a player is worth). UFAs are the
    gold standard (no arbitration, no RFA leverage), so they get a 1.5×
    weight boost inside find_comps().

    Filtering: must have cap_hit AND performance_score.
    """
    # Exclude contracts that don't reflect free-market value:
    #   • Entry-level contracts (ELCs): capped at ~$925k, signed at age 18–21.
    #     These reflect team control, not market price — a 19-year-old star on
    #     $975k is NOT a valid comp for a 30-year-old elite player.
    #     Filter: age ≥ 22 AND cap_hit ≥ $1,100,000.
    #   • Buyout contracts: a bought-out player may show ~$1M even though
    #     their real deal was $8–10M. The $1.1M floor catches most buyouts too.
    MIN_AAV  = 1_100_000
    MIN_AGE  = 22.0

    mask = (
        (df["has_contract_data"].fillna(False).astype(bool)) &
        (df["cap_hit"].notna()) &
        (df["cap_hit"] >= MIN_AAV) &
        (pd.to_numeric(df["age"], errors="coerce").fillna(0) >= MIN_AGE) &
        (df["performance_score"].notna())
    )
    return df[mask].copy()


# ── Comp finding ───────────────────────────────────────────────────────────────

# Distance weights (all inputs normalized to ~0–1 before weighting)
_W_SCORE = 0.30   # performance score similarity
_W_AGE   = 0.25   # age proximity
_W_P60   = 0.45   # points-per-60 proximity (dominant signal — match similar producers)

# Normalisation denominators
_SCORE_RANGE = 200.0   # perf score spans -100 → +100
_AGE_RANGE   = 12.0    # cap age diff at 12 years before clipping to 1
_P60_RANGE   = 2.0     # cap p60 diff at 2.0 (tighter — even 0.5 p60 gap is meaningful)

# UFA contracts get a weight multiplier (freely negotiated, no RFA leverage)
_UFA_WEIGHT_MULT = 1.5


def find_comps(
    player: pd.Series,
    comp_pool: pd.DataFrame,
    n: int = 5,
) -> pd.DataFrame:
    """
    Find n closest comps for a player using a three-dimensional distance:

        dist = W_SCORE * |Δperf_score| / 200
             + W_AGE   * clip(|Δage|  / 12, 0, 1)
             + W_P60   * clip(|Δp60|  / 2.5, 0, 1)

    Same-cluster comps are preferred; if fewer than n exist, cross-cluster
    comps fill in ordered by dist.

    Weight = UFA_mult / (1 + dist)  — closer + UFA → higher weight.
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
    p_age   = float(player.get("age") or 27)
    p_p60   = float(player.get("p60") or 0)

    pool_score = pd.to_numeric(pool["performance_score"], errors="coerce").fillna(p_score)
    pool_age   = pd.to_numeric(pool["age"],               errors="coerce").fillna(p_age)
    pool_p60   = pd.to_numeric(pool["p60"],               errors="coerce").fillna(p_p60)

    score_norm = (pool_score - p_score).abs() / _SCORE_RANGE
    age_norm   = ((pool_age   - p_age  ).abs() / _AGE_RANGE ).clip(upper=1.0)
    p60_norm   = ((pool_p60   - p_p60  ).abs() / _P60_RANGE ).clip(upper=1.0)

    pool["_dist"] = (
        _W_SCORE * score_norm
        + _W_AGE   * age_norm
        + _W_P60   * p60_norm
    )

    # UFA contracts are freely negotiated — boost their weight
    is_ufa = pool["expiry_status"].astype(str).str.upper() == "UFA"
    ufa_mult = np.where(is_ufa, _UFA_WEIGHT_MULT, 1.0)
    pool["_weight"] = ufa_mult / (1.0 + pool["_dist"])

    pool["_same"] = (pool["cluster_id"] == p_cid).astype(int)

    same  = pool[pool["_same"] == 1].nsmallest(n, "_dist")
    other = pool[pool["_same"] == 0].nsmallest(n, "_dist")

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
