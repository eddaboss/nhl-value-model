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
_P60_RANGE   = 1.0     # tight p60 window: 0.5 diff → 50% of max penalty; 1.0+ → clipped
                       # This ensures elite producers (4+ p60) only match other elite
                       # producers, not mid-tier players 0.8+ p60 below them.

# UFA contracts get a weight multiplier (freely negotiated, no RFA leverage)
_UFA_WEIGHT_MULT = 1.5


# Performance score gates: tried in order until enough comps are found.
# A player at score=100 (McDavid) should ONLY comp against other 80+ players,
# not against score=30 players just because they share a cluster label.
# If the pool is thin at the top, we'd rather use 2 true comps than 5 diluted ones.
_SCORE_GATES = [30, 60, 100, 200]

# Once we have this many comps we stop expanding the score gate.
# Quality beats quantity: 2 true peers → more accurate than 5 diluted comps.
_MIN_COMPS_TO_STOP = 2


def find_comps(
    player: pd.Series,
    comp_pool: pd.DataFrame,
    n: int = 5,
) -> pd.DataFrame:
    """
    Find up to n closest comps using a three-dimensional distance:

        dist = W_SCORE * |Δperf_score| / 200
             + W_AGE   * clip(|Δage|  / 12, 0, 1)
             + W_P60   * clip(|Δp60|  / 1.0, 0, 1)

    Score gating: same-cluster comps are selected from progressively wider
    score bands (±30, ±60, ±100, ±200). This prevents a score-100 player
    from being dragged down by score-30 comps in the same cluster.
    Cross-cluster comps fill any remaining slots, also score-gated.

    We return fewer than n comps if no more exist within the widest gate —
    a tight accurate estimate beats a diluted inaccurate one.

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

    # Use cluster_id if available, fall back to cluster_label
    if "cluster_id" in pool.columns:
        p_cid = player.get("cluster_id")
        _cluster_col = "cluster_id"
    else:
        p_cid = player.get("cluster_label")
        _cluster_col = "cluster_label"

    p_score = float(player.get("performance_score") or 0)
    p_age   = float(player.get("age") or 27)
    p_p60   = float(player.get("p60") or 0)

    pool_score = pd.to_numeric(pool["performance_score"], errors="coerce").fillna(p_score)
    pool_age   = pd.to_numeric(pool["age"],               errors="coerce").fillna(p_age)
    pool_p60   = pd.to_numeric(pool["p60"],               errors="coerce").fillna(p_p60)

    pool["_score_diff"] = (pool_score - p_score).abs()

    score_norm = pool["_score_diff"] / _SCORE_RANGE
    age_norm   = ((pool_age - p_age).abs() / _AGE_RANGE).clip(upper=1.0)
    p60_norm   = ((pool_p60 - p_p60).abs() / _P60_RANGE).clip(upper=1.0)

    pool["_dist"] = (
        _W_SCORE * score_norm
        + _W_AGE   * age_norm
        + _W_P60   * p60_norm
    )

    # UFA contracts are freely negotiated — boost their weight
    is_ufa = pool["expiry_status"].astype(str).str.upper() == "UFA"
    ufa_mult = np.where(is_ufa, _UFA_WEIGHT_MULT, 1.0)
    pool["_weight"] = ufa_mult / (1.0 + pool["_dist"])

    pool["_same"] = (pool[_cluster_col] == p_cid).astype(int)

    same  = pool[pool["_same"] == 1]
    other = pool[pool["_same"] == 0]

    # Score-gated same-cluster comps
    # Stop expanding once we hit _MIN_COMPS_TO_STOP — quality beats quantity.
    result_same = pd.DataFrame()
    for gate in _SCORE_GATES:
        candidates = same[same["_score_diff"] <= gate].nsmallest(n, "_dist")
        if len(candidates) >= n:
            return candidates.head(n)            # full set of same-cluster comps → done
        if len(candidates) > len(result_same):
            result_same = candidates
        if len(result_same) >= _MIN_COMPS_TO_STOP:
            break                                # enough quality comps — stop expanding

    # If we found same-cluster comps (even fewer than n), return them directly.
    # DO NOT dilute with cross-cluster players — different cluster = different role.
    # A tight set of true peers beats a padded-out set of dissimilar players.
    if len(result_same) >= _MIN_COMPS_TO_STOP:
        return result_same

    # Only reach here if same-cluster comps are truly scarce (0-1 found).
    # Fall back to cross-cluster, score-gated.
    need = n - len(result_same)
    result_other = pd.DataFrame()
    for gate in _SCORE_GATES:
        candidates = other[other["_score_diff"] <= gate].nsmallest(need, "_dist")
        if len(candidates) >= need:
            result_other = candidates
            break
        if len(candidates) > len(result_other):
            result_other = candidates
        if len(result_other) >= _MIN_COMPS_TO_STOP:
            break

    return pd.concat([result_same, result_other]).head(n)


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
