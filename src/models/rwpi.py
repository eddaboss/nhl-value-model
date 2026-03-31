"""
Role-Weighted Performance Index (RWPI)
======================================
Player value model that scores each skater 0-100 based on on-ice
performance only — independent of salary.

Pipeline
--------
1. Role classification  — k-means on F and D independently (3 each → 6 roles)
                          using deployment features only (PP TOI, PK TOI,
                          OZ start%, total TOI/G)
2. Offensive branch     — 8 per-60 stats, each percentile-ranked, weighted sum,
                          final percentile rank → 0-100
3. Defensive branch     — 6 possession/suppression stats, same treatment → 0-100
4. RWPI score           — role-weighted blend of both branches → 0-100
5. UFA salary curve     — polynomial (degree 2) fit on age_at_signing ≥ 27 UFAs
6. Apply curve          — true open-market value in dollars for every player
7. value_gap            — true_value minus actual cap_hit
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# sklearn imports are deferred to avoid triggering OpenBLAS at module load time.
# assign_roles() imports KMeans/StandardScaler on first call.
# fit_salary_curve() imports the rest on first call.
def _import_sklearn_cluster():
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    return KMeans, StandardScaler

def _import_sklearn_regression():
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.pipeline import Pipeline as SKPipeline
    from sklearn.preprocessing import PolynomialFeatures
    return LinearRegression, cross_val_score, KFold, SKPipeline, PolynomialFeatures

# XGBoost check is also deferred
_HAS_XGB: bool | None = None   # None = not yet checked

def _check_xgb():
    global _HAS_XGB
    if _HAS_XGB is None:
        try:
            from xgboost import XGBRegressor  # noqa: F401
            _HAS_XGB = True
        except ImportError:
            _HAS_XGB = False
    return _HAS_XGB

# Minimum games played for a player to receive an RWPI score
MIN_GP = 20


# ── Role definitions ───────────────────────────────────────────────────────────
ROLE_NAMES: dict[int, str] = {
    0: "Depth / Bottom-6 F",
    1: "Defensive / PK Forward",
    2: "Offensive Driver F",
    3: "Depth / 7th Defenseman",
    4: "Shutdown Defenseman",
    5: "Offensive Defenseman",
}

# (off_weight, def_weight)  — must each sum to 1.0
ROLE_WEIGHTS: dict[int, tuple[float, float]] = {
    0: (0.60, 0.40),
    1: (0.30, 0.70),
    2: (0.70, 0.30),
    3: (0.30, 0.70),
    4: (0.25, 0.75),
    5: (0.65, 0.35),
}

# Offensive branch weights (per the spec)
_OFF_WEIGHTS = {
    "g60":          1.00,
    "a1_60":        0.85,
    "a2_60":        0.60,
    "xgf60":        0.75,
    "pp_pts_60":    0.70,
    "hdxg60":       0.50,
    "shots60":      0.25,
    "shooting_pct": 0.30,
}

# Defensive branch weights
_DEF_WEIGHTS = {
    "pk_toi_pct":  0.80,   # PK TOI as % of total TOI — coach trust signal, individual
    "pk_toi":      0.70,   # PK TOI absolute (per game) — role volume
    "dz_pct":      0.60,   # DZ start % (1 - oz_pct) — deployment signal, coach-decided
    "sca_rel":     0.50,   # xGoals% on-off relative — partial team-quality cancellation
    "blocks60":    0.35,   # blocked shots per 60
    "hits60":      0.25,   # hits per 60 — reduced: low individual signal
}

_CLUSTER_FEATURES = ["pp_toi", "pk_toi", "oz_pct", "toi_per_g"]

# Full-season length used to denominate projected count stats
_SEASON_LENGTH = 82


# ── Helpers ────────────────────────────────────────────────────────────────────
def _pct_rank(s: pd.Series) -> pd.Series:
    """Percentile rank 0-100, NaN preserved."""
    return s.rank(pct=True, na_option="keep") * 100


def _safe_div(num: pd.Series, denom: pd.Series, fill: float = 0.0) -> pd.Series:
    """Element-wise division; fill with `fill` where denom is 0 or NaN."""
    return num.div(denom.replace(0, np.nan)).fillna(fill)


def _toi_hours_82(df: pd.DataFrame) -> pd.Series:
    """Total projected TOI in hours based on 82-game pace."""
    return df["toi_per_g"] * _SEASON_LENGTH / 60


# ── Step 1: Role classification ────────────────────────────────────────────────
def _remap_labels(
    X: np.ndarray,
    labels: np.ndarray,
    features: list[str],
) -> np.ndarray:
    """
    Within a F or D group, remap the 3 k-means labels to consistent roles:
      2 = highest PP TOI mean (offensive)
      1 = highest PK TOI mean among remaining (defensive/PK)
      0 = remainder (depth)
    """
    df_tmp = pd.DataFrame(X, columns=features)
    df_tmp["label"] = labels
    means = df_tmp.groupby("label")[["pp_toi", "pk_toi"]].mean()

    off = int(means["pp_toi"].idxmax())
    remaining = means.drop(index=off)
    def_ = int(remaining["pk_toi"].idxmax())
    depth = [c for c in means.index if c not in (off, def_)][0]

    remap = {off: 2, def_: 1, depth: 0}
    return np.array([remap[lbl] for lbl in labels])


def assign_roles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster forwards and defensemen independently (3 clusters each).
    Adds columns: rwpi_role_id (int 0-5), rwpi_role (str).
    Players with < MIN_GP games played, or missing any cluster feature,
    get role_id = -1 / 'Unknown' and will not receive RWPI scores.
    """
    df = df.copy()
    df["rwpi_role_id"] = -1
    df["rwpi_role"] = "Unknown"

    gp_ok   = df["gp"].fillna(0) >= MIN_GP
    feat_ok = df[_CLUSTER_FEATURES].notna().all(axis=1)

    for is_defense, offset in [(False, 0), (True, 3)]:
        grp = (df["pos"] == "D") if is_defense else (df["pos"] != "D")
        idx = df.index[grp & feat_ok & gp_ok]
        if len(idx) < 3:
            continue

        X_raw = df.loc[idx, _CLUSTER_FEATURES].values
        KMeans, StandardScaler = _import_sklearn_cluster()
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_raw)

        km = KMeans(n_clusters=3, random_state=42, n_init=20)
        raw_labels = km.fit_predict(X_s)
        labels = _remap_labels(X_raw, raw_labels, _CLUSTER_FEATURES)

        df.loc[idx, "rwpi_role_id"] = labels + offset

    df["rwpi_role"] = df["rwpi_role_id"].map(ROLE_NAMES).fillna("Unknown")
    return df


# ── Steps 2-3: Branch scores ───────────────────────────────────────────────────
def _build_off_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute offensive branch raw metrics.  Returns a DataFrame with the 8
    named columns in _OFF_WEIGHTS, aligned to df's index.
    All projected count stats use 82-game pace as denominator.
    """
    h82 = _toi_hours_82(df)

    feats = pd.DataFrame(index=df.index)

    # g60 — from NHL API (actual per-game rate, already per-60)
    feats["g60"] = df["g60"].fillna(0)

    # a1_60, a2_60 — from MoneyPuck (projected to 82-game pace)
    feats["a1_60"] = _safe_div(df.get("a1", pd.Series(0, index=df.index)), h82)
    feats["a2_60"] = _safe_div(df.get("a2", pd.Series(0, index=df.index)), h82)

    # xgf60 — from MoneyPuck (already per-60 rate)
    feats["xgf60"] = df.get("xgf60", pd.Series(np.nan, index=df.index)).fillna(0)

    # pp_pts_60 — PP points per 60 min of PP ice time
    # pp_pts projected to 82 games; pp_toi is per-game rate
    pp_toi_total = df.get("pp_toi", pd.Series(0, index=df.index)) * _SEASON_LENGTH
    pp_pts_proj  = df.get("pp_pts", pd.Series(0, index=df.index)).fillna(0)
    feats["pp_pts_60"] = _safe_div(pp_pts_proj * 60, pp_toi_total)

    # hdxg60 — high-danger xG per 60 (MoneyPuck projected count)
    feats["hdxg60"] = _safe_div(
        df.get("hdxg", pd.Series(0, index=df.index)).fillna(0), h82
    )

    # shots60 — shots per 60 (NHL API projected count)
    feats["shots60"] = _safe_div(df.get("shots", pd.Series(0, index=df.index)).fillna(0), h82)

    # shooting_pct — direct rate (0-1 scale)
    feats["shooting_pct"] = df.get("shooting_pct", pd.Series(np.nan, index=df.index)).fillna(
        df.get("shooting_pct", pd.Series()).median()
    )

    return feats


def _build_def_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute defensive branch raw metrics.  Returns a DataFrame with the 6
    named columns in _DEF_WEIGHTS, aligned to df's index.

    Features
    --------
    pk_toi_pct  PK TOI / total TOI (dimensionless 0-1)
    pk_toi      PK TOI per game (minutes) — absolute volume
    dz_pct      1 - oz_pct — defensive zone start fraction
    sca_rel     on-ice xGoals% minus off-ice xGoals% — team-quality-adjusted
    blocks60    blocked shots per 60 projected minutes
    hits60      hits per 60 projected minutes
    """
    h82 = _toi_hours_82(df)

    feats = pd.DataFrame(index=df.index)

    # pk_toi_pct — PK ice time as fraction of total TOI (both in minutes/game)
    pk_toi  = df.get("pk_toi",    pd.Series(0.0, index=df.index)).fillna(0)
    toi_pg  = df.get("toi_per_g", pd.Series(np.nan, index=df.index))
    feats["pk_toi_pct"] = _safe_div(pk_toi, toi_pg.replace(0, np.nan))

    # pk_toi — absolute PK TOI per game
    feats["pk_toi"] = pk_toi

    # dz_pct — defensive zone start % (inverted oz_pct)
    oz = df.get("oz_pct", pd.Series(np.nan, index=df.index)).fillna(
        df.get("oz_pct", pd.Series()).median()
    )
    feats["dz_pct"] = 1.0 - oz

    # sca_rel — on-ice minus off-ice xGoals%; higher = better relative defensive impact
    # proxy for scoring_chance_against_rel (scf_pct unavailable in MoneyPuck)
    feats["sca_rel"] = df.get("sca_rel", pd.Series(np.nan, index=df.index)).fillna(
        df.get("sca_rel", pd.Series()).median()
    )

    # blocks60, hits60 — projected season counts → per 60
    feats["blocks60"] = _safe_div(
        df.get("blocks", pd.Series(0, index=df.index)).fillna(0), h82
    )
    feats["hits60"] = _safe_div(
        df.get("hits", pd.Series(0, index=df.index)).fillna(0), h82
    )

    return feats


def _weighted_branch_score(
    feats: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """
    For each feature column: percentile-rank → 0-100.
    Take weighted sum of ranked components.
    Final percentile-rank → 0-100 branch score.
    """
    total_w = sum(weights.values())
    ranked_sum = pd.Series(0.0, index=feats.index)
    for col, w in weights.items():
        if col in feats.columns:
            ranked_sum += _pct_rank(feats[col]).fillna(50) * w
    # Divide by total_weight so raw range is 0-100, then final pct-rank
    raw = ranked_sum / total_w
    return _pct_rank(raw).fillna(50)


def compute_branch_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rwpi_off_score and rwpi_def_score (0-100) to df.
    Players with rwpi_role_id == -1 (excluded due to GP or missing features)
    receive NaN for both branch scores.
    """
    df = df.copy()
    eligible = df["rwpi_role_id"] != -1

    # Build features only for eligible players; score against the full eligible pool
    off_feats_all = _build_off_features(df)
    def_feats_all = _build_def_features(df)

    off_score = _weighted_branch_score(off_feats_all.loc[eligible], _OFF_WEIGHTS)
    def_score  = _weighted_branch_score(def_feats_all.loc[eligible], _DEF_WEIGHTS)

    df["rwpi_off_score"] = np.nan
    df["rwpi_def_score"]  = np.nan
    df.loc[eligible, "rwpi_off_score"] = off_score
    df.loc[eligible, "rwpi_def_score"]  = def_score
    return df


# ── Step 4: RWPI composite score ───────────────────────────────────────────────
def compute_rwpi_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine branch scores using role weights → rwpi_score 0-100.
    Players with rwpi_role_id == -1 receive NaN rwpi_score.
    """
    df = df.copy()
    eligible = df["rwpi_role_id"] != -1

    off_w = df["rwpi_role_id"].map({k: v[0] for k, v in ROLE_WEIGHTS.items()})
    def_w = df["rwpi_role_id"].map({k: v[1] for k, v in ROLE_WEIGHTS.items()})

    raw = pd.Series(np.nan, index=df.index)
    raw.loc[eligible] = (
        off_w.loc[eligible] * df.loc[eligible, "rwpi_off_score"] +
        def_w.loc[eligible] * df.loc[eligible, "rwpi_def_score"]
    )
    # Percentile rank only within eligible players
    df["rwpi_score"] = np.nan
    df.loc[eligible, "rwpi_score"] = _pct_rank(raw.loc[eligible])
    return df


# ── Step 5: UFA salary curve ───────────────────────────────────────────────────
def fit_salary_curve(
    df: pd.DataFrame,
    cap_ceiling: float,
    poly_r2_threshold: float = 0.25,
) -> tuple[object, dict]:
    """
    Fit a two-feature salary curve (rwpi_score + age) on the UFA training set.

    Filter: expiry_status='UFA', years_left >= 1, is_estimated=False,
            cap_hit not null, age_at_signing >= 27, rwpi_score not null.

    Strategy
    --------
    1. Try polynomial degree-2 surface (sklearn Pipeline).
    2. If CV R² mean < poly_r2_threshold AND xgboost is installed,
       fall back to XGBoost with early stopping.

    Returns
    -------
    model   : fitted model (sklearn Pipeline or XGBRegressor)
    info    : dict with training stats for validation output
    """
    # ── Build training set ─────────────────────────────────────────────────────
    df_train = df[
        (df["expiry_status"] == "UFA") &
        (df["years_left"].fillna(0) >= 1) &
        (~df["is_estimated"].fillna(False)) &
        (df["cap_hit"].notna()) &
        (df["length_of_contract"].notna()) &
        (df["age"].notna()) &
        (df["rwpi_score"].notna())
    ].copy()

    df_train["age_at_signing"] = (
        df_train["age"] - (df_train["length_of_contract"] - df_train["years_left"])
    )
    df_train = df_train[df_train["age_at_signing"] >= 27].copy()

    # Two-feature matrix: [rwpi_score, age_at_signing]
    X = df_train[["rwpi_score", "age_at_signing"]].values
    y = (df_train["cap_hit"] / cap_ceiling).values   # cap percentage

    LinearRegression, cross_val_score, KFold, SKPipeline, PolynomialFeatures = \
        _import_sklearn_regression()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # ── Attempt 1: polynomial degree-2 surface ─────────────────────────────────
    poly_model = SKPipeline([
        ("poly",  PolynomialFeatures(degree=2, include_bias=True)),
        ("ridge", LinearRegression()),
    ])
    cv_r2_poly = cross_val_score(poly_model, X, y, cv=kf, scoring="r2")
    poly_mean  = float(cv_r2_poly.mean())

    if poly_mean >= poly_r2_threshold or not _check_xgb():
        # Use polynomial
        poly_model.fit(X, y)
        y_pred   = poly_model.predict(X)
        train_r2 = float(np.corrcoef(y, y_pred)[0, 1] ** 2)
        coef     = poly_model.named_steps["ridge"].coef_
        intercept = float(poly_model.named_steps["ridge"].intercept_)

        info = {
            "model_type":    "polynomial_degree2",
            "n_train":       len(df_train),
            "cv_r2_mean":    poly_mean,
            "cv_r2_std":     float(cv_r2_poly.std()),
            "cv_r2_scores":  cv_r2_poly.tolist(),
            "train_r2":      train_r2,
            "coef":          coef.tolist(),
            "intercept":     intercept,
            "cap_ceiling":   cap_ceiling,
            "train_df":      df_train,
        }
        return poly_model, info

    # ── Attempt 2: XGBoost fallback (strongly regularised for ~200 samples) ───
    from xgboost import XGBRegressor as _XGBRegressor
    xgb_model = _XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        reg_alpha=1.0,
        random_state=42,
        verbosity=0,
    )
    cv_r2_xgb = cross_val_score(xgb_model, X, y, cv=kf, scoring="r2")
    xgb_model.fit(X, y)
    y_pred_xgb = xgb_model.predict(X)
    train_r2   = float(np.corrcoef(y, y_pred_xgb)[0, 1] ** 2)

    info = {
        "model_type":       "xgboost",
        "n_train":          len(df_train),
        "poly_cv_r2_mean":  poly_mean,
        "poly_cv_r2_std":   float(cv_r2_poly.std()),
        "cv_r2_mean":       float(cv_r2_xgb.mean()),
        "cv_r2_std":        float(cv_r2_xgb.std()),
        "cv_r2_scores":     cv_r2_xgb.tolist(),
        "train_r2":         train_r2,
        "cap_ceiling":       cap_ceiling,
        "train_df":         df_train,
    }
    return xgb_model, info


# ── Steps 6-7: Apply curve and compute gaps ────────────────────────────────────
def apply_salary_curve(
    df: pd.DataFrame,
    model: object,
    cap_ceiling: float,
) -> pd.DataFrame:
    """
    Apply the UFA salary curve to every eligible player (rwpi_score not null).
    Uses two features: [rwpi_score, age_at_signing].

    age_at_signing:
      - Players with a contract: age - (length_of_contract - years_left)
      - Players without a contract (or missing data): current age
        (interpreted as "would sign today")

    Adds: true_value (dollars), value_gap (true_value - cap_hit).
    Players with null rwpi_score get null true_value and null value_gap.
    """
    df = df.copy()
    eligible = df["rwpi_score"].notna()

    # Compute age_at_signing for prediction
    has_contract_data = (
        df["length_of_contract"].notna() & df["years_left"].notna()
    )
    df["_age_at_signing_pred"] = df["age"]   # default: current age
    df.loc[has_contract_data, "_age_at_signing_pred"] = (
        df.loc[has_contract_data, "age"] - (
            df.loc[has_contract_data, "length_of_contract"] -
            df.loc[has_contract_data, "years_left"]
        )
    )

    X_all = df.loc[eligible, ["rwpi_score", "_age_at_signing_pred"]].values
    cap_pct = model.predict(X_all)
    cap_pct = np.clip(cap_pct, 0.005, 0.20)   # floor ~$500k, ceiling ~20% of cap

    df["true_value"] = np.nan
    df.loc[eligible, "true_value"] = (cap_pct * cap_ceiling).round(0)

    df["value_gap"] = np.nan
    has_contract = eligible & df["cap_hit"].notna()
    df.loc[has_contract, "value_gap"] = (
        df.loc[has_contract, "true_value"] - df.loc[has_contract, "cap_hit"]
    ).round(0)

    df = df.drop(columns=["_age_at_signing_pred"])
    return df


# ── Main entry point ───────────────────────────────────────────────────────────
def build_rwpi_predictions(
    df: pd.DataFrame,
    cap_ceiling: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Full RWPI pipeline.  Takes the merged DataFrame from load_and_merge().

    Returns (df_with_rwpi_columns, curve_info_dict).

    New columns added
    -----------------
    rwpi_role_id   : int  0-5  (-1 = unknown)
    rwpi_role      : str  human-readable role label
    rwpi_off_score : float 0-100 offensive branch percentile
    rwpi_def_score : float 0-100 defensive branch percentile
    rwpi_score     : float 0-100 composite RWPI
    true_value     : float dollars (UFA open-market estimate)
    value_gap      : float true_value - cap_hit  (None if no contract)
    """
    df = assign_roles(df)
    df = compute_branch_scores(df)
    df = compute_rwpi_score(df)
    salary_model, curve_info = fit_salary_curve(df, cap_ceiling)
    df = apply_salary_curve(df, salary_model, cap_ceiling)
    return df, {"salary_model": salary_model, "curve_info": curve_info}
