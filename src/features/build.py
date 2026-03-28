"""Feature engineering for the NHL value model.

Column sources after load_and_merge():
  From NHL API roster:
    player_id, team, pos, name, age

  From PuckPedia scraper (contract data):
    cap_hit, length_of_contract, year_of_contract,
    expiry_year, expiry_status, years_left

  From NHL API stats — current season (blend-projected):
    gp, g, a, p, ppg, toi_per_g, plus_minus, pim,
    pp_pts, shots, shooting_pct, faceoff_pct, g60, p60

  From NHL API stats — prior season (actual):
    gp_24, ppg_24, toi_per_g_24, plus_minus_24,
    g60_24, p60_24, pp_pts_24, shots_24, shooting_pct_24

  From NHL API landing (draft / bio):
    draft_year, draft_position, birth_date
"""
import numpy as np
import pandas as pd

# ── Feature lists ──────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    # Contract
    "age",
    "draft_position",           # sentinel 999 = undrafted
    "draft_year",               # sentinel 9999 = unknown
    "length_of_contract",
    "year_of_contract",
    # Current-season (blend-projected) on-ice
    "gp", "g", "a", "p", "ppg",
    "toi_per_g",
    "plus_minus", "pim",
    "pp_pts", "shots", "shooting_pct", "faceoff_pct",
    "g60", "p60",
    # Prior-season (NHL API actual)
    "gp_24", "ppg_24",
    "toi_per_g_24", "plus_minus_24",
    "g60_24", "p60_24",
    "pp_pts_24", "shooting_pct_24",
    # Engineered
    "years_left",
    "ppg_2yr_avg",
]

CATEGORICAL_FEATURES = [
    "pos",            # C / L / R / D
    "expiry_status",  # UFA / RFA / UDFA
    "draft_tier",     # top_10 / round1 / after1 / undrafted
]

TARGET = "cap_hit"

ID_COLS = [
    "name", "team", "expiry_year", "birth_date",
    "has_contract_data", "has_prior_market_data",
]


# ── Feature engineering ────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and engineer features. Returns model-ready DataFrame."""
    df = df.copy()

    # ── Draft sentinels ────────────────────────────────────────────────────────
    df["draft_position"] = pd.to_numeric(df["draft_position"], errors="coerce").fillna(999)
    df["draft_year"]     = pd.to_numeric(df["draft_year"],     errors="coerce").fillna(9999)

    def _draft_tier(pos):
        if pos >= 999: return "undrafted"
        if pos <= 10:  return "top_10"
        if pos <= 30:  return "round1"
        return "after1"

    df["draft_tier"] = df["draft_position"].apply(_draft_tier).astype("category")

    # ── years_left: use scraper value if present, otherwise NaN ───────────────
    if "years_left" not in df.columns:
        df["years_left"] = np.nan

    # ── 2-year PPG average ─────────────────────────────────────────────────────
    df["ppg_24"]      = pd.to_numeric(df.get("ppg_24",  np.nan), errors="coerce")
    df["ppg_2yr_avg"] = df[["ppg", "ppg_24"]].mean(axis=1, skipna=True)

    # ── Flags ──────────────────────────────────────────────────────────────────
    if "has_contract_data" not in df.columns:
        df["has_contract_data"] = df["cap_hit"].notna()
    if "has_prior_market_data" not in df.columns:
        df["has_prior_market_data"] = df["ppg_24"].notna()

    # ── Faceoff% — null out for non-centers (wings/D take almost no faceoffs) ──
    # Leaving 0.0 for non-centers pollutes the feature; NaN → imputed to median.
    if "faceoff_pct" in df.columns and "pos" in df.columns:
        df.loc[df["pos"] != "C", "faceoff_pct"] = np.nan

    # ── Ensure numeric types ───────────────────────────────────────────────────
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Impute with column median (pipeline's SimpleImputer handles train/test split)
    num_present = [c for c in NUMERIC_FEATURES if c in df.columns]
    df[num_present] = df[num_present].fillna(df[num_present].median())

    # ── Categorical types ──────────────────────────────────────────────────────
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            # Convert to str first so fillna works regardless of current dtype
            df[col] = df[col].astype(str).replace({"None": "UFA", "nan": "UFA"})
            df[col] = df[col].astype("category")

    return df


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Return (X, y). y = cap_hit for rows with contract data, else None."""
    num_present = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_present = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    X = df[num_present + cat_present]
    y = df[TARGET] if TARGET in df.columns else None
    return X, y
