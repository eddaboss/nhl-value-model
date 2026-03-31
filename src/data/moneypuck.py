"""
MoneyPuck free skater data loader.

Downloads seasonSummary CSVs from:
  https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/skaters.csv

where {year} is the first year of the season:
  2024 → 2024-25 season
  2025 → 2025-26 season

The MoneyPuck CSV has one row per player per situation.
We use situation == "all" for all-situations metrics.

NOTE on situation values:
  MoneyPuck uses "all" (not "all situations") for the all-situations filter.

NOTE on column overlap:
  hits, blocks, pp_toi, pk_toi are sourced from the NHL stats REST API
  supplemental fetcher (realtime + timeonice reports) and are NOT produced
  here to avoid merge conflicts.

Columns produced (one row per player_id):
  a1        — primary assists (I_F_primaryAssists, projected to full-season pace)
  a2        — secondary assists (I_F_secondaryAssists, projected)
  ixg       — individual expected goals (I_F_xGoals, projected)
  hdxg      — high-danger expected goals (I_F_highDangerxGoals, projected)
  xgf60     — individual xGoals per 60 min  (I_F_xGoals / TOI_hours_60, rate)
  xgf_oi60  — on-ice team xGoals for per 60 (OnIce_F_xGoals / TOI_hours_60, rate)
  xga60     — on-ice xGoals against per 60  (OnIce_A_xGoals / TOI_hours_60, rate)
  ff_pct    — on-ice Fenwick for %          (onIce_fenwickPercentage, rate)
  cf_pct    — on-ice Corsi for %            (onIce_corsiPercentage, rate)
  oz_pct    — offensive zone start %        (I_F_oZoneShiftStarts / I_F_shifts, rate)

Prior-season versions with _24 suffix are also produced for blending.
"""
from __future__ import annotations

import urllib.request
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR  = Path(__file__).parents[2] / "data" / "raw"
BASE_URL = (
    "https://moneypuck.com/moneypuck/playerData/"
    "seasonSummary/{year}/regular/skaters.csv"
)
_HEADERS = {"User-Agent": "nhl-value-model/2.0"}

# Count stats: raw season totals → project to full-season pace
_COUNT_COLS = ["a1", "a2", "ixg", "hdxg"]
# Rate stats: already per-unit, blend directly without pace projection
# fenwick_rel = on-ice Fenwick% minus off-ice Fenwick% (relative impact, team-adjusted)
_RATE_COLS  = ["xgf60", "xgf_oi60", "xga60", "ff_pct", "cf_pct", "oz_pct", "fenwick_rel"]


# ── Download helpers ───────────────────────────────────────────────────────────
def _season_start_year(season_id: int) -> int:
    """20252026 → 2025  (MoneyPuck labels seasons by their start year)"""
    return season_id // 10000


def _fetch_and_cache(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch MoneyPuck skaters CSV for season start year, cache as CSV."""
    cache_path = RAW_DIR / f"moneypuck_{year}.csv"

    if cache_path.exists() and not force_refresh:
        print(f"  MoneyPuck {year}: cache hit ({cache_path.name})")
        return pd.read_csv(cache_path, low_memory=False)

    url = BASE_URL.format(year=year)
    print(f"  MoneyPuck {year}: fetching {url} …")
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        df = pd.read_csv(StringIO(content), low_memory=False)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"  MoneyPuck {year}: {len(df)} rows fetched and cached")
        return df
    except Exception as exc:
        print(f"  MoneyPuck {year}: fetch failed — {exc}")
        return pd.DataFrame()


# ── Per-season extraction ──────────────────────────────────────────────────────
def _extract_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """
    From a full MoneyPuck raw DataFrame (all situations present),
    return one row per player_id with all needed metrics.

    Output columns:
      player_id, gp_mp,
      a1, a2, ixg, hdxg,                  ← raw season counts (will be projected)
      xgf60, xgf_oi60, xga60,             ← per-60 rates
      ff_pct, cf_pct, oz_pct              ← percentage rates
    """
    if raw.empty:
        return pd.DataFrame()

    # MoneyPuck situation == "all" is all-situations (not "all situations")
    all_sit = raw[raw["situation"] == "all"].copy()
    if all_sit.empty:
        return pd.DataFrame()

    all_sit["player_id"] = all_sit["playerId"].astype(int)

    # Total ice time in hours (all situations) for per-60 rate calculations
    ict   = all_sit["icetime"].clip(lower=1)   # seconds
    hours = ict / 3600                          # → hours (for per-60: val/hours)

    # ── Per-60 rates ──────────────────────────────────────────────────────────
    all_sit["xgf60"]    = (all_sit["I_F_xGoals"]      / hours).round(4)
    all_sit["xgf_oi60"] = (all_sit["OnIce_F_xGoals"]  / hours).round(4)
    all_sit["xga60"]    = (all_sit["OnIce_A_xGoals"]  / hours).round(4)

    # ── OZone start %: OZ starts / total shifts ────────────────────────────────
    oz_starts   = all_sit["I_F_oZoneShiftStarts"].fillna(0)
    total_shifts = all_sit["I_F_shifts"].clip(lower=1)
    all_sit["oz_pct"] = (oz_starts / total_shifts).round(4)
    all_sit.loc[all_sit["I_F_shifts"].fillna(0) == 0, "oz_pct"] = np.nan

    # ── Fenwick% relative: on-ice minus off-ice (isolates individual impact) ───
    all_sit["fenwick_rel"] = (
        all_sit["onIce_fenwickPercentage"] - all_sit["offIce_fenwickPercentage"]
    ).round(4)

    # ── xGoals% relative: on-ice minus off-ice xGoals% ────────────────────────
    # Proxy for scoring_chance_against_rel (scf_pct not available in MoneyPuck).
    # Higher = team's xGoals ratio improves when player is on ice vs. off ice.
    # Partially team-quality adjusted via the off-ice subtraction.
    all_sit["sca_rel"] = (
        all_sit["onIce_xGoalsPercentage"] - all_sit["offIce_xGoalsPercentage"]
    ).round(4)

    base = all_sit[[
        "player_id", "games_played",
        "I_F_primaryAssists", "I_F_secondaryAssists",
        "I_F_xGoals", "I_F_highDangerxGoals",
        "xgf60", "xgf_oi60", "xga60",
        "onIce_fenwickPercentage", "onIce_corsiPercentage", "oz_pct",
        "fenwick_rel", "sca_rel",
    ]].rename(columns={
        "games_played":            "gp_mp",
        "I_F_primaryAssists":      "a1",
        "I_F_secondaryAssists":    "a2",
        "I_F_xGoals":              "ixg",
        "I_F_highDangerxGoals":    "hdxg",
        "onIce_fenwickPercentage": "ff_pct",
        "onIce_corsiPercentage":   "cf_pct",
    }).copy()

    return base


# ── Public API ─────────────────────────────────────────────────────────────────
def load_moneypuck_stats(
    cur_season_id:  int,
    prev_season_id: int,
    use_blend:      bool,
    blend_w:        float,
    season_length:  int,
    force_refresh:  bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame (one row per player_id) with MoneyPuck-sourced stats,
    blended/projected using the same logic as the NHL API stats pipeline.

    Parameters
    ----------
    cur_season_id   e.g. 20252026
    prev_season_id  e.g. 20242025
    use_blend       True when current season < 25% complete
    blend_w         avg_games_per_team / season_length  (used only if use_blend)
    season_length   82 (standard)
    force_refresh   re-download even if cache exists

    Current-season output columns:
      player_id,
      a1, a2, ixg, hdxg,                  (counts, projected to full-season pace)
      xgf60, xgf_oi60, xga60,             (per-60 rates)
      ff_pct, cf_pct, oz_pct              (percentage rates)

    Prior-season versions with _24 suffix are also included.
    """
    cur_year  = _season_start_year(cur_season_id)
    prev_year = _season_start_year(prev_season_id)

    cur_raw  = _fetch_and_cache(cur_year,  force_refresh=force_refresh)
    prev_raw = _fetch_and_cache(prev_year, force_refresh=force_refresh)

    cur_df  = _extract_from_raw(cur_raw)
    prev_df = _extract_from_raw(prev_raw)

    if cur_df.empty:
        print("  MoneyPuck: no current-season data available")
        return pd.DataFrame()

    # ── Project count stats to full-season pace ────────────────────────────────
    gp_cur = cur_df["gp_mp"].clip(lower=1)
    scale  = season_length / gp_cur
    for col in _COUNT_COLS:
        cur_df[col] = (cur_df[col] * scale).round(2)

    # ── Blend with prior season if early in the year ───────────────────────────
    if use_blend and not prev_df.empty:
        all_cols = _COUNT_COLS + _RATE_COLS
        merged = cur_df.merge(
            prev_df[["player_id"] + all_cols],
            on="player_id", how="left", suffixes=("", "_prev"),
        )
        for col in all_cols:
            cur_val  = merged[col].fillna(0)
            prev_val = merged.get(f"{col}_prev", pd.Series(0, index=merged.index)).fillna(0)
            merged[col] = (blend_w * cur_val + (1 - blend_w) * prev_val).round(4)
        drop_cols = [f"{col}_prev" for col in all_cols if f"{col}_prev" in merged.columns]
        cur_df = merged.drop(columns=drop_cols)

    # ── Append prior-season columns (_24 suffix) ──────────────────────────────
    all_cols = _COUNT_COLS + _RATE_COLS
    if not prev_df.empty:
        prior_rename = {col: f"{col}_24" for col in all_cols}
        prev_slim = prev_df[["player_id"] + all_cols].rename(columns=prior_rename)
        cur_df = cur_df.merge(prev_slim, on="player_id", how="left")
    else:
        for col in all_cols:
            cur_df[f"{col}_24"] = np.nan

    cur_df = cur_df.drop(columns=["gp_mp"], errors="ignore")

    n = len(cur_df)
    print(f"  MoneyPuck: {n} players with current-season data loaded")
    return cur_df
