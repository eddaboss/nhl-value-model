"""
Load pipeline: NHL API (rosters + stats) + contracts.db (source of truth).

No CSV files.  Stats fetched live from NHL API, cached on disk.
Contract data loaded from data/contracts.db (SQLite) — pre-built by
scripts/build_contracts_db.py and maintained incrementally.

load_and_merge() → (df, season_context)

Columns produced:
  From NHL API roster:   player_id, team, pos, display_name
  From contracts.db:     cap_hit, length_of_contract, year_of_contract,
                          expiry_year, expiry_status, years_left, is_estimated
  From NHL API stats     gp, g, a, p, ppg, toi_per_g, plus_minus, pim,
  (current + prior):     pp_pts, shots, shooting_pct, faceoff_pct, g60, p60
                          (same with _24 suffix for prior season)
  From NHL API landing:  draft_year, draft_position, birth_date, age
  Flags:                 has_contract_data, has_prior_market_data
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd
import numpy as np

from src.data.nhl_api import (
    build_roster_lookup,
    get_season_context,
    load_all_player_stats,
    fetch_supplemental_stats,
)
from src.data.contracts_db import (
    get_all_contracts,
    player_ids_in_db,
    DB_PATH,
)
from src.data.moneypuck import load_moneypuck_stats

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

# 2025-26 NHL salary cap ceiling
CAP_CEILING = 95_500_000


# ── Stats layer ────────────────────────────────────────────────────────────────
_STAT_KEYS = [
    "gp", "g", "a", "p", "ppg",
    "toi_per_g", "plus_minus", "pim",
    "pp_pts", "shots", "shooting_pct", "faceoff_pct",
    "g60", "p60",
]
_COUNT_KEYS = {"g", "a", "p", "pim", "pp_pts", "shots"}


def _project(stats: dict, season_length: int) -> dict:
    gp = max(stats.get("gp") or 1, 1)
    if gp >= season_length:
        return stats
    scale = season_length / gp
    out = dict(stats)
    for k in _COUNT_KEYS:
        if k in out and out[k] is not None:
            out[k] = round(out[k] * scale, 2)
    out["plus_minus"] = round((out.get("plus_minus") or 0) * scale, 1)
    return out


def _blend(cur: dict, prior: dict, blend_w: float, season_length: int) -> dict:
    cur_proj = _project(cur, season_length)
    out = {}
    for k in _STAT_KEYS:
        c = cur_proj.get(k) or 0
        p = prior.get(k) or 0
        out[k] = round(blend_w * c + (1 - blend_w) * p, 4)
    return out


def _build_stats_df(
    player_ids: list[int],
    ctx: dict,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch per-player stats with blend/projection, return as DataFrame."""
    cur_sid       = ctx["current_season_id"]
    prev_sid      = ctx["prev_season_id"]
    season_length = ctx["season_length"]
    avg_gp        = ctx["avg_games_per_team"]
    use_blend     = ctx["use_blend"]

    raw = load_all_player_stats(player_ids, [cur_sid, prev_sid], force_refresh=force_refresh)

    rows = []
    for pid in player_ids:
        player_raw = raw.get(pid, {})

        cur   = player_raw.get(cur_sid)  or player_raw.get(str(cur_sid))  or {}
        prior = player_raw.get(prev_sid) or player_raw.get(str(prev_sid)) or {}

        if use_blend and cur and prior:
            blend_w   = avg_gp / season_length
            stats_cur = _blend(cur, prior, blend_w, season_length)
        elif cur:
            stats_cur = {k: _project(cur, season_length).get(k) for k in _STAT_KEYS}
        else:
            stats_cur = {k: None for k in _STAT_KEYS}

        _prior_keys = ["gp", "ppg", "toi_per_g", "plus_minus",
                       "g60", "p60", "pp_pts", "shots", "shooting_pct"]
        stats_24 = {f"{k}_24": (prior.get(k) if prior else None) for k in _prior_keys}

        draft = (player_raw.get("draft") or {})

        rows.append({
            "player_id":      pid,
            **stats_cur,
            **stats_24,
            "draft_year":     draft.get("draft_year"),
            "draft_position": draft.get("draft_position"),
            "birth_date":     player_raw.get("birth_date", ""),
        })

    return pd.DataFrame(rows)


# ── Supplemental stats (hits, blocks, pp_toi, pk_toi) ─────────────────────────
_SUPP_COUNT_KEYS = {"hits", "blocks"}   # season totals → project by gp
_SUPP_RATE_KEYS  = {"pp_toi", "pk_toi"} # per-game rates → no projection


def _build_supplemental_df(
    player_ids: list[int],
    ctx: dict,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch hits, blocks, pp_toi, pk_toi from the NHL stats REST API (bulk).
    Applies the same season-projection and blending logic as _build_stats_df.
    Returns a DataFrame keyed by player_id with those four columns + _24 variants.
    """
    cur_sid       = ctx["current_season_id"]
    prev_sid      = ctx["prev_season_id"]
    season_length = ctx["season_length"]
    avg_gp        = ctx["avg_games_per_team"]
    use_blend     = ctx["use_blend"]

    supp = fetch_supplemental_stats([cur_sid, prev_sid], force_refresh=force_refresh)
    cur_data  = supp.get(cur_sid,  {})
    prior_data = supp.get(prev_sid, {})

    all_keys = list(_SUPP_COUNT_KEYS | _SUPP_RATE_KEYS)

    rows = []
    for pid in player_ids:
        cur   = cur_data.get(pid,   {})
        prior = prior_data.get(pid, {})
        row   = {"player_id": pid}

        if cur:
            gp    = max(cur.get("gp") or 1, 1)
            scale = season_length / gp if gp < season_length else 1.0

            if use_blend and prior:
                blend_w = avg_gp / season_length
                for k in _SUPP_COUNT_KEYS:
                    cur_proj = (cur.get(k) or 0) * scale
                    row[k] = round(blend_w * cur_proj + (1 - blend_w) * (prior.get(k) or 0), 2)
                for k in _SUPP_RATE_KEYS:
                    row[k] = round(blend_w * (cur.get(k) or 0) + (1 - blend_w) * (prior.get(k) or 0), 4)
            else:
                for k in _SUPP_COUNT_KEYS:
                    row[k] = round((cur.get(k) or 0) * scale, 2)
                for k in _SUPP_RATE_KEYS:
                    row[k] = cur.get(k) or 0
        else:
            for k in all_keys:
                row[k] = None

        # Prior-season _24 suffixes
        for k in all_keys:
            row[f"{k}_24"] = prior.get(k) if prior else None

        rows.append(row)

    return pd.DataFrame(rows)


# ── Age from birth_date ────────────────────────────────────────────────────────
def _age_from_birth(birth_str: str) -> float | None:
    """Current age in decimal years as of today.
    The NHL API /v1/player landing endpoint does not expose currentAge;
    only birthDate is available, so we calculate from that using today's date."""
    if not birth_str:
        return None
    try:
        from datetime import date
        bd  = date.fromisoformat(birth_str)
        ref = date.today()
        return round((ref - bd).days / 365.25, 4)
    except Exception:
        return None


# ── Main entry point ───────────────────────────────────────────────────────────
def load_and_merge(
    force_refresh: bool = False,
    force_refresh_contracts: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Build the full modelling dataset from live NHL API + PuckPedia scraper.

    Steps:
    1. Detect season context (blend / current-only, season IDs)
    2. Build NHL API roster lookup → active players with player_id + team
    3. Fetch NHL API stats for all players (current + prior season)
    4. Scrape PuckPedia for contract data (cap_hit, expiry, etc.)
    5. Join everything on player_id
    6. Compute age, flag missing data

    Returns (df, season_context).
    """
    # ── 1. Season context ──────────────────────────────────────────────────────
    ctx = get_season_context()
    season_end_year = ctx["current_season_id"] % 10000   # 20252026 → 2026
    print(f"\nSeason context: {ctx['description']}")

    # ── 2. Roster lookup ──────────────────────────────────────────────────────
    roster = build_roster_lookup(force_refresh=force_refresh)  # normalized_name → {player_id, team, ...}
    print(f"Roster lookup: {len(roster)} players")

    # Build unique player list from roster (deduplicate by player_id)
    seen: dict[int, dict] = {}
    for _name, info in roster.items():
        pid = info.get("player_id")
        if pid:
            seen.setdefault(int(pid), info)

    player_ids = list(seen.keys())

    # Roster DataFrame: player_id, team, pos, display_name
    roster_rows = []
    for pid, info in seen.items():
        roster_rows.append({
            "player_id":    pid,
            "team":         info.get("team"),
            "pos":          info.get("position"),
            "display_name": info.get("display_name", ""),
        })
    df_roster = pd.DataFrame(roster_rows)

    # ── 3. NHL API stats ───────────────────────────────────────────────────────
    df_stats = _build_stats_df(player_ids, ctx, force_refresh=force_refresh)
    df_stats["player_id"] = df_stats["player_id"].astype(int)

    # ── 4. Contract database (source of truth) ────────────────────────────────
    # contracts.db is pre-built by scripts/build_contracts_db.py
    # If the DB is empty/missing, warn and fall back to no contracts.
    if not DB_PATH.exists() or len(player_ids_in_db()) == 0:
        print("  WARNING: contracts.db not found or empty. "
              "Run:  py -3 scripts/build_contracts_db.py")
        contracts_all = {}
    else:
        contracts_all = get_all_contracts()
        print(f"  contracts.db: {len(contracts_all)} players loaded")

    contract_rows = []
    for pid in player_ids:
        row    = {"player_id": int(pid)}
        cdata  = contracts_all.get(int(pid))
        if cdata and cdata.get("cap_hit") is not None:
            row["cap_hit"]            = cdata.get("cap_hit")
            row["length_of_contract"] = cdata.get("contract_length") or cdata.get("length_of_contract")
            row["year_of_contract"]   = cdata.get("year_of_contract")
            row["expiry_year"]        = cdata.get("expiry_year")
            row["expiry_status"]      = cdata.get("expiry_status")
            row["years_left"]         = cdata.get("years_left")
            row["is_estimated"]       = bool(cdata.get("is_estimated", 0))
        elif cdata and cdata.get("expiry_status") == "UFA":
            # Confirmed UFA with no active contract (correct, not a scrape failure)
            row.update({
                "cap_hit": None, "length_of_contract": None,
                "year_of_contract": None, "expiry_year": None,
                "expiry_status": "UFA", "years_left": 0,
                "is_estimated": False,
            })
        else:
            row.update({
                "cap_hit": None, "length_of_contract": None,
                "year_of_contract": None, "expiry_year": None,
                "expiry_status": None, "years_left": None,
                "is_estimated": False,
            })
        contract_rows.append(row)

    df_contracts = pd.DataFrame(contract_rows)
    df_contracts["player_id"] = df_contracts["player_id"].astype(int)

    # ── 5. MoneyPuck advanced stats ────────────────────────────────────────────
    blend_w = (ctx["avg_games_per_team"] / ctx["season_length"]
               if ctx["use_blend"] else 1.0)
    df_mp = load_moneypuck_stats(
        cur_season_id=ctx["current_season_id"],
        prev_season_id=ctx["prev_season_id"],
        use_blend=ctx["use_blend"],
        blend_w=blend_w,
        season_length=ctx["season_length"],
        force_refresh=force_refresh,
    )

    # ── 6. Supplemental stats (hits, blocks, pp_toi, pk_toi) ─────────────────
    df_supp = _build_supplemental_df(player_ids, ctx, force_refresh=force_refresh)
    df_supp["player_id"] = df_supp["player_id"].astype(int)

    # ── 7. Join on player_id ──────────────────────────────────────────────────
    df = df_roster.merge(df_stats, on="player_id", how="left")
    df = df.merge(df_contracts, on="player_id", how="left")
    df = df.merge(df_supp, on="player_id", how="left")
    if not df_mp.empty:
        df = df.merge(df_mp, on="player_id", how="left")

    # ── 8. Post-join cleanup ──────────────────────────────────────────────────
    # Clean display name: "FirstName LastName" is already in the right format
    df["name"] = df["display_name"]
    df = df.drop(columns=["display_name"])

    # Remove goalies (position G)
    df = df[df["pos"] != "G"].copy()

    # Disambiguate players who share a display_name (e.g. two "Elias Pettersson")
    # Append " (POS)" suffix so the app can tell them apart
    dup_mask = df.duplicated("name", keep=False)
    if dup_mask.any():
        df.loc[dup_mask, "name"] = (
            df.loc[dup_mask, "name"] + " (" + df.loc[dup_mask, "pos"] + ")"
        )

    # Age from birth_date (current age as of today — NHL API has no currentAge field)
    df["age"] = df["birth_date"].apply(_age_from_birth)

    # Flags
    df["has_contract_data"]     = df["cap_hit"].notna()
    df["has_prior_market_data"] = df["ppg_24"].notna()
    if "is_estimated" not in df.columns:
        df["is_estimated"] = False
    df["is_estimated"] = df["is_estimated"].fillna(False).astype(bool)

    # Numeric cap_hit
    df["cap_hit"] = pd.to_numeric(df["cap_hit"], errors="coerce")

    n_total    = len(df)
    n_contract = df["has_contract_data"].sum()
    n_est      = df["is_estimated"].sum()
    n_prior    = df["has_prior_market_data"].sum()
    n_ufa      = ((~df["has_contract_data"]) & (df["expiry_status"] == "UFA")).sum()
    n_mp       = int(df["xgf60"].notna().sum()) if "xgf60" in df.columns else 0

    print(f"Loaded {n_total} skaters — "
          f"{n_contract} with contracts "
          f"({n_est} estimated*), "
          f"{n_ufa} confirmed UFA, "
          f"{n_prior} with prior-season stats")
    print(f"MoneyPuck coverage: {n_mp}/{n_total} players "
          f"({n_mp/n_total*100:.1f}%)")
    print(f"Duplicate rows: {df.duplicated('name').sum()}")

    return df, ctx


# ── Output helper ──────────────────────────────────────────────────────────────
def save_processed(df: pd.DataFrame, filename: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / filename, index=False)
    print(f"Saved {PROCESSED_DIR / filename}")
