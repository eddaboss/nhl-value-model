#!/usr/bin/env python3
"""
scripts/build_historical_dataset.py
====================================
Historical data pipeline for RWPI salary curve training set.

PART 1 — Historical stats
    Fetch MoneyPuck + NHL API + Supplemental for seasons
    20212022, 20222023, 20232024 → player_seasons table.

PART 2 — UFA signing history
    Scrape PuckPedia transaction pages for summers 2022, 2023, 2024
    → ufa_signings table.

PART 3 — RWPI computation
    Join signings with pre-signing season stats, run full RWPI
    pipeline on each season's population, record scores
    → ufa_training_set table.

Output:  data/historical_stats.db
Run:     python scripts/build_historical_dataset.py
"""
from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import sys
import time
import unicodedata
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

# Force UTF-8 output so Unicode chars don't crash on Windows cp1252 terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.nhl_api import (
    fetch_player_stats,
    _fetch_stats_rest_report,
    normalize_name,
)
from src.data.moneypuck import _fetch_and_cache, _extract_from_raw
from src.models.rwpi import (
    assign_roles,
    compute_branch_scores,
    compute_rwpi_score,
    MIN_GP,
)

# ── Constants ──────────────────────────────────────────────────────────────────
HISTORICAL_SEASONS      = [20212022, 20222023, 20232024]
MP_YEAR_FOR_SEASON      = {20212022: 2021, 20222023: 2022, 20232024: 2023}
SIGNING_TO_STATS_SEASON = {2022: 20212022, 2023: 20222023, 2024: 20232024}
CAP_CEILINGS            = {2022: 82_500_000, 2023: 83_500_000, 2024: 88_000_000}
SEASON_LENGTH           = 82

SIGNING_URLS = {
    2022: "https://puckpedia.com/transactions/signings/2022",
    2023: "https://puckpedia.com/transactions/signings/2023",
    2024: "https://puckpedia.com/transactions/signings/2024",
}

DB_PATH       = Path("data/historical_stats.db")
HTML_CACHE_DIR = Path("data/raw/html_cache")

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]
_ua_index = 0


def _next_ua() -> str:
    global _ua_index
    ua = _USER_AGENTS[_ua_index % len(_USER_AGENTS)]
    _ua_index += 1
    return ua


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════════════
def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create all tables if they don't already exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS player_seasons (
            player_id      INTEGER,
            player_name    TEXT,
            season         INTEGER,
            gp             INTEGER,
            g              REAL,
            a              REAL,
            a1             REAL,
            a2             REAL,
            toi_per_g      REAL,
            pp_toi         REAL,
            pk_toi         REAL,
            xgf60          REAL,
            xga60          REAL,
            oz_pct         REAL,
            fenwick_rel    REAL,
            sca_rel        REAL,
            hdxg           REAL,
            shots          REAL,
            hits           REAL,
            blocks         REAL,
            shooting_pct   REAL,
            pp_pts         REAL,
            position       TEXT,
            PRIMARY KEY (player_id, season)
        );

        CREATE TABLE IF NOT EXISTS ufa_signings (
            player_name     TEXT,
            signing_year    INTEGER,
            cap_hit         REAL,
            contract_length INTEGER,
            signing_team    TEXT,
            player_age      REAL,
            PRIMARY KEY (player_name, signing_year)
        );

        CREATE TABLE IF NOT EXISTS ufa_training_set (
            player_name    TEXT,
            signing_year   INTEGER,
            rwpi_score     REAL,
            cap_hit        REAL,
            cap_pct        REAL,
            cap_ceiling    REAL,
            player_age     REAL,
            role           TEXT,
            PRIMARY KEY (player_name, signing_year)
        );
    """)
    conn.commit()
    conn.close()
    print("  DB initialised:", DB_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — HISTORICAL STATS
# ══════════════════════════════════════════════════════════════════════════════

def _project_count(val: float | None, gp: int, season_length: int = 82) -> float | None:
    """Project raw season count to full-season pace."""
    if val is None or gp is None or gp <= 0:
        return val
    if gp >= season_length:
        return round(float(val), 2)
    return round(float(val) * season_length / gp, 2)


def fetch_season_moneypuck(season_id: int) -> pd.DataFrame:
    """
    Download (or load cached) MoneyPuck CSV for season_id and extract metrics.
    Returns one row per player with all RWPI-relevant fields + 82-game projections.
    Includes player_name and position from the raw CSV.
    """
    year = MP_YEAR_FOR_SEASON[season_id]
    print(f"  MoneyPuck {year} ({season_id})…")
    raw = _fetch_and_cache(year, force_refresh=False)
    if raw.empty:
        print(f"    WARNING: no MoneyPuck data for {year}")
        return pd.DataFrame()

    df = _extract_from_raw(raw)   # player_id, gp_mp, a1, a2, ixg, hdxg, rates…
    if df.empty:
        return pd.DataFrame()

    # Add player_name and position from the raw 'all' situation rows
    raw_all = raw[raw["situation"] == "all"][["playerId", "name", "position"]].drop_duplicates("playerId")
    raw_all = raw_all.rename(columns={"playerId": "player_id", "name": "player_name"})
    df = df.merge(raw_all, on="player_id", how="left")

    # Project count stats to 82-game pace
    for col in ["a1", "a2", "hdxg"]:
        gp_arr = df["gp_mp"].clip(lower=1)
        df[col] = (df[col] * (SEASON_LENGTH / gp_arr)).round(2)

    print(f"    {len(df)} players from MoneyPuck {year}")
    return df


def fetch_season_nhl_api(player_ids: list[int], season_id: int) -> dict[int, dict]:
    """
    Fetch NHL API landing stats for player_ids for a specific season.
    Returns {player_id: stats_dict} for players with data for that season.
    Does NOT use or write to the main stats_cache.json.
    """
    print(f"  NHL API stats for {season_id} ({len(player_ids)} players)…")
    results: dict[int, dict] = {}
    for i, pid in enumerate(player_ids):
        data = fetch_player_stats(pid, [season_id])
        season_stats = data.get(season_id) or data.get(str(season_id))
        if season_stats:
            results[pid] = season_stats
        if i % 100 == 0 and i > 0:
            print(f"    {i}/{len(player_ids)} fetched…")
        time.sleep(0.05)
    print(f"    {len(results)}/{len(player_ids)} had stats for {season_id}")
    return results


def fetch_season_supplemental(season_id: int) -> dict[int, dict]:
    """
    Fetch bulk supplemental stats (pp_toi, pk_toi, hits, blocks) for season_id
    directly from the NHL stats REST API — does NOT touch supplemental_stats_cache.json.
    Returns {player_id: {gp, hits, blocks, pp_toi, pk_toi}}.
    """
    print(f"  Supplemental stats for {season_id}…")
    rt_rows  = _fetch_stats_rest_report("realtime",  season_id)
    toi_rows = _fetch_stats_rest_report("timeonice", season_id)

    rt_by_pid  = {r["playerId"]: r for r in rt_rows}
    toi_by_pid = {r["playerId"]: r for r in toi_rows}

    result: dict[int, dict] = {}
    for pid in set(rt_by_pid) | set(toi_by_pid):
        rt  = rt_by_pid.get(pid, {})
        toi = toi_by_pid.get(pid, {})
        gp  = rt.get("gamesPlayed") or toi.get("gamesPlayed") or 0
        result[pid] = {
            "gp":     gp,
            "hits":   rt.get("hits",         0) or 0,
            "blocks": rt.get("blockedShots",  0) or 0,
            "pp_toi": round((toi.get("ppTimeOnIcePerGame") or 0) / 60, 4),
            "pk_toi": round((toi.get("shTimeOnIcePerGame") or 0) / 60, 4),
        }
    print(f"    {len(result)} skaters from supplemental {season_id}")
    return result


def store_season_to_db(
    season_id: int,
    mp_df: pd.DataFrame,
    api_stats: dict[int, dict],
    supp_stats: dict[int, dict],
) -> int:
    """
    Merge MoneyPuck + NHL API + supplemental and upsert into player_seasons.
    Returns number of rows inserted.
    """
    # MP is the source of truth for player list and advanced stats
    if mp_df.empty:
        return 0

    rows: list[dict] = []
    for _, mp_row in mp_df.iterrows():
        pid  = int(mp_row["player_id"])
        gp   = int(mp_row["gp_mp"]) if pd.notna(mp_row.get("gp_mp")) else None

        api  = api_stats.get(pid, {})
        supp = supp_stats.get(pid, {})

        # Use supp gp as primary (bulk endpoint) if available, else api gp
        gp_final = supp.get("gp") or api.get("gp") or gp or 0

        # Projected count stats: hits and blocks need projection; pp/pk toi are per-game
        hits_raw   = supp.get("hits",   0) or 0
        blocks_raw = supp.get("blocks", 0) or 0
        hits_proj   = _project_count(hits_raw,   gp_final)
        blocks_proj = _project_count(blocks_raw, gp_final)

        # g and shots from API are season totals → project
        g_proj     = _project_count(api.get("g"),    gp_final)
        a_proj     = _project_count(api.get("a"),    gp_final)
        shots_proj = _project_count(api.get("shots"), gp_final)
        pp_pts_proj = _project_count(api.get("pp_pts"), gp_final)

        rows.append({
            "player_id":   pid,
            "player_name": mp_row.get("player_name") or None,
            "season":      season_id,
            "gp":          gp_final,
            "g":           g_proj,
            "a":           a_proj,
            "a1":          mp_row.get("a1"),
            "a2":          mp_row.get("a2"),
            "toi_per_g":   api.get("toi_per_g") or mp_row.get("toi_per_g"),
            "pp_toi":      supp.get("pp_toi"),
            "pk_toi":      supp.get("pk_toi"),
            "xgf60":       mp_row.get("xgf60"),
            "xga60":       mp_row.get("xga60"),
            "oz_pct":      mp_row.get("oz_pct"),
            "fenwick_rel": mp_row.get("fenwick_rel"),
            "sca_rel":     mp_row.get("sca_rel"),
            "hdxg":        mp_row.get("hdxg"),
            "shots":       shots_proj,
            "hits":        hits_proj,
            "blocks":      blocks_proj,
            "shooting_pct": api.get("shooting_pct"),
            "pp_pts":      pp_pts_proj,
            "position":    mp_row.get("position") or None,
        })

    conn = _get_conn()
    inserted = 0
    for row in rows:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO player_seasons
                (player_id, player_name, season, gp, g, a, a1, a2,
                 toi_per_g, pp_toi, pk_toi, xgf60, xga60, oz_pct,
                 fenwick_rel, sca_rel, hdxg, shots, hits, blocks, shooting_pct,
                 pp_pts, position)
                VALUES
                (:player_id, :player_name, :season, :gp, :g, :a, :a1, :a2,
                 :toi_per_g, :pp_toi, :pk_toi, :xgf60, :xga60, :oz_pct,
                 :fenwick_rel, :sca_rel, :hdxg, :shots, :hits, :blocks, :shooting_pct,
                 :pp_pts, :position)
            """, row)
            inserted += 1
        except sqlite3.Error as e:
            print(f"    DB error for player {row['player_id']}: {e}")
    conn.commit()
    conn.close()
    return inserted


def run_part1() -> dict[int, int]:
    """Fetch all three historical seasons. Returns {season_id: rows_stored}."""
    print("\n" + "=" * 60)
    print("PART 1 — Historical stats fetch")
    print("=" * 60)

    summary: dict[int, int] = {}

    for season_id in HISTORICAL_SEASONS:
        print(f"\n--- Season {season_id} ---")

        # Check if already loaded
        conn = _get_conn()
        existing = conn.execute(
            "SELECT COUNT(*) FROM player_seasons WHERE season=?", (season_id,)
        ).fetchone()[0]
        conn.close()
        if existing > 0:
            print(f"  Already have {existing} rows in DB — skipping fetch")
            summary[season_id] = existing
            continue

        # 1. MoneyPuck
        mp_df = fetch_season_moneypuck(season_id)
        if mp_df.empty:
            print(f"  SKIPPING {season_id}: no MoneyPuck data")
            summary[season_id] = 0
            continue

        player_ids = mp_df["player_id"].astype(int).tolist()

        # 2. NHL API stats (per-player, hits landing page for historical seasons)
        api_stats = fetch_season_nhl_api(player_ids, season_id)

        # 3. Supplemental stats (bulk REST API)
        supp_stats = fetch_season_supplemental(season_id)

        # 4. Store
        n = store_season_to_db(season_id, mp_df, api_stats, supp_stats)
        print(f"  Stored {n} rows for season {season_id}")
        summary[season_id] = n

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — UFA SIGNINGS SCRAPER
# ══════════════════════════════════════════════════════════════════════════════
#
# PuckPedia's /transactions/signings/{year} paths do not exist (404).
# Individual /signing/{id} pages ARE server-rendered and parseable.
# Strategy:
#   1. Quick boundary scan (every 50th ID, 7000-10200) → find year cutpoints
#   2. Dense scan within each target window (summer months of 2022/2023/2024)
# ─────────────────────────────────────────────────────────────────────────────

SIGNING_BASE_URL    = "https://puckpedia.com/signing/{id}"
# Summer window = June–September.  Filter by month in parsed signing date.
TARGET_SIGNING_MONTHS = {6, 7, 8, 9}
TARGET_SIGNING_YEARS  = {2022, 2023, 2024}

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _is_cloudflare_block(status: int, html: str) -> bool:
    """Detect a Cloudflare challenge or block page."""
    if status in (403, 503):
        return True
    cf_markers = ["cf-browser-verification", "Checking your browser", "cf_chl_opt"]
    return any(m in html for m in cf_markers)


def _parse_dollar(text: str) -> float | None:
    """Parse '$6,500,000' or '6.5M' → float dollars."""
    text = text.strip().replace(",", "")
    m = re.search(r"\$?([\d.]+)\s*[Mm]", text)
    if m:
        return float(m.group(1)) * 1_000_000
    m = re.search(r"\$?([\d]{6,})", text)
    if m:
        return float(m.group(1))
    return None


def _parse_age(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)", text.strip())
    return float(m.group(1)) if m else None


def _parse_signing_page(html: str, signing_id: int) -> dict | None:
    """
    Parse a single /signing/{id} page.

    Title format:  "Name Contract - N Years x $AAV | Puckpedia"
    Body text contains: "MMM DD, YYYY ... signing age N pos X Team ABC
                         Cap Hit $X  Term N years  Expiry Status UFA/RFA"

    Returns a dict or None if the page is 404 / not parseable.
    """
    # 404 check
    if "Page not found" in html or len(html) < 10_000:
        return None

    # ── Extract from title ─────────────────────────────────────────────────
    title_m = re.search(
        r"<title>(.+?) Contract\s*[-–]\s*(\d+)\s*Years?\s*x\s*\$([\d,]+)",
        html, re.I,
    )
    if not title_m:
        return None

    player_name   = title_m.group(1).strip()
    contract_len  = int(title_m.group(2))
    cap_hit_title = float(title_m.group(3).replace(",", ""))

    # ── Extract plain text from main content ──────────────────────────────
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style noise
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)

    # ── Signing date ───────────────────────────────────────────────────────
    date_m = re.search(
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
        r"[\s.]+(\d{1,2})[,\s]+(\d{4})\b",
        text, re.I,
    )
    if not date_m:
        return None
    month_str = date_m.group(1).lower()[:3]
    month     = _MONTH_MAP.get(month_str, 0)
    year      = int(date_m.group(3))

    # ── Age at signing ─────────────────────────────────────────────────────
    age_m = re.search(r"signing\s+age\s+(\d+)", text, re.I)
    age_at_signing = int(age_m.group(1)) if age_m else None

    # ── Position ───────────────────────────────────────────────────────────
    pos_m = re.search(r"\bpos\s+([GCDLRF])\b", text, re.I)
    position = pos_m.group(1).upper() if pos_m else None

    # ── Team ───────────────────────────────────────────────────────────────
    team_m = re.search(r"\bTeam\s+(.+?)(?:\s+GM\b|\s+Agent\b|\s+Cap Hit\b)", text)
    team = team_m.group(1).strip() if team_m else None

    # ── Cap hit (prefer parsed from title; body is a secondary check) ─────
    body_cap_m = re.search(r"Cap\s+Hit\s+\$([\d,]+)", text, re.I)
    cap_hit = cap_hit_title
    if body_cap_m:
        body_val = float(body_cap_m.group(1).replace(",", ""))
        # Title AAV vs body AAV should match; prefer body if they disagree
        if abs(body_val - cap_hit_title) > 1000:
            cap_hit = body_val

    # ── Expiry status at contract end ──────────────────────────────────────
    exp_m = re.search(r"Expiry\s+Status\s+(UFA|RFA|UDFA)", text, re.I)
    expiry_status = exp_m.group(1).upper() if exp_m else None

    return {
        "signing_id":       signing_id,
        "player_name":      player_name,
        "signing_year":     year,
        "signing_month":    month,
        "cap_hit":          cap_hit,
        "contract_length":  contract_len,
        "signing_team":     team,
        "player_age":       float(age_at_signing) if age_at_signing else None,
        "position":         position,
        "expiry_status":    expiry_status,
    }


def _fetch_signing_id_sync(
    signing_id: int,
    min_delay: float = 1.5,
    max_retries: int = 3,
) -> tuple[int, int, str]:
    """
    Fetch /signing/{id} using cloudscraper (handles Cloudflare).
    Returns (signing_id, status_code, html_text).
    """
    import cloudscraper

    url = SIGNING_BASE_URL.format(id=signing_id)

    # Check per-page disk cache first (saves re-fetching on re-runs)
    cache_file = HTML_CACHE_DIR / f"signing_{signing_id}.html"
    if cache_file.exists():
        return signing_id, 200, cache_file.read_text(encoding="utf-8", errors="replace")

    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )
    for attempt in range(max_retries):
        try:
            resp = scraper.get(url, timeout=20)
            if resp.status_code == 429:
                wait = min_delay * (2 ** attempt)
                print(f"    Rate-limited on {signing_id} -- waiting {wait:.1f}s")
                time.sleep(wait)
                continue
            if resp.status_code == 200 and len(resp.text) > 10_000:
                HTML_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(resp.text, encoding="utf-8")
            time.sleep(min_delay)
            return signing_id, resp.status_code, resp.text
        except Exception:
            time.sleep(min_delay * (2 ** attempt))

    return signing_id, 0, ""


def _scan_id_range_threaded(
    id_range: list[int],
    min_delay: float,
    max_workers: int = 4,
) -> list[dict]:
    """
    Threaded scan using cloudscraper. Returns all successfully parsed signings.
    """
    results: list[dict] = []
    batch_size = 40

    def _fetch_and_parse(sid: int) -> dict | None:
        _, status, html = _fetch_signing_id_sync(sid, min_delay=min_delay)
        if status != 200 or not html:
            return None
        return _parse_signing_page(html, sid)

    for i in range(0, len(id_range), batch_size):
        batch = id_range[i : i + batch_size]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = list(pool.map(_fetch_and_parse, batch))
        for parsed in futures:
            if parsed:
                results.append(parsed)
        # Progress heartbeat every 200 IDs
        so_far = i + len(batch)
        if so_far % 200 < batch_size:
            print(f"    ... scanned {so_far}/{len(id_range)} IDs, "
                  f"{len(results)} signings found so far")

    return results


def _upsert_signings_batch(rows: list[dict]) -> int:
    """
    Write qualifying signing rows to ufa_signings.
    Filter: summer months, target years, skaters only (pos != G), age >= 25.
    Commits every 10 rows. Returns count inserted.
    """
    if not rows:
        return 0
    conn = _get_conn()
    inserted = 0
    for row in rows:
        year  = row.get("signing_year", 0)
        month = row.get("signing_month", 0)
        pos   = row.get("position", "")
        age   = row.get("player_age")
        cap   = row.get("cap_hit", 0)

        # Keep only summer signings in target years for skaters aged >= 25
        if year not in TARGET_SIGNING_YEARS:
            continue
        if month not in TARGET_SIGNING_MONTHS:
            continue
        if pos == "G":
            continue
        if age is not None and age < 25:
            continue
        if cap < 700_000:
            continue

        db_row = {
            "player_name":    row["player_name"],
            "signing_year":   year,
            "cap_hit":        cap,
            "contract_length": row.get("contract_length"),
            "signing_team":   row.get("signing_team"),
            "player_age":     age,
        }
        try:
            conn.execute("""
                INSERT OR REPLACE INTO ufa_signings
                (player_name, signing_year, cap_hit, contract_length,
                 signing_team, player_age)
                VALUES (:player_name, :signing_year, :cap_hit,
                        :contract_length, :signing_team, :player_age)
            """, db_row)
            inserted += 1
        except sqlite3.Error as e:
            print(f"    DB error for {row.get('player_name')}: {e}")
        if inserted % 10 == 0 and inserted > 0:
            conn.commit()
    conn.commit()
    conn.close()
    return inserted


def run_part2_sync() -> dict[int, int]:
    """
    Two-pass approach:
      Pass 1 — sparse sample (every 50th ID, 6800-10400) to map year boundaries
      Pass 2 — dense scan of identified target windows
    Uses cloudscraper via ThreadPoolExecutor (bypasses Cloudflare).
    """
    print("\n" + "=" * 60)
    print("PART 2 -- UFA signings scraper (individual /signing/{id} pages)")
    print("=" * 60)

    # Check how many signings already in DB (re-run safety)
    conn = _get_conn()
    existing = conn.execute("SELECT COUNT(*) FROM ufa_signings").fetchone()[0]
    conn.close()
    if existing > 0:
        print(f"  Already have {existing} signing rows in DB -- skipping scrape")
        conn = _get_conn()
        by_year = conn.execute(
            "SELECT signing_year, COUNT(*) FROM ufa_signings GROUP BY signing_year"
        ).fetchall()
        conn.close()
        return {yr: n for yr, n in by_year}

    HTML_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Pass 1: sparse scan to map year boundaries ─────────────────────────
    print("\nPass 1: sparse boundary scan (every 50th ID, 6800-10400)...")
    sparse_ids = list(range(6800, 10401, 50))
    sparse_results = _scan_id_range_threaded(sparse_ids, min_delay=1.0, max_workers=4)

    # Build {signing_id: (year, month)} map from sparse results
    id_date_map: dict[int, tuple[int, int]] = {
        r["signing_id"]: (r["signing_year"], r["signing_month"])
        for r in sparse_results
        if r.get("signing_year") and r.get("signing_month")
    }

    print(f"  Sparse scan found {len(sparse_results)} parseable pages")
    if id_date_map:
        sampled = sorted(id_date_map.items())
        for sid, (yr, mo) in sampled[::max(1, len(sampled)//20)]:
            print(f"    ID {sid} -> {yr}-{mo:02d}")

    # ── Identify target ID ranges from sparse map ─────────────────────────
    # For each target year, find the min/max IDs that fall in summer months,
    # then expand the range by +/-200 for safety.
    target_ranges: list[int] = []
    for target_year in sorted(TARGET_SIGNING_YEARS):
        year_ids = [
            sid for sid, (yr, mo) in id_date_map.items()
            if yr == target_year and mo in TARGET_SIGNING_MONTHS
        ]
        if year_ids:
            lo = max(6700, min(year_ids) - 200)
            hi = min(10500, max(year_ids) + 200)
            print(f"  {target_year} summer range: IDs {lo}-{hi} "
                  f"(from {len(year_ids)} sparse hits)")
        else:
            # Fallback: use rough estimates from probe data
            fallback = {2022: (7300, 7900), 2023: (7800, 8700), 2024: (8600, 9400)}
            lo, hi = fallback.get(target_year, (7000, 9500))
            print(f"  {target_year}: no sparse hits -- using fallback range {lo}-{hi}")
        # Add to dense range, excluding IDs already sampled
        sampled_set = set(sparse_ids)
        target_ranges.extend(
            sid for sid in range(lo, hi + 1) if sid not in sampled_set
        )

    # De-duplicate and sort
    target_ranges = sorted(set(target_ranges))
    print(f"\nPass 2: dense scan of {len(target_ranges)} IDs across target windows...")

    dense_results = _scan_id_range_threaded(target_ranges, min_delay=1.5, max_workers=4)

    # Combine sparse + dense results
    all_results = sparse_results + dense_results
    print(f"\nTotal parseable signing pages: {len(all_results)}")

    n_stored = _upsert_signings_batch(all_results)
    print(f"Stored {n_stored} qualifying signings (summer, age >=25, skaters)")

    # Report by year
    conn = _get_conn()
    by_year = conn.execute(
        "SELECT signing_year, COUNT(*) FROM ufa_signings GROUP BY signing_year"
    ).fetchall()
    conn.close()
    return {yr: n for yr, n in by_year}


def run_part2() -> dict[int, int]:
    return run_part2_sync()


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — RWPI COMPUTATION FOR EACH SIGNING
# ══════════════════════════════════════════════════════════════════════════════

def _load_season_df(season_id: int) -> pd.DataFrame:
    """Load player_seasons rows for one season into a DataFrame."""
    conn = _get_conn()
    df = pd.read_sql(
        "SELECT * FROM player_seasons WHERE season=?", conn, params=(season_id,)
    )
    conn.close()
    return df


def _build_rwpi_df(season_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a DataFrame in the format expected by rwpi.py from a player_seasons slice.
    Maps column names and ensures all needed columns exist.
    """
    df = season_df.copy()

    # rwpi.py expects 'pos' not 'position'
    if "position" in df.columns and "pos" not in df.columns:
        df["pos"] = df["position"].fillna("F")
    elif "pos" not in df.columns:
        df["pos"] = "F"

    # Map position codes: F / D   (MoneyPuck uses C/L/R/D, NHL API uses C/L/R/D)
    # Keep as-is — rwpi.py checks df["pos"] != "D" for forward group
    # Replace None/NaN with "F"
    df["pos"] = df["pos"].fillna("F").astype(str)

    # g60: compute from g and toi_per_g if not present
    if "g60" not in df.columns:
        h82 = (df["toi_per_g"].fillna(0) * SEASON_LENGTH / 60).clip(lower=0.001)
        df["g60"] = (df["g"].fillna(0) / h82).round(4)

    return df


def _name_match_key(name: str) -> str:
    """Normalize player name for fuzzy matching."""
    nfkd = unicodedata.normalize("NFKD", name)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", nfkd.encode("ascii", "ignore").decode().lower())).strip()


def compute_rwpi_for_season(
    season_id: int,
    signing_year: int,
) -> int:
    """
    Run RWPI on the full population for season_id, then match to UFA signings
    for signing_year. Stores results in ufa_training_set.
    Returns number of matched rows stored.
    """
    print(f"\n  Computing RWPI for {season_id} (signing year {signing_year})…")

    season_df = _load_season_df(season_id)
    if season_df.empty:
        print(f"  No player_seasons data for {season_id} — skipping")
        return 0

    # Build RWPI-ready DataFrame
    df = _build_rwpi_df(season_df)

    # Run RWPI pipeline on full population
    df = assign_roles(df)
    df = compute_branch_scores(df)
    df = compute_rwpi_score(df)

    n_scored = df["rwpi_score"].notna().sum()
    print(f"  {n_scored}/{len(df)} players received RWPI scores for {season_id}")

    # Build name → (player_id, rwpi_score, role) lookup
    df["_match_key"] = df["player_name"].fillna("").apply(_name_match_key)
    name_lookup = {
        row["_match_key"]: {
            "player_id": row["player_id"],
            "rwpi_score": row.get("rwpi_score"),
            "role": row.get("rwpi_role", "Unknown"),
        }
        for _, row in df.iterrows()
        if pd.notna(row.get("rwpi_score"))
    }
    # Also index by player_id for direct lookup
    id_lookup = {
        int(row["player_id"]): {
            "rwpi_score": row.get("rwpi_score"),
            "role": row.get("rwpi_role", "Unknown"),
        }
        for _, row in df.iterrows()
        if pd.notna(row.get("rwpi_score"))
    }

    # Load UFA signings for this signing year
    conn = _get_conn()
    signings = pd.read_sql(
        "SELECT * FROM ufa_signings WHERE signing_year=?", conn, params=(signing_year,)
    )
    conn.close()

    if signings.empty:
        print(f"  No signings for year {signing_year} — skipping RWPI match")
        return 0

    cap_ceiling = float(CAP_CEILINGS[signing_year])
    training_rows: list[dict] = []
    unmatched: list[str] = []

    for _, signing in signings.iterrows():
        raw_name = str(signing["player_name"])
        match_key = _name_match_key(raw_name)

        match = name_lookup.get(match_key)

        # Fuzzy fallback: partial token match
        if not match:
            tokens = set(match_key.split())
            for k, v in name_lookup.items():
                k_tokens = set(k.split())
                # Require at least 2 matching tokens (first + last name overlap)
                if len(tokens & k_tokens) >= 2:
                    match = v
                    break

        if not match:
            unmatched.append(raw_name)
            continue

        rwpi = match["rwpi_score"]
        if rwpi is None or np.isnan(float(rwpi)):
            continue

        cap_hit = float(signing["cap_hit"])
        training_rows.append({
            "player_name":  raw_name,
            "signing_year": int(signing_year),
            "rwpi_score":   round(float(rwpi), 4),
            "cap_hit":      cap_hit,
            "cap_pct":      round(cap_hit / cap_ceiling, 6),
            "cap_ceiling":  cap_ceiling,
            "player_age":   signing.get("player_age"),
            "role":         match["role"],
        })

    if unmatched:
        print(f"  {len(unmatched)} signings unmatched (name mismatch): "
              f"{unmatched[:5]}{'…' if len(unmatched) > 5 else ''}")

    # Store results
    conn = _get_conn()
    stored = 0
    for row in training_rows:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO ufa_training_set
                (player_name, signing_year, rwpi_score, cap_hit, cap_pct,
                 cap_ceiling, player_age, role)
                VALUES (:player_name, :signing_year, :rwpi_score, :cap_hit,
                        :cap_pct, :cap_ceiling, :player_age, :role)
            """, row)
            stored += 1
        except sqlite3.Error as e:
            print(f"  DB error: {e}")
        if stored % 10 == 0 and stored > 0:
            conn.commit()
    conn.commit()
    conn.close()

    print(f"  {stored}/{len(training_rows)} training rows stored for {signing_year}")
    return stored


def run_part3() -> dict[int, int]:
    """Run RWPI for each historical season and build training set."""
    print("\n" + "=" * 60)
    print("PART 3 — RWPI computation & training set")
    print("=" * 60)

    summary: dict[int, int] = {}
    for signing_year, season_id in SIGNING_TO_STATS_SEASON.items():
        n = compute_rwpi_for_season(season_id, signing_year)
        summary[signing_year] = n
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_report(
    p1_summary: dict[int, int],
    p2_summary: dict[int, int],
    p3_summary: dict[int, int],
) -> None:
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)

    print("\nPart 1 — Historical seasons fetched:")
    total_p1 = 0
    for sid, n in p1_summary.items():
        print(f"  {sid}: {n} player-season rows")
        total_p1 += n
    print(f"  Total: {total_p1} rows in player_seasons")

    print("\nPart 2 — UFA signings scraped:")
    total_p2 = 0
    for yr, n in p2_summary.items():
        print(f"  {yr}: {n} signings")
        total_p2 += n
    print(f"  Total: {total_p2} UFA signings in DB")

    print("\nPart 3 — Training set rows:")
    total_p3 = 0
    for yr, n in p3_summary.items():
        print(f"  Signing year {yr}: {n} training rows with RWPI")
        total_p3 += n
    print(f"  Total: {total_p3} rows in ufa_training_set")

    # Distribution of cap_pct by RWPI decile
    conn = _get_conn()
    ts = pd.read_sql(
        "SELECT rwpi_score, cap_hit, cap_pct, cap_ceiling, signing_year "
        "FROM ufa_training_set WHERE rwpi_score IS NOT NULL",
        conn,
    )
    conn.close()

    if ts.empty:
        print("\n  No training set rows to show distribution.")
        return

    print(f"\nTraining set summary (n={len(ts)}):")
    print(f"  RWPI range:   {ts['rwpi_score'].min():.1f} – {ts['rwpi_score'].max():.1f}")
    print(f"  cap_hit range: ${ts['cap_hit'].min():,.0f} – ${ts['cap_hit'].max():,.0f}")
    print(f"  cap_pct range: {ts['cap_pct'].min():.4f} – {ts['cap_pct'].max():.4f}")

    print("\ncap_pct by RWPI decile:")
    ts["decile"] = pd.qcut(ts["rwpi_score"], q=10, labels=False, duplicates="drop")
    summary_tbl = ts.groupby("decile", observed=True).agg(
        n=("cap_pct", "count"),
        mean_cap_pct=("cap_pct", "mean"),
        mean_cap_hit=("cap_hit", "mean"),
        rwpi_min=("rwpi_score", "min"),
        rwpi_max=("rwpi_score", "max"),
    ).round(4)
    print(summary_tbl.to_string())

    print("\nBreakdown by signing year:")
    print(ts.groupby("signing_year")[["cap_hit", "rwpi_score"]].describe().round(2))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Building historical dataset for RWPI salary curve…")
    print(f"  Output DB: {DB_PATH.resolve()}")

    init_db()

    p1 = run_part1()
    p2 = run_part2()
    p3 = run_part3()

    print_report(p1, p2, p3)
    print("\nDone.")


if __name__ == "__main__":
    main()
