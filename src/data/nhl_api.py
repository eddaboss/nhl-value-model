"""
NHL public API helpers.

Responsibilities:
  - Detect current season + season blend mode from standings
  - Build player ID → team lookup (roster cache)
  - Fetch per-player season stats from /v1/player/{id}/landing
  - Cache everything to data/raw/ to avoid hammering the API

No hardcoded season length: detected dynamically from standings context.
"""
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path
from datetime import datetime

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"

BASE = "https://api-web.nhle.com/v1"
STANDINGS_URL = f"{BASE}/standings/now"
ROSTER_URL    = f"{BASE}/roster/{{team}}/{{season}}"
PLAYER_URL    = f"{BASE}/player/{{player_id}}/landing"

ROSTER_CACHE       = RAW_DIR / "nhl_roster_cache.json"
STATS_CACHE        = RAW_DIR / "stats_cache.json"
SUPPLEMENTAL_CACHE = RAW_DIR / "supplemental_stats_cache.json"

STATS_REST_BASE = "https://api.nhle.com/stats/rest/en/skater"

_HEADERS = {"User-Agent": "nhl-value-model/2.0"}


# ── Low-level HTTP ─────────────────────────────────────────────────────────────
def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


# ── Name normalisation ────────────────────────────────────────────────────────
def normalize_name(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    ascii_name = re.sub(r"[^a-z0-9 ]", "", ascii_name.lower())
    return re.sub(r"\s+", " ", ascii_name).strip()


def puckpedia_to_normalized(pp_name: str) -> str:
    """Convert PuckPedia 'LastName, FirstName' (with encoding noise) → 'firstname lastname'."""
    clean = pp_name.encode("ascii", "ignore").decode("ascii")
    clean = re.sub(r"[^\w\s,]", "", clean).strip()
    if "," in clean:
        last, first = clean.split(",", 1)
        clean = f"{first.strip()} {last.strip()}"
    return normalize_name(clean)


# ── TOI parsing ───────────────────────────────────────────────────────────────
def parse_toi(avg_toi: str) -> float:
    """Convert 'MM:SS' average TOI string → decimal minutes."""
    try:
        parts = str(avg_toi).split(":")
        return int(parts[0]) + int(parts[1]) / 60
    except Exception:
        return 0.0


def _parse_toi_total(val) -> float:
    """Parse total season TOI — handles MM:SS string or integer/float seconds."""
    if val is None:
        return 0.0
    if isinstance(val, str) and ":" in val:
        return parse_toi(val)   # MM:SS → decimal minutes
    try:
        return float(val) / 60  # seconds → minutes
    except Exception:
        return 0.0


# ── Season context ────────────────────────────────────────────────────────────
def get_season_context() -> dict:
    """
    Derive season blend mode from live standings.

    Returns dict with:
        current_season_id   int   e.g. 20252026
        prev_season_id      int   e.g. 20242025
        n_teams             int   32
        avg_games_per_team  float current average games played
        season_length       int   detected (fallback 82)
        use_blend           bool
        mode                str   'current_only' | 'blend'
        description         str   human-readable label for the app
    """
    standings = _get(STANDINGS_URL)
    teams = standings["standings"]

    season_id = int(teams[0]["seasonId"])
    prev_season_id = season_id - 10001          # e.g. 20252026 → 20242025

    gp_list = [t["gamesPlayed"] for t in teams]
    n_teams  = len(gp_list)
    total_gp = sum(gp_list)
    avg_gp   = total_gp / n_teams

    # Detect season length: try to back-calculate from schedule endpoint.
    # NHL API doesn't expose a clean season-info endpoint, so we derive it:
    # standard season = 82 games.  If any team has already played ≥82 games
    # the season is finished and we know the length exactly.
    max_gp = max(gp_list)
    if max_gp >= 82:
        season_length = max_gp      # use observed max if ≥ 82
    else:
        # best-effort: assume 82 (standard since 2021-22 with 32 teams).
        # If the user tells us the season is different we update here.
        season_length = 82

    # 25% threshold = n_teams × season_length / 4  (in total team-games)
    threshold = n_teams * season_length / 4
    use_blend = (total_gp < threshold)

    # Human-readable season strings  "20252026" → "2025-26"
    def fmt_season(sid: int) -> str:
        s = str(sid)
        return f"{s[:4]}-{s[6:]}"

    cur_str  = fmt_season(season_id)
    prev_str = fmt_season(prev_season_id)

    if use_blend:
        mode = "blend"
        description = (
            f"Blending {prev_str} + first {int(avg_gp)} games of {cur_str} "
            f"({int(total_gp)}/{int(threshold)} team-games threshold)"
        )
    else:
        mode = "current_only"
        description = (
            f"Using {cur_str} season only "
            f"(projected to {season_length}-game pace, "
            f"{avg_gp:.0f} games played)"
        )

    return {
        "current_season_id":  season_id,
        "prev_season_id":     prev_season_id,
        "n_teams":            n_teams,
        "total_team_games":   total_gp,
        "avg_games_per_team": avg_gp,
        "season_length":      season_length,
        "threshold":          threshold,
        "use_blend":          use_blend,
        "mode":               mode,
        "description":        description,
    }


# ── Roster lookup (player_id → team) ─────────────────────────────────────────
def _fetch_team_roster(team: str, season: str, max_retries: int = 3) -> dict | None:
    """
    Fetch one team's roster with exponential backoff retry.
    Returns the raw API response dict, or None if all retries fail.
    """
    url = ROSTER_URL.format(team=team, season=season)
    for attempt in range(max_retries):
        try:
            return _get(url)
        except Exception as e:
            wait = 2 ** attempt   # 1s, 2s, 4s
            if attempt < max_retries - 1:
                print(f"  {team}: attempt {attempt + 1} failed ({e}), retrying in {wait}s…")
                time.sleep(wait)
            else:
                print(f"  WARNING: {team} roster failed after {max_retries} attempts ({e})")
    return None


def build_roster_lookup(force_refresh: bool = False, season: str | None = None) -> dict:
    """
    Fetch every NHL team's active roster for the given season.

    Safety guarantees:
      - Retries each team 3× with exponential backoff before giving up
      - If a team fails all retries, its existing cached data is preserved
      - Never writes a cache with fewer teams than the current cache
      - If the fresh fetch would downgrade team count, keeps existing cache intact

    Returns:  normalized_name → {player_id, team, position, display_name}
    Cached to nhl_roster_cache.json.
    """
    # Always load existing cache first — used as fallback for failed teams
    existing: dict = {}
    if ROSTER_CACHE.exists():
        with open(ROSTER_CACHE, encoding="utf-8") as f:
            existing = json.load(f)

    if not force_refresh and existing:
        return existing

    if season is None:
        ctx = get_season_context()
        season = str(ctx["current_season_id"])

    print("Fetching NHL roster data from API…")
    standings = _get(STANDINGS_URL)
    expected_teams = {t["teamAbbrev"]["default"] for t in standings["standings"]}
    old_teams      = {v["team"] for v in existing.values()}

    # Start with existing cache; update team-by-team so failures keep old data
    lookup: dict = dict(existing)
    fetched_teams: list = []
    kept_teams:    list = []

    for team in sorted(expected_teams):
        roster_data = _fetch_team_roster(team, season)

        if roster_data is None:
            # Keep whatever we already have for this team
            if team in old_teams:
                kept_teams.append(team)
                print(f"  {team}: using cached data")
            else:
                print(f"  {team}: no data (new team, fetch failed)")
            continue

        # Remove stale entries for this team before inserting fresh ones
        lookup = {k: v for k, v in lookup.items() if v.get("team") != team}

        for group in ("forwards", "defensemen", "goalies"):
            for p in roster_data.get(group, []):
                first   = p["firstName"]["default"]
                last    = p["lastName"]["default"]
                display = f"{first} {last}"
                key     = normalize_name(display)
                # Name collision: two active NHLers share the same normalized name
                # (e.g. two "Elias Pettersson"). Preserve both by appending player_id.
                if key in lookup and lookup[key].get("player_id") != p["id"]:
                    existing_pid = lookup[key]["player_id"]
                    # Rename the already-stored entry so both are kept
                    lookup[f"{key}-{existing_pid}"] = lookup.pop(key)
                    key = f"{key}-{p['id']}"
                lookup[key] = {
                    "player_id":    p["id"],
                    "team":         team,
                    "position":     p["positionCode"],
                    "display_name": display,
                }
        fetched_teams.append(team)

    new_teams = {v["team"] for v in lookup.values()}
    missing   = expected_teams - new_teams

    print(f"  Fetched fresh: {len(fetched_teams)} teams  |  "
          f"Kept from cache: {len(kept_teams)} teams  |  "
          f"Still missing: {sorted(missing) or 'none'}")
    print(f"  Total: {len(lookup)} players across {len(new_teams)} teams")

    # Safety: never write a cache that has fewer teams than what we had before
    if len(new_teams) < len(old_teams):
        print(f"  ABORTED cache write: {len(new_teams)} teams < {len(old_teams)} cached "
              f"— keeping existing cache")
        return existing

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(ROSTER_CACHE, "w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2)
    return lookup


# ── Per-player stats from landing page ───────────────────────────────────────
def _extract_season_stats(season_totals: list, season_id: int) -> dict | None:
    """
    Pull and aggregate all NHL regular-season rows for the requested season.
    Handles mid-season trades by summing count stats across stints and
    computing weighted averages for rate stats (TOI, faceoff%).
    """
    rows = [
        r for r in season_totals
        if r.get("season") == season_id
        and r.get("gameTypeId") == 2
        and r.get("leagueAbbrev") == "NHL"
    ]
    if not rows:
        return None

    # Sum count stats across all stints (handles traded players)
    gp         = sum(r.get("gamesPlayed",      0) or 0 for r in rows)
    g          = sum(r.get("goals",            0) or 0 for r in rows)
    a          = sum(r.get("assists",          0) or 0 for r in rows)
    p          = sum(r.get("points",           0) or 0 for r in rows)
    pim        = sum(r.get("pim",              0) or 0 for r in rows)
    pp_pts     = sum(r.get("powerPlayPoints",  0) or 0 for r in rows)
    shots      = sum(r.get("shots",            0) or 0 for r in rows)
    plus_minus = sum(r.get("plusMinus",        0) or 0 for r in rows)

    # TOI: reconstruct total minutes from avgToi × gamesPlayed per stint
    total_toi_min = sum(
        parse_toi(r.get("avgToi", "0:00")) * (r.get("gamesPlayed", 0) or 0)
        for r in rows
    )
    gp = max(gp, 1)
    toi_min = total_toi_min / gp   # weighted avg TOI per game

    # Shooting %: recompute from totals (more accurate than averaging per-stint %)
    shooting_pct = round(g / shots, 4) if shots > 0 else 0.0

    # Faceoff %: weighted average by games played per stint
    fo_num = sum(
        (r.get("faceoffWinningPctg") or 0.0) * (r.get("gamesPlayed", 0) or 0)
        for r in rows
    )
    faceoff_pct = round(fo_num / gp, 4)

    return {
        "gp":           gp,
        "g":            g,
        "a":            a,
        "p":            p,
        "ppg":          round(p / gp, 4),
        "toi_per_g":    round(toi_min, 4),
        "plus_minus":   plus_minus,
        "pim":          pim,
        "pp_pts":       pp_pts,
        "shots":        shots,
        "shooting_pct": shooting_pct,
        "faceoff_pct":  faceoff_pct,
        "g60":  round(g  / (total_toi_min / 60), 4) if total_toi_min > 0 else 0,
        "p60":  round(p  / (total_toi_min / 60), 4) if total_toi_min > 0 else 0,
    }


def _extract_draft_info(landing: dict) -> dict:
    dd = landing.get("draftDetails", {}) or {}
    return {
        "draft_year":     dd.get("year"),
        "draft_position": dd.get("overallPick") or dd.get("pickInRound"),
    }


def fetch_player_stats(player_id: int, season_ids: list[int]) -> dict:
    """
    Fetch /v1/player/{id}/landing and extract stats for the requested season(s).
    Returns dict: {season_id: stats_dict, ..., 'draft': {...}, 'birth_date': str}
    """
    url = PLAYER_URL.format(player_id=player_id)
    try:
        landing = _get(url)
    except Exception as e:
        return {}

    totals = landing.get("seasonTotals", [])
    result: dict = {}
    for sid in season_ids:
        stats = _extract_season_stats(totals, sid)
        if stats:
            result[sid] = stats

    result["draft"]      = _extract_draft_info(landing)
    result["birth_date"] = landing.get("birthDate", "")
    return result


# ── Supplemental stats from NHL stats REST API ────────────────────────────────
def _fetch_stats_rest_report(report: str, season_id: int) -> list[dict]:
    """
    Bulk-fetch one skater report from api.nhle.com/stats/rest/en/skater/.
    Paginates automatically (server caps at 100 rows per page).
    Returns list of player dicts.
    """
    exp = urllib.parse.quote(
        f"gameTypeId=2 and seasonId>={season_id} and seasonId<={season_id}"
    )
    all_rows: list[dict] = []
    start = 0
    limit = 100   # server enforces this cap regardless of what we request
    while True:
        url = (f"{STATS_REST_BASE}/{report}"
               f"?cayenneExp={exp}&limit={limit}&start={start}")
        try:
            data = _get(url)
        except Exception as e:
            print(f"  WARNING: supplemental/{report} s={season_id} "
                  f"start={start} failed: {e}")
            break
        rows = data.get("data", [])
        all_rows.extend(rows)
        total = data.get("total", 0)
        start += len(rows)
        if not rows or start >= total:
            break
    return all_rows


def fetch_supplemental_stats(
    season_ids: list[int],
    force_refresh: bool = False,
) -> dict:
    """
    Bulk-fetch hits, blockedShots, pp_toi, pk_toi from the NHL stats REST API
    for the requested season(s).

    Uses two reports per season (realtime + timeonice) — no per-player loop.

    Returns:
        {season_id (int): {player_id (int): {gp, hits, blocks, pp_toi, pk_toi}}}
    """
    cache: dict = {}
    if SUPPLEMENTAL_CACHE.exists() and not force_refresh:
        with open(SUPPLEMENTAL_CACHE, encoding="utf-8") as f:
            raw = json.load(f)
        # Re-key everything as ints
        cache = {
            int(k): {int(pid): v for pid, v in sv.items()}
            for k, sv in raw.get("seasons", {}).items()
        }
        if all(sid in cache for sid in season_ids):
            total = sum(len(v) for v in cache.values())
            print(f"  Supplemental stats cache hit: {total} records")
            return cache

    result: dict = {}
    for season_id in season_ids:
        print(f"  Fetching supplemental stats (realtime + timeonice) "
              f"for season {season_id}…")

        rt_rows  = _fetch_stats_rest_report("realtime",  season_id)
        toi_rows = _fetch_stats_rest_report("timeonice", season_id)

        rt_by_pid  = {r["playerId"]: r for r in rt_rows}
        toi_by_pid = {r["playerId"]: r for r in toi_rows}

        season_data: dict = {}
        for pid in set(rt_by_pid) | set(toi_by_pid):
            rt  = rt_by_pid.get(pid,  {})
            toi = toi_by_pid.get(pid, {})
            gp  = rt.get("gamesPlayed") or toi.get("gamesPlayed") or 0
            # ppTimeOnIcePerGame / shTimeOnIcePerGame are in seconds → minutes
            season_data[pid] = {
                "gp":     gp,
                "hits":   rt.get("hits",         0) or 0,
                "blocks": rt.get("blockedShots",  0) or 0,
                "pp_toi": round((toi.get("ppTimeOnIcePerGame") or 0) / 60, 4),
                "pk_toi": round((toi.get("shTimeOnIcePerGame") or 0) / 60, 4),
            }
        print(f"    {len(season_data)} skaters loaded for {season_id}")
        result[season_id] = season_data

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUPPLEMENTAL_CACHE, "w", encoding="utf-8") as f:
        json.dump({
            "fetched_at": datetime.utcnow().isoformat(),
            "seasons": {
                str(k): {str(pid): v for pid, v in sv.items()}
                for k, sv in result.items()
            },
        }, f)

    return result


def load_all_player_stats(
    player_ids: list[int],
    season_ids: list[int],
    force_refresh: bool = False,
) -> dict:
    """
    Fetch stats for all player_ids × season_ids.
    Results cached to stats_cache.json.
    Returns:  player_id (int) → {season_id: stats_dict, 'draft': {...}}
    """
    cache: dict = {}

    if STATS_CACHE.exists() and not force_refresh:
        with open(STATS_CACHE, encoding="utf-8") as f:
            raw = json.load(f)
        cache = {int(k): v for k, v in raw.get("players", {}).items()}
        # Check if we have all needed player IDs and seasons
        # Cache keys are ints (converted on load above); season_ids are also ints.
        needed = [pid for pid in player_ids
                  if pid not in cache or
                  any(sid not in cache[pid] for sid in season_ids)]
        if not needed:
            print(f"  Stats cache hit: {len(cache)} players")
            return cache

    print(f"Fetching stats for {len(player_ids)} players "
          f"(seasons: {season_ids})...")
    fetched = 0
    for i, pid in enumerate(player_ids):
        if pid in cache and all(sid in cache[pid] for sid in season_ids):
            continue
        stats = fetch_player_stats(pid, season_ids)
        if stats:
            # Merge into cache (don't overwrite existing seasons)
            existing = cache.get(pid, {})
            existing.update(stats)
            cache[pid] = existing
            fetched += 1
        if i % 50 == 0 and i > 0:
            print(f"  {i}/{len(player_ids)} players fetched...")
        time.sleep(0.05)   # ~0.05s between requests → ~35s for 700 players

    print(f"  Done. {fetched} new fetches, {len(cache)} total cached.")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATS_CACHE, "w", encoding="utf-8") as f:
        json.dump({"fetched_at": datetime.utcnow().isoformat(), "players": cache}, f)
    return cache
