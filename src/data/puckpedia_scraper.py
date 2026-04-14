"""
PuckPedia contract data scraper — team-page architecture.

Scrapes one page per team (32 total) instead of one page per player (700+).
Each team page yields cap hit, expiry year, expiry status, and approximate
contract length for every player on that roster.

Concurrency : ThreadPoolExecutor, max 8 workers
Cache       : data/raw/contracts_cache.json, 24-hour TTL
Matching    : difflib fuzzy-match on normalized names (handles "Last, First"
              vs "First Last" difference between PuckPedia and NHL API)

robots.txt compliance
---------------------
PuckPedia's robots.txt restricts AI-training crawlers by name (ClaudeBot,
GPTBot, etc.). For general user-agents it only disallows admin/auth/edit paths.
This scraper uses a standard browser User-Agent, caches for 24 h, and scrapes
32 pages total — far below any reasonable rate limit.
"""

import difflib
import json
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup

try:
    import cloudscraper as _cs
    _HAS_CLOUDSCRAPER = True
except ImportError:
    _HAS_CLOUDSCRAPER = False

RAW_DIR         = Path(__file__).parents[2] / "data" / "raw"
CONTRACTS_CACHE = RAW_DIR / "contracts_cache.json"
CACHE_TTL_HOURS = 24
BASE_URL        = "https://puckpedia.com/team"
MAX_WORKERS     = 8

TEAM_SLUGS: dict[str, str] = {
    "ANA": "anaheim-ducks",
    "UTA": "utah-hc",
    "BOS": "boston-bruins",
    "BUF": "buffalo-sabres",
    "CAR": "carolina-hurricanes",
    "CBJ": "columbus-blue-jackets",
    "CGY": "calgary-flames",
    "CHI": "chicago-blackhawks",
    "COL": "colorado-avalanche",
    "DAL": "dallas-stars",
    "DET": "detroit-red-wings",
    "EDM": "edmonton-oilers",
    "FLA": "florida-panthers",
    "LAK": "los-angeles-kings",
    "MIN": "minnesota-wild",
    "MTL": "montreal-canadiens",
    "NJD": "new-jersey-devils",
    "NSH": "nashville-predators",
    "NYI": "new-york-islanders",
    "NYR": "new-york-rangers",
    "OTT": "ottawa-senators",
    "PHI": "philadelphia-flyers",
    "PIT": "pittsburgh-penguins",
    "SEA": "seattle-kraken",
    "SJS": "san-jose-sharks",
    "STL": "st-louis-blues",
    "TBL": "tampa-bay-lightning",
    "TOR": "toronto-maple-leafs",
    "VAN": "vancouver-canucks",
    "VGK": "vegas-golden-knights",
    "WPG": "winnipeg-jets",
    "WSH": "washington-capitals",
}


# ── Thread-local scraper ───────────────────────────────────────────────────────

_local = threading.local()


def _get_scraper():
    if not hasattr(_local, "scraper"):
        if _HAS_CLOUDSCRAPER:
            _local.scraper = _cs.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "mobile": False}
            )
        else:
            import requests
            s = requests.Session()
            s.headers["User-Agent"] = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
            _local.scraper = s
    return _local.scraper


# ── Name normalisation ─────────────────────────────────────────────────────────

def _normalize(name: str) -> str:
    """
    Lowercase, strip accents, remove non-alpha chars.
    Converts "Last, First" → "first last" (PuckPedia format → standard).
    """
    if "," in name:
        parts = name.split(",", 1)
        name = f"{parts[1].strip()} {parts[0].strip()}"
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z ]", "", ascii_str.lower()).strip()


def _build_name_lookup(roster_lookup: dict) -> dict[str, int]:
    """normalized display_name → player_id (from NHL API roster)."""
    result: dict[str, int] = {}
    for info in roster_lookup.values():
        pid  = info.get("player_id")
        name = info.get("display_name", "")
        if pid and name:
            result[_normalize(name)] = int(pid)
    return result


def _match(scraped_name: str, name_lookup: dict) -> Optional[int]:
    """Return player_id for scraped_name using exact then fuzzy match."""
    norm = _normalize(scraped_name)
    if norm in name_lookup:
        return name_lookup[norm]
    hits = difflib.get_close_matches(norm, name_lookup.keys(), n=1, cutoff=0.85)
    return name_lookup[hits[0]] if hits else None


# ── Page parser ────────────────────────────────────────────────────────────────

def _parse_team_page(html: str, season_end_year: int) -> list[dict]:
    """
    Parse a PuckPedia team page and return a list of player contract dicts.

    HTML structure (confirmed via inspection):
      <tr class="group">
        <td> <a href="/player/slug" translate="no">Last, First</a> … </td>
        <td data-js="capcol" data-extract_ch="7,000,000" …>  ← current-year cap
        <td data-js="capcol" data-extract_ch="7,000,000" …>  ← next-year cap
        …  (up to 7 season columns total)
        <td data-js="capcol" data-extract_ch="0">
          <span class="pp-ufa …">UFA</span>   ← first zero column carries status
        </td>
      </tr>

    Each non-zero column = one season of the contract, starting from season_end_year.
    Column index 0 → season_end_year, index 1 → season_end_year + 1, etc.
    """
    soup  = BeautifulSoup(html, "lxml")
    seen  = set()
    results: list[dict] = []

    for a_tag in soup.find_all("a", href=re.compile(r"^/player/[^/]+$"), translate="no"):
        name = a_tag.get_text(strip=True)
        if not name or name in seen:
            continue

        row = a_tag.find_parent("tr", class_="group")
        if row is None:
            continue

        cap_cells = row.find_all("td", attrs={"data-js": "capcol"})
        if not cap_cells:
            continue

        # ── Cap hit: first cell's data-extract_ch ─────────────────────────────
        first_cell = cap_cells[0]
        raw_ch = first_cell.get("data-extract_ch", "0").replace(",", "")
        try:
            cap_hit = int(raw_ch)
        except ValueError:
            continue
        if cap_hit <= 0:
            continue  # no contract data for this player

        # ── Contract columns: count consecutive non-zero cells ─────────────────
        nonzero_count = sum(
            1 for td in cap_cells
            if td.get("data-extract_ch", "0").replace(",", "").strip() not in ("0", "")
        )

        # expiry_year: the last season with cap data (end-year format)
        # e.g. 2 non-zero cols with season_end_year=2026 → 2026, 2027 → expires after 2026-27 → 2027
        expiry_year = season_end_year + nonzero_count - 1

        # length_of_contract: years remaining (best we can extract from team pages)
        length_of_contract = nonzero_count

        # ── Expiry status: pp-ufa / pp-rfa span (in the first zero cell) ────────
        ufa_span = row.find("span", class_="pp-ufa")
        rfa_span = row.find("span", class_="pp-rfa")
        if ufa_span:
            expiry_status = "UFA"
        elif rfa_span:
            expiry_status = "RFA"
        else:
            # Fallback: check text of the zero cells
            row_text = row.get_text(" ", strip=True)
            if re.search(r"\bRFA\b", row_text):
                expiry_status = "RFA"
            else:
                expiry_status = "UFA"  # default to UFA if unknown

        seen.add(name)
        results.append({
            "name":               name,
            "cap_hit":            cap_hit,
            "length_of_contract": length_of_contract,
            "expiry_year":        expiry_year,
            "expiry_status":      expiry_status,
        })

    return results


# ── Team fetcher ───────────────────────────────────────────────────────────────

def _fetch_team(
    team_abbr: str,
    slug: str,
    season_end_year: int,
) -> tuple[str, list[dict], bool]:
    """Fetch and parse one team page. Returns (team_abbr, players, had_error)."""
    url     = f"{BASE_URL}/{slug}"
    scraper = _get_scraper()

    for attempt in range(2):
        try:
            resp = scraper.get(url, timeout=25)
            if resp.status_code == 200:
                return team_abbr, _parse_team_page(resp.text, season_end_year), False
            if attempt == 0:
                time.sleep(2.0)
        except Exception:
            if attempt == 0:
                time.sleep(2.0)

    return team_abbr, [], True


# ── Cache ──────────────────────────────────────────────────────────────────────

def _cache_valid() -> bool:
    if not CONTRACTS_CACHE.exists():
        return False
    try:
        data = json.loads(CONTRACTS_CACHE.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(data.get("fetched_at", "2000-01-01"))
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600
        return age_h < CACHE_TTL_HOURS
    except Exception:
        return False


def _load_cache() -> dict:
    data = json.loads(CONTRACTS_CACHE.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data.get("contracts", {}).items()}


def _save_cache(contracts: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "contracts":  {str(k): v for k, v in contracts.items()},
    }
    CONTRACTS_CACHE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ── Public API ─────────────────────────────────────────────────────────────────

def scrape_contracts(
    roster_lookup: dict,
    season_end_year: int,
    force_refresh: bool = False,
    birth_years: dict | None = None,    # kept for signature compat; unused
) -> dict:
    """
    Scrape PuckPedia contract data by fetching one page per team (32 pages).

    Parameters
    ----------
    roster_lookup   : name → {player_id, team, position, display_name}
                      (from nhl_api.build_roster_lookup)
    season_end_year : e.g. 2026 for the 2025-26 season
    force_refresh   : ignore the 24-hour cache
    birth_years     : unused (kept for backwards-compatibility)

    Returns
    -------
    dict : player_id (int) → contract_dict | None
           contract_dict has keys: cap_hit, length_of_contract,
                                   expiry_year, expiry_status
           None means the player was not found on any team page.
    """
    if not force_refresh and _cache_valid():
        cached = _load_cache()
        print(f"  Contracts cache hit: {len(cached)} players "
              f"({sum(1 for v in cached.values() if v)} with data)")
        return cached

    if not _HAS_CLOUDSCRAPER:
        try:
            import requests as _r  # noqa: F401 — requests is a fallback
        except ImportError:
            print("  WARNING: neither cloudscraper nor requests installed. "
                  "Run: pip install cloudscraper")
            return {}

    name_lookup = _build_name_lookup(roster_lookup)

    print(f"  Scraping 32 PuckPedia team pages ({MAX_WORKERS} workers)…")
    t0 = time.perf_counter()

    all_scraped: list[dict] = []   # {name, cap_hit, …, team_abbr}
    failed_teams: list[str] = []
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_team, abbr, slug, season_end_year): abbr
            for abbr, slug in TEAM_SLUGS.items()
        }
        for fut in as_completed(futures):
            abbr, players, had_error = fut.result()
            with lock:
                if had_error:
                    failed_teams.append(abbr)
                else:
                    for p in players:
                        all_scraped.append({**p, "_team": abbr})

    elapsed = time.perf_counter() - t0
    n_teams_ok = len(TEAM_SLUGS) - len(failed_teams)
    print(f"  {len(all_scraped)} players scraped from {n_teams_ok}/32 teams "
          f"in {elapsed:.1f}s")
    if failed_teams:
        print(f"  !! Failed teams: {', '.join(failed_teams)}")

    # ── Match names → player_ids ───────────────────────────────────────────────
    contracts: dict[int, Optional[dict]] = {}
    no_match:  list[str] = []

    for p in all_scraped:
        pid = _match(p["name"], name_lookup)
        if pid is None:
            no_match.append(f"{p['name']} ({p['_team']})")
            continue
        # Traded players appear on two rosters; keep the entry with higher cap hit
        # (the current team's prorated value is lower than the full-season AAV
        # stored by the other team's page for an extension scenario)
        if pid in contracts and contracts[pid] is not None:
            existing_ch = contracts[pid].get("cap_hit", 0)
            if p["cap_hit"] <= existing_ch:
                continue  # keep the existing (higher) entry

        contracts[pid] = {
            "cap_hit":            p["cap_hit"],
            "length_of_contract": p["length_of_contract"],
            "expiry_year":        p["expiry_year"],
            "expiry_status":      p["expiry_status"],
        }

    # Ensure every roster player has an entry (None = not found on any team page)
    for info in roster_lookup.values():
        pid = info.get("player_id")
        if pid and int(pid) not in contracts:
            contracts[int(pid)] = None

    matched = sum(1 for v in contracts.values() if v is not None)
    print(f"  Matched {matched}/{len(all_scraped)} scraped players to player_ids")

    if no_match:
        print(f"  {len(no_match)} unmatched players (logged below):")
        for nm in sorted(no_match)[:50]:   # cap at 50 to keep output readable
            print(f"    - {nm}")
        if len(no_match) > 50:
            print(f"    … and {len(no_match) - 50} more")

    _save_cache(contracts)
    return contracts
