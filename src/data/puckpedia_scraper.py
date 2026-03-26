"""
PuckPedia contract data scraper.

Fetches current cap hit, contract length, expiry year, and expiry status
for every active NHL player by scraping individual player pages on puckpedia.com.

robots.txt compliance
─────────────────────
PuckPedia's robots.txt restricts *AI training* crawlers (ClaudeBot, GPTBot, etc.)
by name.  For general user-agents it only disallows admin, auth, and certain
edit pages — not public player profile pages.  This scraper uses a standard
browser User-Agent, adds delays between requests, and caches for 24 hours
to minimise server load.

Architecture
────────────
- Cloudflare protection is handled by the `cloudscraper` library.
- Contract data is parsed from the server-side-rendered summary paragraph
  that PuckPedia embeds on every player page.
- Results are cached to data/raw/contracts_cache.json (24-hour TTL).
- Up to 10 concurrent requests via ThreadPoolExecutor (thread-local scrapers).
- If a player's page cannot be fetched or parsed, they are flagged as
  'Contract data unavailable' rather than being dropped.
"""
import json
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import cloudscraper as _cs
    _HAS_CLOUDSCRAPER = True
except ImportError:
    _HAS_CLOUDSCRAPER = False

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
CONTRACTS_CACHE = RAW_DIR / "contracts_cache.json"
CACHE_TTL_HOURS = 24
BASE_URL = "https://puckpedia.com/player"
MAX_WORKERS   = 10    # concurrent requests
REQUEST_DELAY = 0.1   # seconds per request (polite, not blocking)


# ── Thread-local scraper ───────────────────────────────────────────────────────
_local = threading.local()


def _get_scraper():
    """Return a thread-local cloudscraper instance (required for thread safety)."""
    if not hasattr(_local, 'scraper'):
        _local.scraper = _cs.create_scraper(
            browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
        )
    return _local.scraper


# ── Helpers ────────────────────────────────────────────────────────────────────
def _display_to_slug(display_name: str) -> str:
    """
    Convert 'FirstName LastName' → 'firstname-lastname' PuckPedia slug.
    Strips accents, punctuation, and title suffixes (Jr., Sr., etc.).
    """
    name = re.sub(r'\b(Jr\.?|Sr\.?|III?|IV)\b', '', display_name, flags=re.IGNORECASE)
    nfkd = unicodedata.normalize('NFKD', name)
    ascii_name = nfkd.encode('ascii', 'ignore').decode('ascii')
    slug = re.sub(r'[^a-z0-9]+', '-', ascii_name.lower()).strip('-')
    return slug


def _parse_contract(html: str, season_end_year: int) -> Optional[dict]:
    """
    Parse contract data from a PuckPedia player page.

    Looks for the standard summary sentence:
      "[Name] is signed to a {N} year, ${TCV} contract with a cap hit of
       ${CAP} per season, playing for the {Team}. His contract expires at
       the end of the {YYYY-YY} season, making [Name] an Unrestricted/
       Restricted Free Agent."

    Returns a contract dict, or None if no active contract is found.
    """
    # ── Cap hit + contract length ──────────────────────────────────────────────
    m = re.search(
        r'is signed to (?:a )?(\d+) year[^$]+'
        r'\$[\d,]+ contract(?:\s+extension)?\s+with a cap hit of \$([\d,]+) per season',
        html, re.IGNORECASE
    )
    if not m:
        return None

    contract_years = int(m.group(1))
    cap_hit        = int(m.group(2).replace(',', ''))

    # ── Expiry year ────────────────────────────────────────────────────────────
    exp_m = re.search(
        r'expires at the end of the (\d{4})-\d{2} season',
        html, re.IGNORECASE
    )
    if exp_m:
        expiry_year = int(exp_m.group(1)) + 1   # "2025-26" → 2026
    else:
        expiry_year = season_end_year            # fallback: assume expiring this year

    # ── Expiry status ──────────────────────────────────────────────────────────
    if 'Unrestricted Free Agent' in html or "class='pp-ufa'" in html or 'class="pp-ufa"' in html:
        expiry_status = 'UFA'
    elif 'Restricted Free Agent' in html or "class='pp-rfa'" in html or 'class="pp-rfa"' in html:
        expiry_status = 'RFA'
    elif 'pp-udfa' in html or 'UDFA' in html:
        expiry_status = 'UDFA'
    else:
        expiry_status = 'UFA'

    # ── Derived contract fields ────────────────────────────────────────────────
    years_left       = expiry_year - season_end_year    # ≥ 0
    year_of_contract = max(1, contract_years - years_left)

    return {
        'cap_hit':            cap_hit,
        'length_of_contract': contract_years,
        'expiry_year':        expiry_year,
        'expiry_status':      expiry_status,
        'years_left':         years_left,
        'year_of_contract':   year_of_contract,
    }


# ── Cache ──────────────────────────────────────────────────────────────────────
def _cache_valid() -> bool:
    if not CONTRACTS_CACHE.exists():
        return False
    data = json.loads(CONTRACTS_CACHE.read_text(encoding='utf-8'))
    fetched_at = datetime.fromisoformat(data.get('fetched_at', '2000-01-01'))
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    age_h = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600
    return age_h < CACHE_TTL_HOURS


def _load_cache() -> dict:
    data = json.loads(CONTRACTS_CACHE.read_text(encoding='utf-8'))
    return {int(k): v for k, v in data.get('contracts', {}).items()}


def _save_cache(contracts: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        'fetched_at': datetime.now(timezone.utc).isoformat(),
        'contracts':  {str(k): v for k, v in contracts.items()},
    }
    CONTRACTS_CACHE.write_text(json.dumps(payload, indent=2), encoding='utf-8')


# ── Per-player fetch (runs in thread pool) ─────────────────────────────────────
def _fetch_one(player: dict, season_end_year: int) -> tuple[int, Optional[dict], bool]:
    """
    Fetch and parse one player's contract page.
    Retries once on failure.  Returns (player_id, contract_or_None, had_error).
    """
    pid  = player['player_id']
    slug = _display_to_slug(player['display_name'])
    url  = f"{BASE_URL}/{slug}"
    scraper = _get_scraper()

    time.sleep(REQUEST_DELAY)   # polite per-request delay

    for attempt in range(2):
        try:
            resp = scraper.get(url, timeout=15)
            if resp.status_code == 200:
                return pid, _parse_contract(resp.text, season_end_year), False
            if resp.status_code == 404:
                return pid, None, False
            # Other HTTP error — retry once
            if attempt == 0:
                time.sleep(1.0)
        except Exception:
            if attempt == 0:
                time.sleep(1.0)

    return pid, None, True   # both attempts failed


# ── Main scrape function ───────────────────────────────────────────────────────
def scrape_contracts(
    roster_lookup: dict,
    season_end_year: int,
    force_refresh: bool = False,
) -> dict:
    """
    Scrape PuckPedia for contract data for all players in roster_lookup.

    Parameters
    ----------
    roster_lookup   : name → {player_id, team, position, display_name}
                      (from nhl_api.build_roster_lookup)
    season_end_year : e.g. 2026 for the 2025-26 season
    force_refresh   : ignore the 24-hour cache

    Returns
    -------
    dict : player_id (int) → contract_dict | None
           None means the player has no current contract on PuckPedia.
    """
    if not force_refresh and _cache_valid():
        cached = _load_cache()
        print(f"  Contracts cache hit: {len(cached)} players "
              f"({sum(1 for v in cached.values() if v)} with data)")
        return cached

    if not _HAS_CLOUDSCRAPER:
        print("  WARNING: cloudscraper not installed — cannot scrape PuckPedia. "
              "Run: pip install cloudscraper")
        return {}

    # Build unique player list from roster lookup
    players_seen: dict[int, str] = {}
    for info in roster_lookup.values():
        pid  = info.get('player_id')
        name = info.get('display_name', '')
        if pid and name:
            players_seen[int(pid)] = name

    players = [{'player_id': pid, 'display_name': name}
               for pid, name in players_seen.items()]

    print(f"Scraping {len(players)} player contracts from PuckPedia "
          f"({MAX_WORKERS} workers, cached 24 h)...")

    contracts: dict[int, Optional[dict]] = {}
    errors    = 0
    done      = 0
    lock      = threading.Lock()

    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_one, p, season_end_year): p
            for p in players
        }

        for fut in as_completed(futures):
            pid, contract, had_error = fut.result()

            with lock:
                contracts[pid] = contract
                if had_error:
                    errors += 1
                done += 1
                if done % 50 == 0:
                    have = sum(1 for v in contracts.values() if v)
                    elapsed = time.perf_counter() - t0
                    print(f"  {done}/{len(players)} done  "
                          f"({have} contracts found, {elapsed:.0f}s elapsed)")

    elapsed = time.perf_counter() - t0
    have    = sum(1 for v in contracts.values() if v)
    print(f"  Done. {have}/{len(players)} contracts scraped "
          f"({errors} fetch errors) in {elapsed:.1f}s  "
          f"({elapsed/60:.1f} min).")

    _save_cache(contracts)
    return contracts
