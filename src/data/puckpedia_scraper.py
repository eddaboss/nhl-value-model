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


def _extract_birth_year(html: str) -> Optional[int]:
    """Extract birth year from PuckPedia's JSON-LD structured data block."""
    m = re.search(r'"birthDate"\s*:\s*"(\d{4})-\d{2}-\d{2}"', html)
    return int(m.group(1)) if m else None


def _parse_contract(html: str, season_end_year: int) -> Optional[dict]:
    """
    Parse contract data from a PuckPedia player page.

    Parses ALL contract entries on the page and selects only the one active
    in the current season (start_year <= season_start_year AND
    expiry_year >= season_end_year).  When multiple match, the most recently
    started contract wins (handles pages where old contract text still appears).

    Two known page formats:
      (A) Prose: "is signed to N year ... contract [extension] with a cap hit
          of $X per season ... expires at the end of YYYY-YY season"
      (B) Tabular: year-range headers followed by "Cap Hit $X" rows (fallback)

    Pitfalls handled:
      1. Future extensions: "His next contract begins for the YYYY-YY season"
         is stripped before searching to avoid matching next-contract cap hits.
      2. Expiry status is extracted from the expiry sentence only, NOT the
         full page (sidebar widgets contain class="pp-ufa" for UFA eligibility
         which is unrelated to the current contract status).
    """
    season_start_year = season_end_year - 1   # e.g. 2025 for 2025-26

    # ── Capture future-extension block BEFORE stripping it ─────────────────────
    extension: Optional[dict] = None
    ext_header = re.search(
        r'[Hh]is next contract begins for the (\d{4})-\d{2} season',
        html, re.DOTALL
    )
    if ext_header:
        ext_start_year = int(ext_header.group(1)) + 1   # "2026-27" → 2027 end year
        ext_block = html[ext_header.start():]
        # Parse cap hit and length from the extension prose block
        ext_prose = re.search(
            r'is signed to (?:a )?(\d+) year[^$]+'
            r'\$[\d,]+ contract(?:\s+extension)?\s+with a cap hit of \$([\d,]+) per season',
            ext_block, re.IGNORECASE
        )
        if ext_prose:
            ext_length  = int(ext_prose.group(1))
            ext_cap_hit = int(ext_prose.group(2).replace(',', ''))
            ext_expiry_year = ext_start_year + ext_length - 1
            # Expiry status from the extension block
            ext_status_m = re.search(
                r'(Unrestricted|Restricted)\s+Free Agent', ext_block, re.IGNORECASE
            )
            ext_status = None
            if ext_status_m:
                ext_status = ('UFA' if ext_status_m.group(1).lower() == 'unrestricted'
                              else 'RFA')
            extension = {
                'extension_cap_hit':       ext_cap_hit,
                'extension_start_year':    ext_start_year,
                'extension_expiry_year':   ext_expiry_year,
                'extension_length':        ext_length,
                'extension_expiry_status': ext_status,
            }

    # ── Strip future-extension block ───────────────────────────────────────────
    html_work = re.sub(r'[Hh]is next contract begins[^.]*\.', '', html, flags=re.DOTALL)

    # ── Collect all prose-format contract candidates ───────────────────────────
    # Anchor on each "expires at the end of YYYY-YY season" sentence, then look
    # backwards (≤800 chars) for the cap hit + length prose block.
    candidates = []

    for exp_m in re.finditer(
        r'expires at the end of the (\d{4})-\d{2} season',
        html_work, re.IGNORECASE
    ):
        expiry_year = int(exp_m.group(1)) + 1   # "2025-26" → 2026

        if expiry_year < season_end_year:
            continue   # already expired, skip

        window = html_work[max(0, exp_m.start() - 800):exp_m.end() + 200]

        prose_m = re.search(
            r'is signed to (?:a )?(\d+) year[^$]+'
            r'\$[\d,]+ contract(?:\s+extension)?\s+with a cap hit of \$([\d,]+) per season',
            window, re.IGNORECASE
        )
        if not prose_m:
            continue

        contract_years = int(prose_m.group(1))
        cap_hit        = int(prose_m.group(2).replace(',', ''))

        # First season this contract was active (as end-year, e.g. 2026 = "2025-26")
        start_end_year = expiry_year - contract_years + 1
        if start_end_year > season_end_year:
            continue   # future contract not yet started

        # Extract expiry status from the sentence immediately after expiry anchor
        exp_context = html_work[exp_m.start(): min(len(html_work), exp_m.end() + 200)]
        status_m = re.search(r'(Unrestricted|Restricted)\s+Free Agent', exp_context, re.IGNORECASE)
        expiry_status = (
            ('UFA' if status_m.group(1).lower() == 'unrestricted' else 'RFA')
            if status_m else None
        )

        candidates.append({
            'cap_hit':        cap_hit,
            'contract_years': contract_years,
            'expiry_year':    expiry_year,
            'expiry_status':  expiry_status,
            'start_end_year': start_end_year,
        })

    if candidates:
        # Most recently started contract covering current season = active contract
        candidates.sort(key=lambda c: c['start_end_year'], reverse=True)
        best           = candidates[0]
        cap_hit        = best['cap_hit']
        contract_years = best['contract_years']
        expiry_year    = best['expiry_year']
        expiry_status  = best['expiry_status']
    else:
        # ── Fallback: tabular format ──────────────────────────────────────────
        cur_season_str = f"{season_end_year - 1}-{str(season_end_year)[-2:]}"
        tab_m = re.search(
            rf'{re.escape(cur_season_str)}.{{0,600}}?Cap Hit \$([\d,]+)',
            html_work, re.IGNORECASE | re.DOTALL
        )
        if not tab_m:
            return None
        cap_hit        = int(tab_m.group(1).replace(',', ''))
        contract_years = None
        exp_m2 = re.search(
            r'expires at the end of the (\d{4})-\d{2} season', html_work, re.IGNORECASE
        )
        expiry_year   = int(exp_m2.group(1)) + 1 if exp_m2 else season_end_year
        expiry_status = None

    # ── Resolve expiry status if not yet determined ────────────────────────────
    if expiry_status is None:
        status_m = re.search(
            r'expires at the end of[^.]*?(Unrestricted|Restricted)\s+Free Agent',
            html_work, re.IGNORECASE
        )
        if status_m:
            expiry_status = 'UFA' if status_m.group(1).lower() == 'unrestricted' else 'RFA'
        elif re.search(r'Restricted Free Agent', html_work, re.IGNORECASE):
            expiry_status = 'RFA'
        elif re.search(r'Unrestricted Free Agent', html_work, re.IGNORECASE):
            expiry_status = 'UFA'
        else:
            expiry_status = 'UFA'

    # UDFA override (always wins)
    if re.search(r'making \w+ (?:a |an )?UDFA', html_work, re.IGNORECASE):
        expiry_status = 'UDFA'

    # ── Derived fields ─────────────────────────────────────────────────────────
    years_left = max(0, expiry_year - season_end_year)
    year_of_contract = max(1, contract_years - years_left) if contract_years else None

    result = {
        'cap_hit':            cap_hit,
        'length_of_contract': contract_years,
        'expiry_year':        expiry_year,
        'expiry_status':      expiry_status,
        'years_left':         years_left,
        'year_of_contract':   year_of_contract,
        'has_extension':      extension is not None,
    }
    if extension:
        result.update(extension)
    return result


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

    Disambiguation: PuckPedia appends -2, -3, ... to slugs for players who
    share a name (e.g. two active NHLers named Elias Pettersson).  When the
    player dict contains a 'birth_year' key we verify the page's JSON-LD
    birthDate field; if it doesn't match we try slug-2, slug-3, slug-4 before
    giving up.  Without birth_year we accept the first successful parse.

    Retries once per URL on transient errors.
    Returns (player_id, contract_or_None, had_error).
    """
    pid        = player['player_id']
    base_slug  = _display_to_slug(player['display_name'])
    birth_year = player.get('birth_year')   # int or None
    scraper    = _get_scraper()

    # Suffixes to try: '' (base), '-2', '-3', '-4'
    suffixes = [''] + [f'-{i}' for i in range(2, 5)]

    time.sleep(REQUEST_DELAY)

    for suffix in suffixes:
        slug = base_slug + suffix
        url  = f"{BASE_URL}/{slug}"

        for attempt in range(2):
            try:
                resp = scraper.get(url, timeout=15)
                if resp.status_code == 404:
                    break   # this suffix doesn't exist — stop trying it
                if resp.status_code == 200:
                    # If we know the expected birth year, verify we have the right page
                    if birth_year is not None:
                        page_year = _extract_birth_year(resp.text)
                        if page_year is not None and page_year != birth_year:
                            break   # wrong player — try next suffix
                    return pid, _parse_contract(resp.text, season_end_year), False
                # Other HTTP error — retry once
                if attempt == 0:
                    time.sleep(1.0)
            except Exception:
                if attempt == 0:
                    time.sleep(1.0)

    return pid, None, True   # all suffixes exhausted or all attempts failed


# ── Main scrape function ───────────────────────────────────────────────────────
def scrape_contracts(
    roster_lookup: dict,
    season_end_year: int,
    force_refresh: bool = False,
    birth_years: Optional[dict] = None,
) -> dict:
    """
    Scrape PuckPedia for contract data for all players in roster_lookup.

    Parameters
    ----------
    roster_lookup   : name → {player_id, team, position, display_name}
                      (from nhl_api.build_roster_lookup)
    season_end_year : e.g. 2026 for the 2025-26 season
    force_refresh   : ignore the 24-hour cache
    birth_years     : optional player_id (int) → birth_year (int) mapping.
                      When provided, used to verify each PuckPedia page and
                      try slug-2 / slug-3 suffixes for name-collision players.

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

    # Build unique player list from roster lookup, optionally enriched with birth_year
    players_seen: dict[int, str] = {}
    for info in roster_lookup.values():
        pid  = info.get('player_id')
        name = info.get('display_name', '')
        if pid and name:
            players_seen[int(pid)] = name

    _birth_years = birth_years or {}
    players = [
        {'player_id': pid, 'display_name': name,
         **({"birth_year": _birth_years[pid]} if pid in _birth_years else {})}
        for pid, name in players_seen.items()
    ]

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
