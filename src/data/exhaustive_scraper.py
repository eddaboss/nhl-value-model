"""
Exhaustive multi-source contract scraper.

For each player, tries sources in order until one succeeds:
  1. PuckPedia — base slug
  2. PuckPedia — disambiguation suffixes (-1, -2, -3)
  3. PuckPedia — apostrophe / punctuation / middle-name variants
  4. OverTheCap — player page
  5. Spotrac     — player search
  6. Hockey Reference — constructed player ID
  7. Salary estimation (ELC / mid-tier / veteran) — never fails

Only after all seven fail is a player assigned an estimated salary with
is_estimated=True and logged to missing_contracts.json for manual review.
"""
import re
import time
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

try:
    import cloudscraper as _cs
    _HAS_CLOUDSCRAPER = True
except ImportError:
    _HAS_CLOUDSCRAPER = False

try:
    from rapidfuzz import process as _rfprocess
    _HAS_RF = True
except ImportError:
    _HAS_RF = False

PUCKPEDIA_BASE  = "https://puckpedia.com/player"
OTC_BASE        = "https://overthecap.com/player"
SPOTRAC_SEARCH  = "https://www.spotrac.com/search/?search="
HR_BASE         = "https://www.hockey-reference.com/players"

MAX_WORKERS      = 10
PER_REQUEST_WAIT = 0.1   # seconds


# ── Thread-local cloudscraper ──────────────────────────────────────────────────
_local = threading.local()


def _get_scraper():
    if not hasattr(_local, "scraper"):
        _local.scraper = _cs.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
    return _local.scraper


# ── Slug / ID builders ─────────────────────────────────────────────────────────
# German/Scandinavian umlaut convention used by PuckPedia
_UMLAUT_MAP = str.maketrans({
    'ü': 'ue', 'Ü': 'ue',
    'ö': 'oe', 'Ö': 'oe',
    'ä': 'ae', 'Ä': 'ae',
    'ß': 'ss',
})

# Common first-name nicknames → legal names PuckPedia may use
_NICKNAME_MAP = {
    'mitch': 'mitchell', 'matt': 'matthew', 'matty': 'matthew',
    'jake': 'jacob', 'mike': 'michael', 'nick': 'nicholas',
    'cam': 'cameron', 'zach': 'zachary', 'kris': 'kristopher',
    'alex': 'alexander', 'tj': 'tyler',
    'pat': 'patrick', 'nate': 'nathaniel', 'will': 'william',
    'andy': 'andrew', 'danny': 'daniel', 'tommy': 'thomas',
    'rick': 'richard', 'jeff': 'jeffrey', 'rob': 'robert',
    'jon': 'jonathan', 'josh': 'joshua', 'ben': 'benjamin',
    'sam': 'samuel', 'max': 'maxime',
    # Short-form first names that PuckPedia stores as full legal names
    'chris': 'christopher', 'tony': 'anthony', 'vince': 'vincent',
    'brent': 'brenton', 'corey': 'cory', 'col': 'colton',
    'zac': 'zachary', 'trey': 'trevor', 'cal': 'calvin',
    'gus': 'gustave', 'mat': 'matthew', 'cale': 'caleb',
}


def _to_slug(text: str) -> str:
    """Normalise → lowercase ASCII slug (NFKD accent stripping)."""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_ = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", ascii_.lower()).strip("-")


def _to_slug_umlaut(text: str) -> str:
    """Slug using German umlaut convention: ü→ue, ö→oe, ä→ae."""
    text2 = text.translate(_UMLAUT_MAP)
    nfkd   = unicodedata.normalize("NFKD", text2)
    ascii_ = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", ascii_.lower()).strip("-")


def _puckpedia_slug_variants(display_name: str) -> list[str]:
    """
    All PuckPedia slug variants to try, in priority order.
    Covers: disambiguation, apostrophes, periods, German umlauts,
    first-name nicknames, and hyphenated-name variations.
    """
    seen: list[str] = []

    def _add(s: str):
        s = s.strip("-")
        if s and s not in seen:
            seen.append(s)

    def _add_with_disambig(v: str, n: int = 3):
        _add(v)
        for i in range(1, n + 1):
            _add(f"{v}-{i}")

    # ── Canonical normalisations ───────────────────────────────────────────────
    base        = _to_slug(display_name)         # NFKD accent stripping
    base_umlaut = _to_slug_umlaut(display_name)  # German ü→ue convention

    _add_with_disambig(base)
    if base_umlaut != base:
        _add_with_disambig(base_umlaut)

    # ── Apostrophe variants ────────────────────────────────────────────────────
    no_apos = display_name.replace("'", "")
    _add_with_disambig(_to_slug(no_apos), 2)
    _add(_to_slug(display_name.replace("'", "-")))

    # ── Dots / periods (e.g. "P.K. Subban" → "pk-subban") ────────────────────
    no_dots = re.sub(r"\.", "", display_name)
    _add_with_disambig(_to_slug(no_dots), 2)

    # ── Jr./Sr./suffix removal ─────────────────────────────────────────────────
    no_suffix = re.sub(r"\b(Jr\.?|Sr\.?|III?|IV)\b", "", display_name, flags=re.I)
    _add(_to_slug(no_suffix.strip()))

    # ── Middle name / initial stripping ───────────────────────────────────────
    parts = display_name.split()
    if len(parts) > 2:
        first_last = f"{parts[0]} {parts[-1]}"
        _add_with_disambig(_to_slug(first_last), 2)

    # ── Nickname → legal first name (e.g. "Mitch" → "Mitchell") ─────────────
    if parts:
        first_lower = re.sub(r"[^a-z]", "", parts[0].lower())
        legal_first = _NICKNAME_MAP.get(first_lower)
        if legal_first:
            legal_name = f"{legal_first} {' '.join(parts[1:])}"
            _add_with_disambig(_to_slug(legal_name), 2)
            _add_with_disambig(_to_slug_umlaut(legal_name), 2)

    # ── Hyphenated names: compound + split ────────────────────────────────────
    if "-" in display_name:
        no_hyph = display_name.replace("-", "")
        _add(_to_slug(no_hyph))
        # Also try each side of the hyphen as "first" name
        for part in display_name.split("-"):
            rest_name = display_name.replace(f"{part}-", "").replace(f"-{part}", "")
            _add(_to_slug(f"{part} {rest_name}"))

    return seen


def _hr_player_id(display_name: str) -> Optional[str]:
    """
    Construct a likely Hockey Reference player ID.
    Pattern: last[:5] + first[:2] + '01'
    e.g. "Anze Kopitar" → "kopitan01"
    """
    parts = display_name.split()
    if len(parts) < 2:
        return None
    first = re.sub(r"[^a-z]", "", parts[0].lower())
    last  = re.sub(r"[^a-z]", "", parts[-1].lower())
    if not first or not last:
        return None
    return last[:5] + first[:2] + "01"


# ── Source-specific fetchers ───────────────────────────────────────────────────
def _fetch_url(url: str, timeout: int = 12) -> Optional[str]:
    """GET url, return text or None on any error / non-200."""
    scraper = _get_scraper()
    time.sleep(PER_REQUEST_WAIT)
    try:
        resp = scraper.get(url, timeout=timeout)
        return resp.text if resp.status_code == 200 else None
    except Exception:
        return None


_UFA_DICT = {
    "cap_hit": None, "length_of_contract": None,
    "expiry_year": None, "expiry_status": "UFA",
    "years_left": 0, "year_of_contract": None,
    "_no_contract": True,   # sentinel: real UFA, not a parse failure
}


def _parse_puckpedia(html: str, season_end_year: int) -> Optional[dict]:
    """
    Parse PuckPedia player page HTML.

    Parses ALL contract entries on the page and selects the one active in the
    current season (most recently started contract with expiry >= season_end_year).
    Returns contract dict, _UFA_DICT if player is unsigned/UFA, or None on failure.
    """
    # Strip future-extension narrative before searching
    html_work = re.sub(r"[Hh]is next contract begins[^.]*\.", "", html, flags=re.DOTALL)

    # Check for unsigned / free agent page (no contract text at all)
    if not re.search(r"is signed to", html_work, re.IGNORECASE):
        if re.search(
            r"is (?:an?\s+)?(?:Unrestricted|Restricted) Free Agent",
            html_work, re.IGNORECASE,
        ):
            return _UFA_DICT
        return None

    # Collect all prose-format contract candidates anchored on expiry sentences
    candidates = []
    for exp_m in re.finditer(
        r"expires at the end of the (\d{4})-\d{2} season", html_work, re.IGNORECASE
    ):
        expiry_year = int(exp_m.group(1)) + 1   # "2025-26" → 2026
        if expiry_year < season_end_year:
            continue   # already expired

        window = html_work[max(0, exp_m.start() - 800): exp_m.end() + 200]
        prose_m = re.search(
            r"is signed to (?:a )?(\d+) year[^$]+"
            r"\$[\d,]+ contract(?:\s+extension)?\s+with a cap hit of \$([\d,]+) per season",
            window, re.IGNORECASE,
        )
        if not prose_m:
            continue

        contract_years = int(prose_m.group(1))
        cap_hit        = int(prose_m.group(2).replace(",", ""))
        start_end_year = expiry_year - contract_years + 1
        if start_end_year > season_end_year:
            continue   # future contract not yet started

        candidates.append({
            "cap_hit":        cap_hit,
            "contract_years": contract_years,
            "expiry_year":    expiry_year,
            "start_end_year": start_end_year,
        })

    if not candidates:
        return None

    # Most recently started contract covering current season = active contract
    candidates.sort(key=lambda c: c["start_end_year"], reverse=True)
    best           = candidates[0]
    cap_hit        = best["cap_hit"]
    contract_years = best["contract_years"]
    expiry_year    = best["expiry_year"]

    # Expiry status — from expiry sentence only (avoid sidebar widget false matches)
    exp_ctx_m = re.search(
        r"expires at the end of[^.]*?(Unrestricted|Restricted)\s+Free Agent",
        html_work, re.IGNORECASE,
    )
    if exp_ctx_m:
        expiry_status = "UFA" if exp_ctx_m.group(1).lower() == "unrestricted" else "RFA"
    elif re.search(r"pp-udfa|making \w+ (?:a |an )?UDFA", html_work, re.IGNORECASE):
        expiry_status = "UDFA"
    elif re.search(r"Restricted Free Agent", html_work, re.IGNORECASE):
        expiry_status = "RFA"
    elif re.search(r"Unrestricted Free Agent", html_work, re.IGNORECASE):
        expiry_status = "UFA"
    else:
        expiry_status = "UFA"

    years_left       = max(0, expiry_year - season_end_year)
    year_of_contract = max(1, contract_years - years_left)

    return {
        "cap_hit":            cap_hit,
        "length_of_contract": contract_years,
        "expiry_year":        expiry_year,
        "expiry_status":      expiry_status,
        "years_left":         years_left,
        "year_of_contract":   year_of_contract,
    }


def _parse_overthecap(html: str, season_end_year: int) -> Optional[dict]:
    """
    Parse OverTheCap player page HTML.
    OTC shows cap hit as "Cap Hit: $X,XXX,XXX" or in a <td> with class "salary".
    """
    # Try AAV / cap hit value
    m = re.search(
        r"(?:Cap Hit|AAV)[:\s]+\$\s*([\d,]+)", html, re.IGNORECASE
    )
    if not m:
        # Fallback: look for labelled salary line
        m = re.search(r"\$([\d,]+)\s*/\s*year", html, re.IGNORECASE)
    if not m:
        return None

    cap_hit = int(m.group(1).replace(",", ""))
    if cap_hit < 750_000:
        return None   # noise / wrong match

    # Contract length
    cl_m = re.search(r"(\d+)[- ]year contract", html, re.IGNORECASE)
    contract_years = int(cl_m.group(1)) if cl_m else 1

    # Expiry year
    exp_m = re.search(r"(\d{4})-\d{2,4}\s+season", html, re.IGNORECASE)
    expiry_year = (int(exp_m.group(1)) + 1) if exp_m else season_end_year

    # Expiry status
    if "unrestricted" in html.lower():
        expiry_status = "UFA"
    elif "restricted" in html.lower():
        expiry_status = "RFA"
    else:
        expiry_status = "UFA"

    years_left       = max(0, expiry_year - season_end_year)
    year_of_contract = max(1, contract_years - years_left)

    return {
        "cap_hit":            cap_hit,
        "length_of_contract": contract_years,
        "expiry_year":        expiry_year,
        "expiry_status":      expiry_status,
        "years_left":         years_left,
        "year_of_contract":   year_of_contract,
    }


def _parse_hockeyref(html: str, season_end_year: int) -> Optional[dict]:
    """
    Parse Hockey Reference player page.
    Salary table rows look like: <td ...>$X,XXX,XXX</td>
    """
    # Find salary for current season
    cap_m = re.search(r"\$\s*([\d,]{7,})", html)   # at least 7 chars → $XXX,XXX+
    if not cap_m:
        return None
    cap_hit = int(cap_m.group(1).replace(",", ""))
    if cap_hit < 750_000:
        return None

    return {
        "cap_hit":            cap_hit,
        "length_of_contract": 1,
        "expiry_year":        season_end_year,
        "expiry_status":      "UFA",
        "years_left":         0,
        "year_of_contract":   1,
    }


# ── Source 1-6: attempt scraped data ──────────────────────────────────────────
def _try_puckpedia(display_name: str, season_end_year: int
                   ) -> tuple[Optional[dict], str]:
    """
    Try all PuckPedia slug variants.
    Returns (contract, slug_used).
    contract may be _UFA_DICT (player is confirmed UFA/unsigned).
    """
    for slug in _puckpedia_slug_variants(display_name):
        html = _fetch_url(f"{PUCKPEDIA_BASE}/{slug}")
        if html:
            contract = _parse_puckpedia(html, season_end_year)
            if contract is not None:   # includes _UFA_DICT case
                return contract, f"puckpedia:{slug}"
    return None, ""


def _try_overthecap(display_name: str, season_end_year: int
                    ) -> tuple[Optional[dict], str]:
    """Try OverTheCap player page."""
    slug  = _to_slug(display_name)
    parts = display_name.lower().split()
    # OTC sometimes uses last-first ordering or just first-last
    slugs_to_try = [slug]
    if len(parts) >= 2:
        slugs_to_try.append(_to_slug(f"{parts[-1]} {parts[0]}"))  # last-first

    for s in slugs_to_try:
        html = _fetch_url(f"{OTC_BASE}/{s}")
        if html:
            contract = _parse_overthecap(html, season_end_year)
            if contract:
                return contract, f"overthecap:{s}"
    return None, ""


def _try_hockeyref(display_name: str, season_end_year: int
                   ) -> tuple[Optional[dict], str]:
    """Try Hockey Reference player page (constructed ID)."""
    # Normalize name first (strip accents)
    nfkd  = unicodedata.normalize("NFKD", display_name)
    ascii_ = nfkd.encode("ascii", "ignore").decode("ascii")
    hr_id  = _hr_player_id(ascii_)
    if not hr_id:
        return None, ""

    first_letter = hr_id[0]
    url = f"{HR_BASE}/{first_letter}/{hr_id}.html"
    html = _fetch_url(url)
    if html:
        # Verify it's the right player (name appears in page)
        check_name = display_name.split()[0].lower()
        if check_name not in html.lower():
            # Try -02 variant for same-name players
            hr_id2 = hr_id[:-2] + "02"
            url2   = f"{HR_BASE}/{first_letter}/{hr_id2}.html"
            html2  = _fetch_url(url2)
            if html2 and check_name in html2.lower():
                html = html2
                hr_id = hr_id2
            else:
                html = None

    if html:
        contract = _parse_hockeyref(html, season_end_year)
        if contract:
            return contract, f"hockeyref:{hr_id}"

    return None, ""


# ── Source 7: salary estimation ────────────────────────────────────────────────
def estimate_salary(
    player_id: int,
    display_name: str,
    age: Optional[float],
    position: Optional[str],
    toi_per_g: Optional[float],
    gp_24: Optional[float],
    season_end_year: int,
    position_medians: dict,
) -> dict:
    """
    Assign an estimated salary when no scraped source succeeds.

    Tiers:
      ELC     — age < 25 AND (no prior season OR gp_24 < 50)
      Mid     — 3-6 seasons estimated (gp_24 exists, age < 30)
      Veteran — age >= 30 or has multiple prior seasons
    """
    age      = age or 26
    toi      = toi_per_g or 12.0
    has_prev = gp_24 is not None and gp_24 > 10
    pos      = (position or "F").upper()
    pos_key  = "D" if pos == "D" else "F"

    # Classify tier
    is_elc = (age < 25) and not has_prev
    is_mid = not is_elc and age < 30
    # else veteran

    if is_elc:
        cap_hit      = 975_000
        exp_years    = max(1, 25 - int(age))
        expiry_year  = season_end_year + min(exp_years, 3)
        expiry_status = "RFA"
        contract_len  = min(exp_years, 3)
    else:
        # Ice time tier for mid/veteran
        if toi >= 20:
            tier = "top"
        elif toi >= 15:
            tier = "mid"
        else:
            tier = "bottom"

        key = f"{pos_key}_{tier}"
        cap_hit = position_medians.get(key, position_medians.get(f"{pos_key}_mid", 2_500_000))

        if is_mid:
            contract_len  = 2
            expiry_year   = season_end_year + 2
            expiry_status = "RFA" if age < 27 else "UFA"
        else:
            contract_len  = 1
            expiry_year   = season_end_year + 1
            expiry_status = "UFA"

    years_left       = max(0, expiry_year - season_end_year)
    year_of_contract = max(1, contract_len - years_left)

    return {
        "cap_hit":            cap_hit,
        "length_of_contract": contract_len,
        "expiry_year":        expiry_year,
        "expiry_status":      expiry_status,
        "years_left":         years_left,
        "year_of_contract":   year_of_contract,
    }


# ── Single-player orchestrator ─────────────────────────────────────────────────
def scrape_player_exhaustive(
    player: dict,
    season_end_year: int,
    position_medians: dict,
) -> tuple[int, Optional[dict], str, bool]:
    """
    Try all scraped sources, then fall back to estimation.

    Returns (player_id, contract_dict, source_label, is_estimated).
    """
    pid          = player["player_id"]
    display_name = player["display_name"]
    age          = player.get("age")
    position     = player.get("position")
    toi_per_g    = player.get("toi_per_g")
    gp_24        = player.get("gp_24")

    if not _HAS_CLOUDSCRAPER:
        contract = estimate_salary(
            pid, display_name, age, position, toi_per_g, gp_24,
            season_end_year, position_medians,
        )
        return pid, contract, "estimated:no_cloudscraper", True

    # Source 1-3: PuckPedia (all slug variants)
    contract, src = _try_puckpedia(display_name, season_end_year)
    if contract is not None:
        # _no_contract=True → confirmed UFA/unsigned, not a failure
        is_ufa = contract.get("_no_contract", False)
        clean  = {k: v for k, v in contract.items() if k != "_no_contract"}
        return pid, clean, src, False

    # Source 4: OverTheCap
    contract, src = _try_overthecap(display_name, season_end_year)
    if contract:
        return pid, contract, src, False

    # Source 5: Hockey Reference
    contract, src = _try_hockeyref(display_name, season_end_year)
    if contract:
        return pid, contract, src, False

    # Source 6: Spotrac — search-based (best effort)
    search_url = SPOTRAC_SEARCH + display_name.replace(" ", "+")
    html = _fetch_url(search_url)
    if html:
        # Spotrac search returns HTML list; look for salary patterns in preview
        m = re.search(r"\$([\d,]{7,})\s*(?:cap hit|AAV|/\s*yr)", html, re.IGNORECASE)
        if m:
            cap_hit = int(m.group(1).replace(",", ""))
            if cap_hit > 750_000:
                contract = {
                    "cap_hit":            cap_hit,
                    "length_of_contract": 1,
                    "expiry_year":        season_end_year,
                    "expiry_status":      "UFA",
                    "years_left":         0,
                    "year_of_contract":   1,
                }
                return pid, contract, "spotrac:search", False

    # Source 7: Salary estimation
    contract = estimate_salary(
        pid, display_name, age, position, toi_per_g, gp_24,
        season_end_year, position_medians,
    )
    return pid, contract, "estimated", True


# ── Batch orchestrator (parallel) ─────────────────────────────────────────────
def scrape_missing_exhaustive(
    players: list[dict],
    season_end_year: int,
    position_medians: dict,
) -> dict[int, tuple[Optional[dict], str, bool]]:
    """
    Run exhaustive scraping for a list of players in parallel.

    Returns {player_id: (contract_dict, source, is_estimated)}.
    """
    results: dict[int, tuple] = {}
    lock  = threading.Lock()
    done  = 0
    total = len(players)

    import time as _time
    t0 = _time.perf_counter()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(
                scrape_player_exhaustive, p, season_end_year, position_medians
            ): p
            for p in players
        }
        for fut in as_completed(futures):
            pid, contract, src, is_est = fut.result()
            with lock:
                results[pid] = (contract, src, is_est)
                done += 1
                if done % 20 == 0 or done == total:
                    found = sum(1 for c, _, e in results.values() if c and not e)
                    est   = sum(1 for _, _, e in results.values() if e)
                    elapsed = _time.perf_counter() - t0
                    print(f"  {done}/{total}  scraped={found}  estimated={est}"
                          f"  ({elapsed:.0f}s)")

    return results


# ── Position medians helper ────────────────────────────────────────────────────
def compute_position_medians(contracts_db_rows: list[dict]) -> dict:
    """
    Compute salary medians by (position, ice-time tier) from known contracts.
    Used as the estimation baseline.

    Input: list of dicts with keys pos, toi_per_g, cap_hit.
    """
    import statistics

    buckets: dict[str, list[int]] = {
        "F_top": [], "F_mid": [], "F_bottom": [],
        "D_top": [], "D_mid": [], "D_bottom": [],
    }

    for row in contracts_db_rows:
        cap  = row.get("cap_hit")
        pos  = (row.get("pos") or "F").upper()
        toi  = row.get("toi_per_g") or 12.0
        if not cap or cap < 750_000:
            continue
        pk = "D" if pos == "D" else "F"
        if toi >= 20:
            tier = "top"
        elif toi >= 15:
            tier = "mid"
        else:
            tier = "bottom"
        buckets[f"{pk}_{tier}"].append(cap)

    medians = {}
    for key, vals in buckets.items():
        if vals:
            medians[key] = int(statistics.median(vals))

    # Fill any empty buckets with overall median
    all_vals = [c for row in contracts_db_rows
                if (c := row.get("cap_hit")) and c > 750_000]
    overall = int(statistics.median(all_vals)) if all_vals else 2_500_000
    for key in buckets:
        medians.setdefault(key, overall)

    return medians
