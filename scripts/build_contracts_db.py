"""
Build the contracts.db from scratch (or audit + patch gaps).

Usage:
    py -3 scripts/build_contracts_db.py              # use existing cache as seed, then exhaust missing
    py -3 scripts/build_contracts_db.py --full        # ignore cache, re-scrape everything
    py -3 scripts/build_contracts_db.py --missing     # only re-scrape currently missing players
    py -3 scripts/build_contracts_db.py --kings       # show Kings roster summary after build

The script:
  1. Seeds contracts.db from the existing puckpedia cache (fast if cache is fresh)
  2. Identifies players missing from the DB or with is_estimated=1
  3. Runs the exhaustive 7-source scraper against them
  4. Assigns estimated salaries to any still-missing players
  5. Prints a full audit: hit rate, source breakdown, Kings roster, remaining gaps
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.nhl_api import build_roster_lookup, get_season_context, load_all_player_stats
from src.data.puckpedia_scraper import scrape_contracts, _cache_valid, _load_cache
from src.data.exhaustive_scraper import (
    scrape_missing_exhaustive,
    compute_position_medians,
)
from src.data.contracts_db import (
    upsert, get_all_contracts, player_ids_in_db,
    log_missing, db_stats, DB_PATH,
)

RAW_DIR = Path(__file__).parents[1] / "data" / "raw"
STATS_CACHE = RAW_DIR / "stats_cache.json"


def _load_stats_lookup() -> dict[int, dict]:
    """Load stats cache → {player_id: {pos, toi_per_g, gp_24, birth_date, ...}}"""
    if not STATS_CACHE.exists():
        return {}
    raw = json.loads(STATS_CACHE.read_text(encoding="utf-8"))
    players_raw = raw.get("players", raw)   # handle both cache formats
    result = {}
    for pid_str, data in players_raw.items():
        pid = int(pid_str)
        # Flatten season stats into top-level keys
        seasons = data if isinstance(data, dict) else {}
        cur = next((v for k, v in seasons.items()
                    if isinstance(v, dict) and "gp" in v), {})
        result[pid] = {
            "toi_per_g": cur.get("toi_per_g"),
            "gp_24":     None,
            "birth_date": data.get("birth_date", ""),
        }
    return result


def _build_birth_years(stats_lookup: dict) -> dict[int, int]:
    """Extract {player_id: birth_year} from the stats lookup for slug disambiguation."""
    result = {}
    for pid, data in stats_lookup.items():
        bd = data.get("birth_date", "")
        if bd and len(bd) >= 4:
            try:
                result[int(pid)] = int(bd[:4])
            except (ValueError, TypeError):
                pass
    return result


def seed_from_cache(roster: dict, season_end_year: int,
                    birth_years: dict | None = None) -> int:
    """
    Load the puckpedia contracts cache and upsert everything into contracts.db.
    Returns number of records upserted.
    """
    if not _cache_valid():
        print("  No valid contracts cache — running full PuckPedia scrape first...")
        contracts = scrape_contracts(roster, season_end_year, force_refresh=False,
                                     birth_years=birth_years)
    else:
        contracts = _load_cache()
        print(f"  Seeding from cache: {len(contracts)} players")

    # Build name lookup from roster
    name_lookup: dict[int, str] = {}
    for info in roster.values():
        pid = info.get("player_id")
        if pid:
            name_lookup[int(pid)] = info.get("display_name", "")

    n = 0
    for pid, cdata in contracts.items():
        if cdata:
            upsert(int(pid), name_lookup.get(int(pid), ""), cdata,
                   source="puckpedia", is_estimated=False)
            n += 1
    print(f"  Seeded {n} real contracts into contracts.db")
    return n


def build_player_list(roster: dict, stats_lookup: dict, season_end_year: int) -> list[dict]:
    """Build enriched player list for exhaustive scraping."""
    from datetime import date

    players = []
    seen: set[int] = set()

    for _name, info in roster.items():
        pid = info.get("player_id")
        if not pid or int(pid) in seen:
            continue
        seen.add(int(pid))

        stats = stats_lookup.get(int(pid), {})
        birth = info.get("birth_date", "") or stats.get("birth_date", "")
        age   = None
        if birth:
            try:
                bd  = date.fromisoformat(birth)
                ref = date(season_end_year, 9, 30)
                age = (ref - bd).days / 365.25
            except Exception:
                pass

        players.append({
            "player_id":    int(pid),
            "display_name": info.get("display_name", ""),
            "position":     info.get("position", "F"),
            "team":         info.get("team", ""),
            "age":          age,
            "toi_per_g":    stats.get("toi_per_g"),
            "gp_24":        stats.get("gp_24"),
        })

    return players


def main(full: bool = False, missing_only: bool = False, show_kings: bool = False):
    print("\n=== Building contracts.db ===\n")

    ctx = get_season_context()
    season_end_year = ctx["current_season_id"] % 10000
    print(f"Season: {ctx['description']}  (end year {season_end_year})")

    # Roster + stats
    roster = build_roster_lookup()
    print(f"Roster: {len(roster)} players")

    stats_lookup = _load_stats_lookup()
    birth_years  = _build_birth_years(stats_lookup)

    # ── Step 1: Seed DB from existing puckpedia cache ──────────────────────────
    if not missing_only:
        if full:
            print("\n[1/3] Full mode — clearing DB and re-seeding from fresh PuckPedia scrape...")
            from src.data.puckpedia_scraper import scrape_contracts
            contracts = scrape_contracts(roster, season_end_year, force_refresh=True,
                                         birth_years=birth_years)
            name_lookup = {int(info["player_id"]): info.get("display_name", "")
                           for info in roster.values() if info.get("player_id")}
            for pid, cdata in contracts.items():
                if cdata:
                    upsert(int(pid), name_lookup.get(int(pid), ""), cdata,
                           source="puckpedia", is_estimated=False)
        else:
            print("\n[1/3] Seeding DB from existing contracts cache...")
            seed_from_cache(roster, season_end_year, birth_years=birth_years)
    else:
        print("\n[1/3] Missing-only mode — skipping seed step")

    # ── Step 2: Identify gaps ──────────────────────────────────────────────────
    all_players  = build_player_list(roster, stats_lookup, season_end_year)
    db_ids       = player_ids_in_db()
    all_contracts = get_all_contracts()

    missing = [
        p for p in all_players
        if p["player_id"] not in db_ids
        or (all_contracts.get(p["player_id"], {}) or {}).get("is_estimated", 0)
    ]

    print(f"\n[2/3] Players needing exhaustive scrape: {len(missing)}")
    if not missing:
        print("  Nothing to do — database is complete!")
    else:
        # Compute position medians from known real contracts
        real_contracts = [
            {**all_contracts[p["player_id"]], **p}
            for p in all_players
            if p["player_id"] in all_contracts
            and not (all_contracts[p["player_id"]] or {}).get("is_estimated", 0)
        ]
        position_medians = compute_position_medians(real_contracts)
        print(f"  Position medians: {position_medians}")

        # ── Step 3: Exhaustive scrape ──────────────────────────────────────────
        print(f"\n[3/3] Running exhaustive 7-source scrape ({len(missing)} players)...")
        t0 = time.perf_counter()

        results = scrape_missing_exhaustive(missing, season_end_year, position_medians)

        elapsed = time.perf_counter() - t0
        print(f"\n  Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

        # Upsert results
        for player in missing:
            pid = player["player_id"]
            contract, src, is_est = results.get(pid, (None, "unknown", True))
            if contract:
                upsert(pid, player["display_name"], contract,
                       source=src, is_estimated=is_est)
                if is_est:
                    log_missing(pid, player["display_name"],
                                reason=f"All scrape sources failed — estimated from {src}",
                                estimated_cap_hit=contract.get("cap_hit"))

    # ── Audit report ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("CONTRACTS.DB AUDIT")
    print("="*60)

    stats = db_stats()
    all_db = get_all_contracts()
    total_players = len(set(int(info["player_id"]) for info in roster.values()
                            if info.get("player_id")))

    # Breakdown: real with cap_hit / real UFA (cap_hit=None) / estimated
    real_with_cap = sum(1 for r in all_db.values()
                        if r and not r.get("is_estimated") and r.get("cap_hit"))
    real_ufa      = sum(1 for r in all_db.values()
                        if r and not r.get("is_estimated") and not r.get("cap_hit"))

    print(f"\nRoster players:        {total_players}")
    print(f"DB total:              {stats['total']}")
    print(f"  Real + contract:     {real_with_cap}")
    print(f"  Real UFA/unsigned:   {real_ufa}")
    print(f"  Estimated:           {stats['estimated']}")
    print(f"Contract hit rate:     {real_with_cap/total_players*100:.1f}%")
    print(f"Coverage (incl. est):  {stats['total']/total_players*100:.1f}%")
    print(f"\nBy source:")
    for src, n in sorted(stats["by_source"].items(), key=lambda x: -x[1]):
        print(f"  {src:<25} {n:>4}")

    # Still-missing (genuinely not in DB at all)
    all_player_ids = {p["player_id"] for p in all_players}
    not_in_db = all_player_ids - player_ids_in_db()
    if not_in_db:
        not_in_db_names = {p["player_id"]: p["display_name"]
                           for p in all_players if p["player_id"] in not_in_db}
        print(f"\nPlayers NOT in DB at all ({len(not_in_db)}):")
        for pid, name in sorted(not_in_db_names.items(), key=lambda x: x[1]):
            print(f"  {pid:>10}  {name}")

    # Estimated players
    estimated = {pid: row for pid, row in all_db.items()
                 if row and row.get("is_estimated")}
    if estimated:
        print(f"\nEstimated players ({len(estimated)}):")
        name_lookup = {p["player_id"]: (p["display_name"], p["team"])
                       for p in all_players}
        for pid, row in sorted(estimated.items(),
                               key=lambda x: name_lookup.get(x[0], ("z",""))[0]):
            name, team = name_lookup.get(pid, ("?", "?"))
            cap = row.get("cap_hit", 0)
            src = row.get("source", "?")
            print(f"  {name:<25} {team:<4}  est ${cap:>9,}  [{src}]")

    # Kings roster
    print("\n" + "-"*60)
    print("LA KINGS ROSTER — contracts.db")
    print("-"*60)
    kings = [p for p in all_players if p["team"] == "LAK"]
    print(f"{'Name':<25} {'Pos':<4} {'Cap Hit':>12} {'Exp':>5} {'Stat':>6} {'Source'}")
    print("-"*60)
    for p in sorted(kings, key=lambda x: -((all_db.get(x["player_id"], {}) or {}).get("cap_hit") or 0)):
        pid  = p["player_id"]
        row  = all_db.get(pid) or {}
        cap  = row.get("cap_hit")
        exp  = row.get("expiry_year", "?")
        src  = row.get("source", "N/A")[:12]
        flag = "*est*" if row.get("is_estimated") else "     "
        cap_str = f"${cap:>10,}" if cap else "         N/A"
        print(f"  {p['display_name']:<23} {p['position']:<4} {cap_str}  {exp}  {flag}  {src}")

    print(f"\ncontracts.db saved to: {DB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",    action="store_true", help="Full re-scrape (ignore cache)")
    parser.add_argument("--missing", action="store_true", dest="missing_only",
                        help="Only scrape currently-missing players")
    parser.add_argument("--kings",   action="store_true", help="Show Kings roster at end")
    args = parser.parse_args()
    main(full=args.full, missing_only=args.missing_only, show_kings=args.kings)
