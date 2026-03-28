"""
Detect and repair DB entries where a future contract was stored as the current one.

This happens when build_contracts_db.py runs during the off-season: the NHL API
already reports the next season (e.g. season_end_year=2027) so contracts starting
2026-27 pass the start_end_year <= season_end_year filter, even though those players
are still under their old deal during the current 2025-26 season.

Usage:
    py -3 scripts/fix_future_contracts.py [--dry-run]
"""
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.contracts_db import _conn, upsert
from src.data.puckpedia_scraper import _fetch_one
from src.data.nhl_api import get_season_context

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true",
                    help="Show affected players without making changes")
args = parser.parse_args()

ctx             = get_season_context()
season_end_year = ctx["current_season_id"] % 10000
print(f"Season end year: {season_end_year}\n")

# ── Find DB entries where stored contract is a future deal ─────────────────────
# A future contract has: start_end_year = expiry_year - contract_length + 1 > season_end_year
con = _conn()
rows = con.execute("""
    SELECT player_id, name, cap_hit, contract_length, expiry_year, expiry_status,
           source
    FROM contracts
    WHERE contract_length IS NOT NULL
      AND expiry_year IS NOT NULL
      AND (expiry_year - contract_length + 1) > ?
      AND source != 'override'
""", (season_end_year,)).fetchall()
bad_rows = [dict(r) for r in rows]
con.close()

if not bad_rows:
    print("No future-contract entries found in DB. Nothing to fix.")
    sys.exit(0)

print(f"Found {len(bad_rows)} entries where stored contract starts after current season:\n")
print(f"  {'NAME':<28} {'CAP HIT':>10} {'EXP':>5} {'LEN':>4} {'START':>6}  SOURCE")
print("  " + "-" * 68)
for r in sorted(bad_rows, key=lambda x: x['name']):
    start = r['expiry_year'] - r['contract_length'] + 1
    ch = f"${r['cap_hit']:,}" if r.get('cap_hit') else "N/A"
    print(f"  {r['name']:<28} {ch:>10} {r['expiry_year']:>5} {r['contract_length']:>4}yr  "
          f"{start:>5}  {r.get('source', '?')}")

if args.dry_run:
    print("\n[dry-run] No changes made.")
    sys.exit(0)

print(f"\nRe-scraping {len(bad_rows)} players from PuckPedia...\n")

# ── Load birth years for slug disambiguation ───────────────────────────────────
import json
stats_cache = Path("data/raw/stats_cache.json")
birth_years: dict[int, int] = {}
if stats_cache.exists():
    raw = json.loads(stats_cache.read_text(encoding="utf-8"))
    players_raw = raw.get("players", raw)
    for pid_str, data in players_raw.items():
        bd = data.get("birth_date", "")
        if bd and len(bd) >= 4:
            try:
                birth_years[int(pid_str)] = int(bd[:4])
            except (ValueError, TypeError):
                pass

players = [
    {
        "player_id":    r["player_id"],
        "display_name": r["name"],
        **({"birth_year": birth_years[r["player_id"]]}
           if r["player_id"] in birth_years else {}),
    }
    for r in bad_rows
]

# ── Scrape ─────────────────────────────────────────────────────────────────────
scraped: dict[int, dict | None] = {}
errors  = 0
done    = 0
lock    = threading.Lock()
t0      = time.perf_counter()

with ThreadPoolExecutor(max_workers=5) as pool:
    futures = {pool.submit(_fetch_one, p, season_end_year): p for p in players}
    for fut in as_completed(futures):
        pid, contract, had_error = fut.result()
        with lock:
            scraped[pid] = contract
            if had_error:
                errors += 1
            done += 1

elapsed = time.perf_counter() - t0
print(f"Done in {elapsed:.1f}s. {errors} fetch errors.\n")

# ── Update DB ──────────────────────────────────────────────────────────────────
fixed   = 0
skipped = 0
still_bad = []

name_map = {r["player_id"]: r["name"] for r in bad_rows}

print(f"{'NAME':<28} {'OLD CH':>10} {'NEW CH':>10} {'OLD EXP':>7} {'NEW EXP':>7}  STATUS")
print("-" * 78)

for r in bad_rows:
    pid      = r["player_id"]
    name     = r["name"]
    contract = scraped.get(pid)

    if contract is None:
        print(f"  {name:<28} {'—':>10} {'SCRAPE FAIL':>10} {r['expiry_year']:>7}  —       SKIP (no data)")
        skipped += 1
        continue

    new_exp    = contract.get("expiry_year", 0) or 0
    new_length = contract.get("length_of_contract")
    new_start  = (new_exp - new_length + 1) if new_length else new_exp

    if new_start > season_end_year:
        print(f"  {name:<28} ${r['cap_hit']:>9,} {'STILL FUT':>10} {r['expiry_year']:>7} {new_exp:>7}  SKIP (still future)")
        still_bad.append(name)
        skipped += 1
        continue

    old_ch = f"${r['cap_hit']:,}"
    new_ch = f"${contract['cap_hit']:,}" if contract.get('cap_hit') else "N/A"
    print(f"  {name:<28} {old_ch:>10} {new_ch:>10} {r['expiry_year']:>7} {new_exp:>7}  FIXED")
    upsert(pid, name, contract, source="puckpedia", is_estimated=False)
    fixed += 1

print(f"\nSummary: {fixed} fixed, {skipped} skipped")

if still_bad:
    print(f"\nPlayers still showing future contract after re-scrape "
          f"({len(still_bad)}) — add to contract_overrides.json:")
    for name in still_bad:
        print(f"  {name}")
