"""
SQLite-backed contract database — source of truth for player salary data.

Schema
──────
  player_id              INTEGER  PRIMARY KEY
  name                   TEXT
  cap_hit                INTEGER  annual salary ($)
  contract_length        INTEGER  total years on the contract
  expiry_year            INTEGER  year contract expires
  expiry_status          TEXT     UFA | RFA | UDFA
  years_left             INTEGER
  year_of_contract       INTEGER  which year of contract we are currently in
  last_verified          TEXT     ISO-8601 datetime (UTC)
  source                 TEXT     puckpedia | spotrac | overthecap | hockeyref
                                  | estimated | override
  is_estimated           INTEGER  0 = real data, 1 = model-estimated fallback
  has_extension          INTEGER  1 if a future extension was found on PuckPedia
  extension_cap_hit      INTEGER  AAV of the signed extension
  extension_start_year   INTEGER  first season-end-year the extension is active
  extension_expiry_year  INTEGER  year extension expires
  extension_length       INTEGER  total years on the extension
  extension_expiry_status TEXT    UFA | RFA of the extension

Overrides
─────────
  contract_overrides.json  — checked FIRST; always wins over the database.
  missing_contracts.json   — players that exhaustive scraper could not find;
                             auto-populated for manual review.
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BASE_DIR       = Path(__file__).parents[2]
DB_PATH        = BASE_DIR / "data" / "contracts.db"
OVERRIDES_PATH = BASE_DIR / "contract_overrides.json"
MISSING_PATH   = BASE_DIR / "missing_contracts.json"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS contracts (
    player_id               INTEGER PRIMARY KEY,
    name                    TEXT,
    cap_hit                 INTEGER,
    contract_length         INTEGER,
    expiry_year             INTEGER,
    expiry_status           TEXT,
    years_left              INTEGER,
    year_of_contract        INTEGER,
    last_verified           TEXT,
    source                  TEXT,
    is_estimated            INTEGER DEFAULT 0,
    has_extension           INTEGER DEFAULT 0,
    extension_cap_hit       INTEGER,
    extension_start_year    INTEGER,
    extension_expiry_year   INTEGER,
    extension_length        INTEGER,
    extension_expiry_status TEXT
);
"""

_EXTENSION_COLS = [
    ("has_extension",           "INTEGER DEFAULT 0"),
    ("extension_cap_hit",       "INTEGER"),
    ("extension_start_year",    "INTEGER"),
    ("extension_expiry_year",   "INTEGER"),
    ("extension_length",        "INTEGER"),
    ("extension_expiry_status", "TEXT"),
]


# ── Connection helper ──────────────────────────────────────────────────────────
def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute(_SCHEMA)
    # Migrate existing databases that predate the extension columns
    for col, typedef in _EXTENSION_COLS:
        try:
            con.execute(f"ALTER TABLE contracts ADD COLUMN {col} {typedef}")
        except sqlite3.OperationalError:
            pass  # column already exists
    con.commit()
    return con


# ── Overrides ──────────────────────────────────────────────────────────────────
def load_overrides() -> dict[int, dict]:
    """Load contract_overrides.json → {player_id: contract_dict}."""
    if not OVERRIDES_PATH.exists():
        return {}
    raw = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()
            if not k.startswith("_")}


# ── CRUD ───────────────────────────────────────────────────────────────────────
def upsert(player_id: int, name: str, data: dict, source: str,
           is_estimated: bool = False) -> None:
    """Insert or replace a contract row."""
    now = datetime.now(timezone.utc).isoformat()
    has_ext = int(bool(data.get("has_extension", False)))
    with _conn() as con:
        con.execute("""
            INSERT INTO contracts
                (player_id, name, cap_hit, contract_length, expiry_year,
                 expiry_status, years_left, year_of_contract,
                 last_verified, source, is_estimated,
                 has_extension, extension_cap_hit, extension_start_year,
                 extension_expiry_year, extension_length, extension_expiry_status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(player_id) DO UPDATE SET
                name                    = excluded.name,
                cap_hit                 = excluded.cap_hit,
                contract_length         = excluded.contract_length,
                expiry_year             = excluded.expiry_year,
                expiry_status           = excluded.expiry_status,
                years_left              = excluded.years_left,
                year_of_contract        = excluded.year_of_contract,
                last_verified           = excluded.last_verified,
                source                  = excluded.source,
                is_estimated            = excluded.is_estimated,
                has_extension           = excluded.has_extension,
                extension_cap_hit       = excluded.extension_cap_hit,
                extension_start_year    = excluded.extension_start_year,
                extension_expiry_year   = excluded.extension_expiry_year,
                extension_length        = excluded.extension_length,
                extension_expiry_status = excluded.extension_expiry_status
        """, (
            player_id,
            name,
            data.get("cap_hit"),
            data.get("length_of_contract"),
            data.get("expiry_year"),
            data.get("expiry_status"),
            data.get("years_left"),
            data.get("year_of_contract"),
            now,
            source,
            int(is_estimated),
            has_ext,
            data.get("extension_cap_hit"),
            data.get("extension_start_year"),
            data.get("extension_expiry_year"),
            data.get("extension_length"),
            data.get("extension_expiry_status"),
        ))


def get_contract(player_id: int) -> Optional[dict]:
    """
    Return contract dict for player_id, or None.
    Overrides file is checked first and always wins.
    """
    overrides = load_overrides()
    if player_id in overrides:
        row = dict(overrides[player_id])
        row["source"] = "override"
        row["is_estimated"] = False
        return row

    with _conn() as con:
        row = con.execute(
            "SELECT * FROM contracts WHERE player_id = ?", (player_id,)
        ).fetchone()
    if row is None:
        return None
    return dict(row)


def get_all_contracts() -> dict[int, Optional[dict]]:
    """
    Return {player_id: contract_dict} for every row in the database.
    Overrides are applied on top.
    """
    with _conn() as con:
        rows = con.execute("SELECT * FROM contracts").fetchall()
    result = {row["player_id"]: dict(row) for row in rows}

    overrides = load_overrides()
    for pid, data in overrides.items():
        row = dict(data)
        row["source"] = "override"
        row["is_estimated"] = False
        result[pid] = row

    return result


def get_players_needing_refresh(season_end_year: int) -> list[int]:
    """
    Return player_ids that should be re-scraped during the weekly contract check:
      1. expiry_year == current season (contract expires this year)
      2. last_verified > 30 days ago AND expiry_year == next season
      3. is_estimated == 1 (estimation placeholder — always retry)
    """
    now      = datetime.now(timezone.utc)
    next_yr  = season_end_year + 1
    cutoff   = (now.timestamp() - 30 * 86400)

    with _conn() as con:
        rows = con.execute("""
            SELECT player_id FROM contracts
            WHERE
                expiry_year = ?
                OR (expiry_year = ? AND
                    CAST(strftime('%s', last_verified) AS INTEGER) < ?)
                OR is_estimated = 1
        """, (season_end_year, next_yr, int(cutoff))).fetchall()

    overrides = set(load_overrides().keys())
    return [r["player_id"] for r in rows if r["player_id"] not in overrides]


def player_ids_in_db() -> set[int]:
    """Return set of all player_ids currently in the database."""
    with _conn() as con:
        rows = con.execute("SELECT player_id FROM contracts").fetchall()
    return {r["player_id"] for r in rows}


# ── Missing-contracts log ──────────────────────────────────────────────────────
def log_missing(player_id: int, name: str, reason: str,
                estimated_cap_hit: Optional[int] = None) -> None:
    """Append a player to missing_contracts.json for manual review."""
    existing: list = []
    if MISSING_PATH.exists():
        existing = json.loads(MISSING_PATH.read_text(encoding="utf-8"))

    # Avoid duplicates
    existing = [e for e in existing if e.get("player_id") != player_id]
    existing.append({
        "player_id":         player_id,
        "name":              name,
        "reason":            reason,
        "estimated_cap_hit": estimated_cap_hit,
        "logged_at":         datetime.now(timezone.utc).isoformat(),
    })
    MISSING_PATH.write_text(
        json.dumps(existing, indent=2), encoding="utf-8"
    )


# ── Database stats ─────────────────────────────────────────────────────────────
def db_stats() -> dict:
    with _conn() as con:
        total     = con.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
        estimated = con.execute(
            "SELECT COUNT(*) FROM contracts WHERE is_estimated=1"
        ).fetchone()[0]
        real      = total - estimated
        sources   = con.execute(
            "SELECT source, COUNT(*) as n FROM contracts GROUP BY source"
        ).fetchall()
    return {
        "total":     total,
        "real":      real,
        "estimated": estimated,
        "by_source": {r["source"]: r["n"] for r in sources},
    }
