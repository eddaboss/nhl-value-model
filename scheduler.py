"""
Automated refresh scheduler.

Nightly (11 PM PST / 02:00 UTC next day):
  - Pull fresh NHL API stats only
  - Retrain model on fresh stats + existing contracts.db
  - Log timestamp + CV R²

Weekly (Sunday midnight PST / 08:00 UTC Sunday):
  - Re-scrape only: expiring players, stale verifications, previously missing
  - Update contracts.db incrementally
  - Retrain model on updated data

Usage:
    py -3 scheduler.py                # start blocking scheduler
    py -3 scheduler.py --run-stats    # manual: stats refresh + retrain right now
    py -3 scheduler.py --run-contracts # manual: weekly contract check right now

Logs:  data/logs/scheduler.log
State: data/processed/last_updated.json
"""
import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ── Logging setup ──────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "scheduler.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

LAST_UPDATED = Path(__file__).parent / "data" / "processed" / "last_updated.json"
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"


# ── Run helpers ────────────────────────────────────────────────────────────────
def _write_last_updated(event: str, extra: dict | None = None) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if LAST_UPDATED.exists():
        try:
            existing = json.loads(LAST_UPDATED.read_text(encoding="utf-8"))
        except Exception:
            pass
    existing[event] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }
    LAST_UPDATED.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def run_nightly_stats() -> None:
    """
    Stats-only nightly refresh:
      1. Fetch fresh NHL API stats (force_refresh=True)
      2. Use existing contracts.db (no contract scraping)
      3. Retrain model, save predictions
    """
    log.info("=== Nightly stats refresh starting ===")
    t0 = time.perf_counter()

    try:
        import warnings
        warnings.filterwarnings("ignore")
        sys.path.insert(0, str(Path(__file__).parent))

        from src.data.load import load_and_merge
        from src.features.build import build_features, get_feature_matrix
        from src.models.train import build_pipelines, evaluate, train_and_save
        from src.models.explain import save_shap_artifacts
        from src.data.load import CAP_CEILING, save_processed
        import pandas as pd
        import json as _json

        # Load with fresh stats, existing contracts
        df_raw, ctx = load_and_merge(force_refresh=True, force_refresh_contracts=False)
        df = build_features(df_raw)
        X, y_full = get_feature_matrix(df)

        train_mask = df["has_contract_data"].fillna(False).astype(bool)
        X_train    = X[train_mask].copy()
        y_train    = y_full[train_mask].copy()
        y_norm     = y_train / CAP_CEILING

        pipes   = build_pipelines(X_train)
        results = evaluate(pipes, X_train, y_norm)
        results["rmse_dollars"] = (results["rmse_mean"] * CAP_CEILING).round(0).astype(int)

        best_name  = results.iloc[0]["model"]
        final_pipe = train_and_save(pipes[best_name], X_train, y_norm, best_name)

        df["predicted_value"] = final_pipe.predict(X) * CAP_CEILING
        df["value_delta"] = df.apply(
            lambda r: r["predicted_value"] - r["cap_hit"]
            if r.get("has_contract_data") else None, axis=1
        )

        from pipeline import resign_label
        df["resign_signal"] = df.apply(resign_label, axis=1)

        keep = ["name","team","pos","age","cap_hit","predicted_value","value_delta",
                "expiry_status","expiry_year","years_left","length_of_contract",
                "gp","g","a","p","ppg","toi_per_g","plus_minus","pim",
                "g60","p60","pp_pts","shots","shooting_pct",
                "resign_signal","player_id","has_contract_data","has_prior_market_data",
                "is_estimated"]
        out_df = df[[c for c in keep if c in df.columns]].copy()
        for col in ["cap_hit","predicted_value","value_delta"]:
            if col in out_df.columns:
                out_df[col] = pd.to_numeric(out_df[col], errors="coerce").round(0)
                out_df[col] = out_df[col].astype("Int64")

        save_processed(out_df, "predictions.csv")

        best_r2   = float(results.iloc[0]["r2_mean"])
        best_rmse = int(results.iloc[0]["rmse_dollars"])
        elapsed   = time.perf_counter() - t0

        # Save season context
        _json.dump(ctx, open(PROCESSED_DIR / "season_context.json", "w"), indent=2)

        save_shap_artifacts(final_pipe, X_train, df[train_mask]["name"])

        log.info(f"Nightly stats done — R²={best_r2:.4f}  RMSE=${best_rmse:,}  "
                 f"({len(X_train)} players, {elapsed:.0f}s)")

        _write_last_updated("nightly_stats", {
            "r2":        best_r2,
            "rmse":      best_rmse,
            "n_players": int(len(X_train)),
            "model":     best_name,
        })

    except Exception as exc:
        log.error(f"Nightly stats FAILED: {exc}\n{traceback.format_exc()}")
        _write_last_updated("nightly_stats_error", {"error": str(exc)})


def run_weekly_contracts() -> None:
    """
    Weekly incremental contract check:
      1. Identify players needing re-verification (expiring, stale, missing)
      2. Run exhaustive scraper only on those players
      3. Update contracts.db
      4. Retrain model
    """
    log.info("=== Weekly contract check starting ===")
    t0 = time.perf_counter()

    try:
        sys.path.insert(0, str(Path(__file__).parent))

        from src.data.nhl_api import build_roster_lookup, get_season_context
        from src.data.contracts_db import (
            get_players_needing_refresh, player_ids_in_db,
            get_all_contracts, upsert, log_missing, db_stats,
        )
        from src.data.exhaustive_scraper import (
            scrape_missing_exhaustive, compute_position_medians,
        )
        from datetime import date

        ctx = get_season_context()
        season_end_year = ctx["current_season_id"] % 10000
        roster = build_roster_lookup()

        # Build player info lookup
        players_by_id: dict = {}
        for info in roster.values():
            pid = info.get("player_id")
            if pid:
                bid = info.get("birth_date", "") or ""
                age = None
                if bid:
                    try:
                        bd  = date.fromisoformat(bid)
                        ref = date(season_end_year, 9, 30)
                        age = (ref - bd).days / 365.25
                    except Exception:
                        pass
                players_by_id[int(pid)] = {
                    "player_id":    int(pid),
                    "display_name": info.get("display_name", ""),
                    "position":     info.get("position", "F"),
                    "team":         info.get("team", ""),
                    "age":          age,
                }

        # New players not yet in DB
        known_ids   = player_ids_in_db()
        new_players = [p for pid, p in players_by_id.items() if pid not in known_ids]

        # Players needing re-verification
        stale_ids   = get_players_needing_refresh(season_end_year)
        stale_players = [players_by_id[pid] for pid in stale_ids if pid in players_by_id]

        to_refresh = {p["player_id"]: p for p in new_players + stale_players}
        log.info(f"  New: {len(new_players)}  Stale/expiring: {len(stale_players)}  "
                 f"Total to scrape: {len(to_refresh)}")

        if to_refresh:
            all_contracts = get_all_contracts()
            real_rows = [
                {**all_contracts[pid], "pos": players_by_id.get(pid, {}).get("position","F")}
                for pid in all_contracts
                if all_contracts[pid] and all_contracts[pid].get("cap_hit")
                and not all_contracts[pid].get("is_estimated")
            ]
            position_medians = compute_position_medians(real_rows)

            results = scrape_missing_exhaustive(
                list(to_refresh.values()), season_end_year, position_medians
            )

            scraped = 0
            estimated = 0
            for pid, (contract, src, is_est) in results.items():
                p = to_refresh[pid]
                if contract:
                    upsert(pid, p["display_name"], contract, src, is_estimated=is_est)
                    if is_est:
                        estimated += 1
                        log_missing(pid, p["display_name"],
                                    reason=f"Weekly check: all sources failed — est from {src}",
                                    estimated_cap_hit=contract.get("cap_hit"))
                    else:
                        scraped += 1

            log.info(f"  Scraped: {scraped}  Estimated: {estimated}")

        # Retrain
        run_nightly_stats()

        elapsed = time.perf_counter() - t0
        s = db_stats()
        log.info(f"Weekly contracts done — DB: {s['real']} real / {s['estimated']} est  "
                 f"({elapsed:.0f}s)")

        _write_last_updated("weekly_contracts", {
            "players_checked": len(to_refresh),
            "db_real":         s["real"],
            "db_estimated":    s["estimated"],
        })

    except Exception as exc:
        log.error(f"Weekly contracts FAILED: {exc}\n{traceback.format_exc()}")
        _write_last_updated("weekly_contracts_error", {"error": str(exc)})


# ── Scheduler ──────────────────────────────────────────────────────────────────
def start_scheduler() -> None:
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        print("APScheduler not installed.  Run:  pip install apscheduler")
        sys.exit(1)

    scheduler = BlockingScheduler(timezone="UTC")

    # Nightly 11 PM PST = 07:00 UTC (PST = UTC-8)
    scheduler.add_job(
        run_nightly_stats,
        CronTrigger(hour=7, minute=0),
        id="nightly_stats",
        name="Nightly NHL stats refresh + retrain",
        misfire_grace_time=3600,
    )

    # Weekly Sunday midnight PST = Sunday 08:00 UTC
    scheduler.add_job(
        run_weekly_contracts,
        CronTrigger(day_of_week="sun", hour=8, minute=0),
        id="weekly_contracts",
        name="Weekly contract re-verification",
        misfire_grace_time=7200,
    )

    log.info("Scheduler started.")
    log.info("  Nightly stats:     07:00 UTC daily  (11 PM PST)")
    log.info("  Weekly contracts:  08:00 UTC Sunday  (midnight PST)")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        log.info("Scheduler stopped.")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NHL value model refresh scheduler")
    parser.add_argument("--run-stats",     action="store_true",
                        help="Run nightly stats refresh now")
    parser.add_argument("--run-contracts", action="store_true",
                        help="Run weekly contract check now")
    args = parser.parse_args()

    if args.run_stats:
        run_nightly_stats()
    elif args.run_contracts:
        run_weekly_contracts()
    else:
        start_scheduler()
