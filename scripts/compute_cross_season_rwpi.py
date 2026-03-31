"""
Compute cross-season RWPI by pooling all 4 seasons into one population
for percentile ranking, then rebuild ufa_training_set and refit curves.
"""
from __future__ import annotations
import sys, sqlite3, time, unicodedata, re
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))

from src.data.moneypuck import _fetch_and_cache, _extract_from_raw
from src.data.nhl_api import fetch_player_stats, _fetch_stats_rest_report
from src.models.rwpi import assign_roles, compute_branch_scores, compute_rwpi_score

DB = "data/historical_stats.db"
SEASON_LENGTH = 82
SIGNING_TO_STATS_SEASON = {2022: 20212022, 2023: 20222023, 2024: 20232024}
CAP_CEILINGS = {2022: 82_500_000, 2023: 83_500_000, 2024: 88_000_000}


def _project(val, gp, sl=82):
    if val is None or not gp or gp <= 0:
        return val
    if gp >= sl:
        return round(float(val), 2)
    return round(float(val) * sl / gp, 2)


def _build_rwpi_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "position" in df.columns and "pos" not in df.columns:
        df["pos"] = df["position"].fillna("F")
    elif "pos" not in df.columns:
        df["pos"] = "F"
    df["pos"] = df["pos"].fillna("F").astype(str)
    if "g60" not in df.columns:
        h82 = (df["toi_per_g"].fillna(0) * SEASON_LENGTH / 60).clip(lower=0.001)
        df["g60"] = (df["g"].fillna(0) / h82).round(4)
    return df


def _name_key(name: str) -> str:
    if not isinstance(name, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    return re.sub(r"\s+", " ",
                  re.sub(r"[^a-z ]", "",
                         nfkd.encode("ascii", "ignore").decode().lower())).strip()


# ---- Load historical seasons from DB -------------------------------------
print("Loading historical seasons from DB...")
conn = sqlite3.connect(DB)
hist_dfs = {}
for season_id in [20212022, 20222023, 20232024]:
    df = pd.read_sql(f"SELECT * FROM player_seasons WHERE season={season_id}", conn)
    df = _build_rwpi_df(df)
    hist_dfs[season_id] = df
    print(f"  {season_id}: {len(df)} players")
conn.close()

# ---- Load 20242025 from cached files -------------------------------------
print("\nLoading 20242025 from cached files (moneypuck_2024.csv)...")
season_id_cur = 20242025
raw = _fetch_and_cache(2024, force_refresh=False)
mp_df = _extract_from_raw(raw)

raw_all = raw[raw["situation"] == "all"].copy()
raw_all["sca_rel"] = (
    raw_all["onIce_xGoalsPercentage"] - raw_all["offIce_xGoalsPercentage"]
).round(4)
raw_slim = raw_all[["playerId", "name", "position", "sca_rel"]].drop_duplicates("playerId")
raw_slim = raw_slim.rename(columns={"playerId": "player_id",
                                     "name": "player_name"})
mp_df = mp_df.merge(raw_slim, on="player_id", how="left")

gp_arr = mp_df["gp_mp"].clip(lower=1)
for col in ["a1", "a2", "hdxg"]:
    mp_df[col] = (mp_df[col] * (SEASON_LENGTH / gp_arr)).round(2)

player_ids = mp_df["player_id"].tolist()
print(f"  MoneyPuck 2024: {len(mp_df)} players")

print(f"  Fetching NHL API stats ({len(player_ids)} players)...")
api_stats: dict = {}
for i, pid in enumerate(player_ids):
    data = fetch_player_stats(pid, [season_id_cur])
    stats = data.get(season_id_cur) or data.get(str(season_id_cur))
    if stats:
        api_stats[pid] = stats
    if i % 200 == 0 and i > 0:
        print(f"    {i}/{len(player_ids)}...")
    time.sleep(0.04)
print(f"  API: {len(api_stats)} players")

print("  Fetching supplemental stats...")
rt_rows  = _fetch_stats_rest_report("realtime",  season_id_cur)
toi_rows = _fetch_stats_rest_report("timeonice", season_id_cur)
rt_by  = {r["playerId"]: r for r in rt_rows}
toi_by = {r["playerId"]: r for r in toi_rows}
supp: dict = {}
for pid in set(rt_by) | set(toi_by):
    rt  = rt_by.get(pid, {})
    toi = toi_by.get(pid, {})
    gp  = rt.get("gamesPlayed") or toi.get("gamesPlayed") or 0
    supp[pid] = {
        "gp":     gp,
        "hits":   rt.get("hits", 0) or 0,
        "blocks": rt.get("blockedShots", 0) or 0,
        "pp_toi": round((toi.get("ppTimeOnIcePerGame") or 0) / 60, 4),
        "pk_toi": round((toi.get("shTimeOnIcePerGame") or 0) / 60, 4),
    }
print(f"  Supplemental: {len(supp)} players")

rows_cur = []
for _, mp_row in mp_df.iterrows():
    pid = int(mp_row["player_id"])
    gp_mp = int(mp_row["gp_mp"]) if pd.notna(mp_row.get("gp_mp")) else 0
    api  = api_stats.get(pid, {})
    s    = supp.get(pid, {})
    gp_f = s.get("gp") or api.get("gp") or gp_mp or 0
    rows_cur.append({
        "player_id":   pid,
        "player_name": mp_row.get("player_name"),
        "season":      season_id_cur,
        "gp":          gp_f,
        "g":           _project(api.get("g"),    gp_f),
        "a":           _project(api.get("a"),    gp_f),
        "a1":          mp_row.get("a1"),
        "a2":          mp_row.get("a2"),
        "toi_per_g":   api.get("toi_per_g") or mp_row.get("toi_per_g"),
        "pp_toi":      s.get("pp_toi"),
        "pk_toi":      s.get("pk_toi"),
        "xgf60":       mp_row.get("xgf60"),
        "xga60":       mp_row.get("xga60"),
        "oz_pct":      mp_row.get("oz_pct"),
        "fenwick_rel": mp_row.get("fenwick_rel"),
        "sca_rel":     mp_row.get("sca_rel"),
        "hdxg":        mp_row.get("hdxg"),
        "shots":       _project(api.get("shots"),  gp_f),
        "hits":        _project(s.get("hits", 0),  gp_f),
        "blocks":      _project(s.get("blocks", 0),gp_f),
        "shooting_pct": api.get("shooting_pct"),
        "pp_pts":      _project(api.get("pp_pts"), gp_f),
        "position":    mp_row.get("position"),
    })

df_2425 = pd.DataFrame(rows_cur)
df_2425 = _build_rwpi_df(df_2425)
print(f"  Built 20242025 DataFrame: {len(df_2425)} players")

# ---- Pool all 4 seasons ---------------------------------------------------
print("\nPooling all 4 seasons...")
combined = pd.concat(
    list(hist_dfs.values()) + [df_2425],
    ignore_index=True, sort=False,
)
print(f"  Combined pool: {len(combined)} player-seasons")
for s, n in combined.groupby("season").size().items():
    print(f"    {s}: {n}")

# ---- Compute cross-season RWPI -------------------------------------------
print("\nComputing cross-season RWPI on combined pool...")
combined = assign_roles(combined)
combined = compute_branch_scores(combined)
combined = compute_rwpi_score(combined)
scored = combined["rwpi_score"].notna().sum()
print(f"  {scored}/{len(combined)} player-seasons received RWPI scores")

# ---- Build lookup --------------------------------------------------------
combined["_nk"] = combined["player_name"].fillna("").apply(_name_key)
lookup: dict = {}
for _, row in combined.iterrows():
    if pd.notna(row.get("rwpi_score")):
        key = (row["_nk"], int(row["season"]))
        lookup[key] = {
            "rwpi_score": float(row["rwpi_score"]),
            "role":       row.get("rwpi_role", "Unknown"),
        }
print(f"  Lookup: {len(lookup)} entries")

# ---- Rebuild ufa_training_set -------------------------------------------
print("\nRebuilding ufa_training_set with cross-season RWPI...")
conn = sqlite3.connect(DB)
signings = pd.read_sql("SELECT * FROM ufa_signings", conn)
conn.close()

training_rows: list[dict] = []
unmatched: dict[int, list] = {2022: [], 2023: [], 2024: []}

for _, s in signings.iterrows():
    sy = int(s["signing_year"])
    stats_season = SIGNING_TO_STATS_SEASON.get(sy)
    if not stats_season:
        continue
    nk = _name_key(str(s["player_name"]))
    match = lookup.get((nk, stats_season))
    if not match:
        tokens = set(nk.split())
        for (kn, ks), v in lookup.items():
            if ks == stats_season and len(tokens & set(kn.split())) >= 2:
                match = v
                break
    if not match:
        unmatched[sy].append(str(s["player_name"]))
        continue
    ceil = float(CAP_CEILINGS[sy])
    ch   = float(s["cap_hit"])
    training_rows.append({
        "player_name":  str(s["player_name"]),
        "signing_year": sy,
        "rwpi_score":   round(match["rwpi_score"], 4),
        "cap_hit":      ch,
        "cap_pct":      round(ch / ceil, 6),
        "cap_ceiling":  ceil,
        "player_age":   s.get("player_age"),
        "role":         match["role"],
    })

ts = pd.DataFrame(training_rows)
print(f"  Training rows: {len(ts)}")
for yr in [2022, 2023, 2024]:
    n_m = len(ts[ts["signing_year"] == yr])
    n_u = len(unmatched[yr])
    print(f"    {yr}: {n_m} matched, {n_u} unmatched")

conn = sqlite3.connect(DB)
conn.execute("DELETE FROM ufa_training_set")
for _, row in ts.iterrows():
    conn.execute("""
        INSERT INTO ufa_training_set
        (player_name, signing_year, rwpi_score, cap_hit, cap_pct,
         cap_ceiling, player_age, role)
        VALUES (:player_name, :signing_year, :rwpi_score, :cap_hit,
                :cap_pct, :cap_ceiling, :player_age, :role)
    """, row.to_dict())
conn.commit()
conn.close()
print(f"  Stored {len(training_rows)} rows")

# ---- Correlations & quintile trends --------------------------------------
print("\n" + "=" * 65)
print("SPEARMAN + QUINTILE BREAKDOWN BY ROLE")
print("=" * 65)

from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.model_selection import cross_val_score
from sklearn.isotonic import IsotonicRegression

CAP_CURRENT = 88_000_000
ROLES_ORDER = [
    "Offensive Driver F",
    "Defensive / PK Forward",
    "Depth / Bottom-6 F",
    "Shutdown Defenseman",
    "Depth / 7th Defenseman",
    "Offensive Defenseman",
]

curve_results = {}

for role in ROLES_ORDER:
    sub = ts[ts["role"] == role].copy()
    n = len(sub)
    if n == 0:
        print(f"\n{role}: NO DATA")
        continue

    X = sub["rwpi_score"].values.reshape(-1, 1)
    y = sub["cap_pct"].values

    rho, p_val = spearmanr(sub["rwpi_score"], sub["cap_pct"])

    # Quintile trend
    sub["q"] = pd.qcut(sub["rwpi_score"], q=5, labels=False, duplicates="drop")
    qtbl = sub.groupby("q", observed=True).agg(
        q_n=("cap_hit","count"),
        q_rwpi_min=("rwpi_score","min"),
        q_rwpi_max=("rwpi_score","max"),
        q_avg_cap=("cap_hit","mean"),
    ).round(0)

    # Try polynomial
    poly_pipe = SKPipeline([
        ("poly",  PolynomialFeatures(degree=2, include_bias=True)),
        ("lm",    LinearRegression()),
    ])
    if n >= 10:
        poly_cv = cross_val_score(poly_pipe, X, y, cv=min(5, n//5), scoring="r2")
        poly_r2 = float(poly_cv.mean())
    else:
        poly_r2 = -99.0

    # Try isotonic
    iso = IsotonicRegression(out_of_bounds="clip")
    if n >= 10:
        iso_cv = cross_val_score(iso, X.ravel(), y, cv=min(5, n//5), scoring="r2")
        iso_r2 = float(iso_cv.mean())
    else:
        iso_r2 = -99.0

    # Select method
    if poly_r2 >= 0.20:
        method, cv_r2 = "polynomial", poly_r2
        poly_pipe.fit(X, y)
        model = ("poly", poly_pipe)
    elif iso_r2 >= 0.20:
        method, cv_r2 = "isotonic", iso_r2
        iso.fit(X.ravel(), y)
        model = ("iso", iso)
    else:
        method, cv_r2 = "decile_lookup", max(poly_r2, iso_r2)
        sub2 = sub.copy()
        sub2["dec"] = pd.qcut(sub2["rwpi_score"], q=10, labels=False, duplicates="drop")
        dec_tbl = sub2.groupby("dec", observed=True)["cap_pct"].median().reset_index()
        model = ("decile", dec_tbl, sub2)

    # Predict at benchmarks
    preds = {}
    for rv in [10, 25, 50, 75, 90]:
        xq = np.array([[rv]])
        if model[0] == "poly":
            p = float(np.clip(model[1].predict(xq)[0], 0.005, 0.20))
        elif model[0] == "iso":
            p = float(np.clip(model[1].predict([rv])[0], 0.005, 0.20))
        else:
            sub_s = model[2].sort_values("rwpi_score")
            sub_s["dec"] = pd.qcut(sub_s["rwpi_score"], q=10, labels=False, duplicates="drop")
            bins = sub_s.groupby("dec", observed=True)["rwpi_score"].max()
            d = bins.index[-1]
            for dec_id, max_v in sorted(bins.items()):
                if rv <= max_v:
                    d = dec_id
                    break
            tbl = model[1]
            rows_d = tbl.loc[tbl["dec"] == d, "cap_pct"]
            med = float(rows_d.values[0]) if len(rows_d) else float(tbl["cap_pct"].median())
            p = float(np.clip(med, 0.005, 0.20))
        preds[rv] = p

    curve_results[role] = {
        "n": n, "rho": rho, "poly_r2": poly_r2, "iso_r2": iso_r2,
        "method": method, "cv_r2": cv_r2, "preds": preds, "qtbl": qtbl,
    }

    flag = ""
    if cv_r2 < 0.20:
        flag = "  *** POOR FIT"

    print(f"\n{'─'*60}")
    print(f"{role}  (n={n}){flag}")
    print(f"  Spearman rho={rho:+.3f}  p={p_val:.3f}")
    print(f"  Poly CV R²={poly_r2:.4f}   Isotonic CV R²={iso_r2:.4f}   Method={method}")
    print(f"  Quintile salary trend (avg cap hit):")
    for qi, qrow in qtbl.iterrows():
        print(f"    Q{int(qi)}: RWPI {int(qrow['q_rwpi_min'])}-{int(qrow['q_rwpi_max'])}"
              f"  n={int(qrow['q_n'])}  avg=${qrow['q_avg_cap']:>10,.0f}")
    print(f"  Predicted salary (${CAP_CURRENT/1e6:.0f}M ceiling):")
    print(f"    {'RWPI':>5}  {'cap%':>7}  {'salary':>12}")
    for rv, p in preds.items():
        print(f"    {rv:>5}  {p:>7.3%}  ${p*CAP_CURRENT:>11,.0f}")

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"{'Role':<30} {'n':>4}  {'Spearman':>9}  {'Method':<16}  {'CV R2':>7}")
print("-" * 75)
for role in ROLES_ORDER:
    r = curve_results.get(role)
    if not r:
        continue
    flag = " ***" if r["cv_r2"] < 0.20 else ""
    print(f"{role:<30} {r['n']:>4}  {r['rho']:>+9.3f}  {r['method']:<16}  {r['cv_r2']:>7.4f}{flag}")
