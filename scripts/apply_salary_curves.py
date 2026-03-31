"""
Apply role-specific salary curves to all current-season players.
Pure numpy/pandas — no sklearn, no OpenBLAS dependency.
Loads salary_curves.json from fix_salary_curves.py.
Outputs: data/processed/rwpi_values.csv
"""
from __future__ import annotations
import sys, json, pathlib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

# Import only the branch-score functions from rwpi — these are pure pandas/numpy
# (assign_roles uses sklearn KMeans and is replaced below)
from src.models.rwpi import (
    compute_branch_scores, compute_rwpi_score,
    ROLE_NAMES, ROLE_WEIGHTS, MIN_GP,
    _CLUSTER_FEATURES,
)

# ── Pure-numpy role assignment (replaces sklearn KMeans + StandardScaler) ──────

def _numpy_kmeans(X: np.ndarray, n_clusters: int = 3, n_init: int = 20,
                  max_iter: int = 300, random_state: int = 42) -> np.ndarray:
    """K-means in pure numpy — no BLAS matrix multiply."""
    rng = np.random.default_rng(random_state)
    best_labels = None
    best_inertia = np.inf
    n = len(X)

    for _ in range(n_init):
        idx = rng.choice(n, n_clusters, replace=False)
        centers = X[idx].copy()

        for _ in range(max_iter):
            # Squared Euclidean: (n, k, f) → sum over f → (n, k)
            diffs   = X[:, None, :] - centers[None, :, :]
            dists   = (diffs ** 2).sum(axis=2)
            labels  = dists.argmin(axis=1)

            new_centers = np.array([
                X[labels == k].mean(axis=0) if (labels == k).any() else centers[k]
                for k in range(n_clusters)
            ])
            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers

        inertia = sum(
            ((X[labels == k] - centers[k]) ** 2).sum()
            for k in range(n_clusters)
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels  = labels.copy()

    return best_labels


def _remap_labels_numpy(X_raw: np.ndarray, labels: np.ndarray,
                        features: list[str]) -> np.ndarray:
    """Map raw cluster IDs to: 2=offensive(high PP TOI), 1=defensive(high PK TOI), 0=depth."""
    df_tmp = pd.DataFrame(X_raw, columns=features)
    df_tmp["label"] = labels
    means = df_tmp.groupby("label")[["pp_toi", "pk_toi"]].mean()

    off       = int(means["pp_toi"].idxmax())
    remaining = means.drop(index=off)
    def_      = int(remaining["pk_toi"].idxmax())
    depth     = [c for c in means.index if c not in (off, def_)][0]

    remap = {off: 2, def_: 1, depth: 0}
    return np.array([remap[lbl] for lbl in labels])


def assign_roles_numpy(df: pd.DataFrame) -> pd.DataFrame:
    """Pure-numpy equivalent of rwpi.assign_roles (no sklearn)."""
    df = df.copy()
    df["rwpi_role_id"] = -1
    df["rwpi_role"]    = "Unknown"

    gp_ok   = df["gp"].fillna(0) >= MIN_GP
    feat_ok = df[_CLUSTER_FEATURES].notna().all(axis=1)

    for is_defense, offset in [(False, 0), (True, 3)]:
        grp = (df["pos"] == "D") if is_defense else (df["pos"] != "D")
        idx = df.index[grp & feat_ok & gp_ok]
        if len(idx) < 3:
            continue

        X_raw = df.loc[idx, _CLUSTER_FEATURES].values.astype(float)

        # StandardScaler equivalent
        mean = X_raw.mean(axis=0)
        std  = X_raw.std(axis=0, ddof=0)
        std[std == 0] = 1.0
        X_s = (X_raw - mean) / std

        raw_labels = _numpy_kmeans(X_s, n_clusters=3, n_init=20, random_state=42)
        labels     = _remap_labels_numpy(X_raw, raw_labels, _CLUSTER_FEATURES)

        df.loc[idx, "rwpi_role_id"] = labels + offset

    df["rwpi_role"] = df["rwpi_role_id"].map(ROLE_NAMES).fillna("Unknown")
    return df


# ── Load player data ───────────────────────────────────────────────────────────
print("Loading current-season player data...")

pred = pd.read_csv("data/processed/predictions.csv")
pred["player_id"] = pred["player_id"].astype(str)
print(f"  predictions.csv: {len(pred)} players")

with open("data/raw/supplemental_stats_cache.json") as f:
    supp_raw = json.load(f)
supp_rows = [
    {"player_id": pid, **vals}
    for pid, vals in supp_raw["seasons"].get("20252026", {}).items()
]
supp = pd.DataFrame(supp_rows)
supp["player_id"] = supp["player_id"].astype(str)
print(f"  supplemental cache: {len(supp)} players")

mp_raw = pd.read_csv("data/raw/moneypuck_2025.csv", low_memory=False)
mp_all = mp_raw[mp_raw["situation"] == "all"].copy()
mp_all["player_id"] = mp_all["playerId"].astype(str)

ict   = mp_all["icetime"].clip(lower=1)
hours = ict / 3600
mp_all["xgf60"]     = (mp_all["I_F_xGoals"] / hours).round(4)
mp_all["sca_rel"]   = (
    mp_all["onIce_xGoalsPercentage"] - mp_all["offIce_xGoalsPercentage"]
).round(4)
mp_all["gp_mp"]  = mp_all["games_played"].clip(lower=1)
scale = 82 / mp_all["gp_mp"]
for src, dst in [
    ("I_F_primaryAssists",   "a1"),
    ("I_F_secondaryAssists", "a2"),
    ("I_F_highDangerxGoals", "hdxg"),
]:
    mp_all[dst] = (mp_all[src] * scale).round(2)
oz  = mp_all["I_F_oZoneShiftStarts"].fillna(0)
shf = mp_all["I_F_shifts"].clip(lower=1)
mp_all["oz_pct"] = (oz / shf).round(4)
mp_all.loc[mp_all["I_F_shifts"].fillna(0) == 0, "oz_pct"] = np.nan
mp_slim = mp_all[[
    "player_id", "xgf60", "a1", "a2", "hdxg", "oz_pct", "sca_rel",
]]
print(f"  moneypuck_2025.csv: {len(mp_slim)} players")

df = pred.merge(
    supp[["player_id", "hits", "blocks", "pp_toi", "pk_toi"]],
    on="player_id", how="left",
)
df = df.merge(mp_slim, on="player_id", how="left")
print(f"  merged: {len(df)} players total")

# ── Compute RWPI ───────────────────────────────────────────────────────────────
print("\nComputing RWPI...")
df = assign_roles_numpy(df)
df = compute_branch_scores(df)
df = compute_rwpi_score(df)

elig = df["rwpi_score"].notna()
print(f"  {elig.sum()} players received RWPI scores")

role_counts = df[elig]["rwpi_role"].value_counts()
for role, cnt in role_counts.items():
    print(f"    {role:<35} {cnt:>4}")

# ── Load salary curves ─────────────────────────────────────────────────────────
with open("data/processed/salary_curves.json") as f:
    curves = json.load(f)
cap_current = curves["cap_current"]
print(f"\nSalary curves loaded (cap_current=${cap_current:,.0f})")


def predict_true_value(role: str, rwpi: float) -> float | None:
    if role in curves["polynomial"]:
        p   = curves["polynomial"][role]
        pct = float(np.clip(p["coef_a"] * rwpi**2 + p["coef_b"] * rwpi + p["coef_c"],
                            0.005, 0.20))
        return pct * cap_current
    if role in curves["three_tier"]:
        t = curves["three_tier"][role]
        if rwpi <= t["p33"]:
            return t["t1"]
        elif rwpi <= t["p66"]:
            return t["t2"]
        else:
            return t["t3"]
    return None


# ── Apply curves ───────────────────────────────────────────────────────────────
print("Applying salary curves...")
df["true_value"] = df.apply(
    lambda r: predict_true_value(r["rwpi_role"], r["rwpi_score"])
    if pd.notna(r.get("rwpi_score")) else None,
    axis=1,
)
df["value_gap"] = df.apply(
    lambda r: r["true_value"] - r["cap_hit"]
    if pd.notna(r.get("true_value")) and pd.notna(r.get("cap_hit")) else None,
    axis=1,
)

# ── Save output ────────────────────────────────────────────────────────────────
out_cols = [
    "name", "team", "pos", "age",
    "rwpi_role", "rwpi_off_score", "rwpi_def_score", "rwpi_score",
    "cap_hit", "true_value", "value_gap",
    "expiry_status", "expiry_year", "years_left",
    "gp", "player_id",
]
out = df[[c for c in out_cols if c in df.columns]].copy()
out.to_csv("data/processed/rwpi_values.csv", index=False)
print(f"\nSaved -> data/processed/rwpi_values.csv  ({len(out)} rows)")

# ── Named player checks ────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("NAMED PLAYER SPOT CHECKS")
print("=" * 72)
checks = [
    "Connor McDavid", "Jaccob Slavin", "Macklin Celebrini",
    "Kirill Kaprizov", "Anze Kopitar",
]
print(f"\n  {'Player':<22} {'Role':<28} {'RWPI':>5}  "
      f"{'Cap Hit':>10}  {'True Val':>10}  {'Gap':>14}")
print("  " + "-" * 98)
for name in checks:
    rows = out[out["name"] == name]
    if rows.empty:
        print(f"  {name:<22}  NOT FOUND")
        continue
    r = rows.iloc[0]
    rwpi_s  = f"{r['rwpi_score']:.1f}" if pd.notna(r.get("rwpi_score")) else "N/A"
    cap_s   = f"${r['cap_hit']:>9,.0f}" if pd.notna(r.get("cap_hit")) else "N/A"
    tv_s    = f"${r['true_value']:>9,.0f}" if pd.notna(r.get("true_value")) else "N/A"
    if pd.notna(r.get("value_gap")):
        g = r["value_gap"]
        gap_s = f"+${g:>9,.0f}  underpaid" if g > 0 else f"-${abs(g):>9,.0f}  overpaid"
    else:
        gap_s = "N/A"
    print(f"  {name:<22} {r['rwpi_role']:<28} {rwpi_s:>5}  "
          f"{cap_s:>10}  {tv_s:>10}  {gap_s}")

# ── Top 15 underpaid / overpaid ────────────────────────────────────────────────
has_gap = out[out["value_gap"].notna()].copy()

print("\n" + "=" * 72)
print("TOP 15 UNDERPAID  (largest positive value_gap)")
print("=" * 72)
top_under = has_gap[has_gap["value_gap"] > 0].nlargest(15, "value_gap")
for _, r in top_under.iterrows():
    print(f"  {r['name']:<22} {r['rwpi_role']:<28} RWPI={r['rwpi_score']:>4.0f}  "
          f"cap=${r['cap_hit']:>8,.0f}  true=${r['true_value']:>8,.0f}  "
          f"gap=+${r['value_gap']:>8,.0f}")

print("\n" + "=" * 72)
print("TOP 15 OVERPAID  (largest negative value_gap)")
print("=" * 72)
top_over = has_gap[has_gap["value_gap"] < 0].nsmallest(15, "value_gap")
for _, r in top_over.iterrows():
    print(f"  {r['name']:<22} {r['rwpi_role']:<28} RWPI={r['rwpi_score']:>4.0f}  "
          f"cap=${r['cap_hit']:>8,.0f}  true=${r['true_value']:>8,.0f}  "
          f"gap=-${abs(r['value_gap']):>8,.0f}")

# ── Role summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("VALUE GAP SUMMARY BY ROLE")
print("=" * 72)
rs = (
    has_gap.groupby("rwpi_role")
    .agg(n=("value_gap", "count"),
         median_gap=("value_gap", "median"),
         pct_under=("value_gap", lambda x: f"{(x > 0).mean()*100:.0f}%"))
    .sort_values("median_gap", ascending=False)
)
print(rs.to_string())
print("\nDone.")
