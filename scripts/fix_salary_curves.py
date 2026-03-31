"""
Role-specific salary curve fitting — final approach.
Pure numpy/pandas only — no scipy, no sklearn, no OpenBLAS dependency.

Method per role:
  Offensive Driver F    : polynomial degree-2 (numpy polyfit, k-fold CV)
  All other 5 roles     : three-tier median with role-specific percentile boundaries
                          (Tier 1: min-p33, Tier 2: p33-p66, Tier 3: p66-max)

Contamination filter:
  Remove non-arm's-length signings: cap_hit < $2.5M AND rwpi_score > 60
"""
from __future__ import annotations
import sys, sqlite3, json, pathlib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB          = "data/historical_stats.db"
CAP_CURRENT = 88_000_000

POLY_ROLES = ["Offensive Driver F"]
TIER_ROLES = [
    "Defensive / PK Forward",
    "Depth / Bottom-6 F",
    "Shutdown Defenseman",
    "Depth / 7th Defenseman",
    "Offensive Defenseman",
]
ALL_ROLES = POLY_ROLES + TIER_ROLES


# ── Pure-numpy helpers ─────────────────────────────────────────────────────────

def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation without scipy."""
    n = len(x)
    rx = np.argsort(np.argsort(x)).astype(float) + 1
    ry = np.argsort(np.argsort(y)).astype(float) + 1
    d2 = np.sum((rx - ry) ** 2)
    return float(1 - 6 * d2 / (n * (n**2 - 1)))


def poly_cv_r2(x: np.ndarray, y: np.ndarray, deg: int = 2, k: int = 5) -> float:
    """K-fold CV R² for numpy polyfit."""
    n = len(x)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    r2s = []
    for i in range(k):
        val   = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        if len(train) <= deg:
            continue
        coeffs = np.polyfit(x[train], y[train], deg)
        yp = np.polyval(coeffs, x[val])
        ss_res = np.sum((y[val] - yp) ** 2)
        ss_tot = np.sum((y[val] - y[val].mean()) ** 2)
        r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    return float(np.mean(r2s)) if r2s else 0.0


def poly_predict(coeffs: np.ndarray, rwpi: float, cap_ceiling: float) -> float:
    pct = float(np.clip(np.polyval(coeffs, rwpi), 0.005, 0.20))
    return pct * cap_ceiling


# ── Load training set ──────────────────────────────────────────────────────────
conn = sqlite3.connect(DB)
ts_orig = pd.read_sql(
    "SELECT player_name, signing_year, rwpi_score, cap_hit, cap_pct, role "
    "FROM ufa_training_set WHERE rwpi_score IS NOT NULL",
    conn,
)
conn.close()
print(f"Loaded {len(ts_orig)} rows from ufa_training_set")

# ── Contamination filter ───────────────────────────────────────────────────────
mask_bad  = (ts_orig["cap_hit"] < 2_500_000) & (ts_orig["rwpi_score"] > 60)
ts_clean  = ts_orig[~mask_bad].copy()
removed   = mask_bad.sum()

print("=" * 65)
print(f"Contamination filter: removed {removed} rows  "
      f"({len(ts_orig)} -> {len(ts_clean)})")
print("=" * 65)

# ── Polynomial fit — Offensive Driver F ───────────────────────────────────────
poly_params: dict[str, np.ndarray] = {}

print("\n--- Offensive Driver F: polynomial degree-2 ---")
for role in POLY_ROLES:
    sub  = ts_clean[ts_clean["role"] == role]
    n    = len(sub)
    x    = sub["rwpi_score"].values.astype(float)
    y    = sub["cap_pct"].values.astype(float)
    rho  = spearman_rho(x, y)
    cv   = poly_cv_r2(x, y, deg=2, k=5)
    coeffs = np.polyfit(x, y, 2)
    poly_params[role] = coeffs
    flag = "  PASS" if cv >= 0.20 else "  *** POOR FIT"
    print(f"  {role}  (n={n}){flag}")
    print(f"    Spearman={rho:+.3f}   CV R2={cv:.4f}")
    print(f"    a={coeffs[0]:.7f}  b={coeffs[1]:.6f}  c={coeffs[2]:.6f}")

# ── Three-tier median — all other roles ───────────────────────────────────────
tier_params: dict[str, dict] = {}

print("\n--- Other roles: three-tier median (role-specific boundaries) ---")
for role in TIER_ROLES:
    sub  = ts_clean[ts_clean["role"] == role]
    n    = len(sub)
    x    = sub["rwpi_score"].values.astype(float)
    y_ch = sub["cap_hit"].values.astype(float)
    y_cp = sub["cap_pct"].values.astype(float)
    rho  = spearman_rho(x, y_cp)

    p33 = float(np.percentile(x, 33.3))
    p66 = float(np.percentile(x, 66.7))

    mask1 = x <= p33
    mask2 = (x > p33) & (x <= p66)
    mask3 = x > p66
    overall_med = float(np.median(y_ch))

    med1 = float(np.median(y_ch[mask1])) if mask1.sum() > 0 else overall_med
    med2 = float(np.median(y_ch[mask2])) if mask2.sum() > 0 else overall_med
    med3 = float(np.median(y_ch[mask3])) if mask3.sum() > 0 else overall_med

    tier_params[role] = {
        "p33": p33, "p66": p66,
        "t1":  med1, "t2": med2, "t3": med3,
        "n1":  int(mask1.sum()), "n2": int(mask2.sum()), "n3": int(mask3.sum()),
        "rwpi_min": float(x.min()), "rwpi_max": float(x.max()),
    }

    print(f"\n  {role}  (n={n})  Spearman={rho:+.3f}")
    print(f"    Boundaries: p33={p33:.1f}  p66={p66:.1f}")
    print(f"    Tier 1: RWPI {x.min():.0f} - {p33:.1f}   "
          f"n={mask1.sum():>3}  median=${med1:>9,.0f}")
    print(f"    Tier 2: RWPI {p33:.1f} - {p66:.1f}   "
          f"n={mask2.sum():>3}  median=${med2:>9,.0f}")
    print(f"    Tier 3: RWPI {p66:.1f} - {x.max():.0f}   "
          f"n={mask3.sum():>3}  median=${med3:>9,.0f}")
    for lbl, cnt in [("Tier 1", mask1.sum()), ("Tier 2", mask2.sum()),
                     ("Tier 3", mask3.sum())]:
        if cnt == 0:
            print(f"    *** WARNING: {lbl} EMPTY — overall median fallback")


# ── Predict helper ─────────────────────────────────────────────────────────────
def predict_salary(role: str, rwpi: float) -> float:
    if role in poly_params:
        return poly_predict(poly_params[role], rwpi, CAP_CURRENT)
    t = tier_params[role]
    if rwpi <= t["p33"]:
        return t["t1"]
    elif rwpi <= t["p66"]:
        return t["t2"]
    else:
        return t["t3"]


# ── Full role salary tables ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("FULL SALARY TABLES BY ROLE")
print("=" * 65)

for role in ALL_ROLES:
    sub = ts_clean[ts_clean["role"] == role]
    print(f"\n  {role}  (n={len(sub)})")
    if role in poly_params:
        print(f"  Method: polynomial degree-2")
        print(f"  {'RWPI':>5}  {'Predicted':>12}")
        for rv in [10, 25, 50, 75, 90]:
            print(f"  {rv:>5}  ${predict_salary(role, rv):>11,.0f}")
    else:
        t = tier_params[role]
        print(f"  Method: three-tier median")
        print(f"  {'Tier':<8} {'RWPI range':<24} {'n':>4}  {'Median':>12}")
        print(f"  {'Tier 1':<8} {t['rwpi_min']:.0f} to {t['p33']:.1f}{'':>16}"
              f"{t['n1']:>4}  ${t['t1']:>11,.0f}")
        print(f"  {'Tier 2':<8} {t['p33']:.1f} to {t['p66']:.1f}{'':>16}"
              f"{t['n2']:>4}  ${t['t2']:>11,.0f}")
        print(f"  {'Tier 3':<8} {t['p66']:.1f} to {t['rwpi_max']:.0f}{'':>16}"
              f"{t['n3']:>4}  ${t['t3']:>11,.0f}")

# ── Generic RWPI-50 per role ───────────────────────────────────────────────────
print("\n" + "=" * 65)
print("GENERIC RWPI-50 — ALL ROLES")
print("=" * 65)
for role in ALL_ROLES:
    print(f"  {role:<35}  RWPI=50  ${predict_salary(role, 50.0):>10,.0f}")

# ── Named spot checks ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("NAMED SPOT CHECKS")
print("=" * 65)

named_spots = [
    ("Connor McDavid",     "Offensive Driver F",   99.0,  12_500_000),
    ("Jaccob Slavin",      "Shutdown Defenseman",   40.1,   7_000_000),
    ("Macklin Celebrini",  "Offensive Driver F",    78.0,     975_000),
    ("Kirill Kaprizov",    "Offensive Driver F",    92.0,   9_000_000),
    ("Anze Kopitar",       "Offensive Driver F",    65.0,  10_000_000),
]

print(f"\n  {'Player':<22} {'Role':<28} {'RWPI':>5}  "
      f"{'Actual':>12}  {'Predicted':>12}  {'Gap':>13}")
print("  " + "-" * 97)
for label, role, rwpi, actual in named_spots:
    pred = predict_salary(role, rwpi)
    gap  = pred - actual
    sign = "+" if gap >= 0 else ""
    print(f"  {label:<22} {role:<28} {rwpi:>5.1f}  "
          f"${actual:>11,.0f}  ${pred:>11,.0f}  {sign}${abs(gap):>10,.0f}"
          f"  ({'underpaid' if gap > 0 else 'overpaid'})")

# ── Empty tier confirmation ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("EMPTY TIER CHECK")
print("=" * 65)
all_ok = True
for role, t in tier_params.items():
    for lbl, cnt in [("Tier 1", t["n1"]), ("Tier 2", t["n2"]), ("Tier 3", t["n3"])]:
        if cnt == 0:
            print(f"  EMPTY: {role} - {lbl}")
            all_ok = False
print("  All tiers populated." if all_ok else "  Fix above empty tiers before applying.")

# ── Save curves ────────────────────────────────────────────────────────────────
pathlib.Path("data/processed").mkdir(parents=True, exist_ok=True)
curves_path = "data/processed/salary_curves.json"

curves_out = {
    "cap_current": CAP_CURRENT,
    "polynomial": {
        role: {"coef_a": float(c[0]), "coef_b": float(c[1]), "coef_c": float(c[2])}
        for role, c in poly_params.items()
    },
    "three_tier": {
        role: {"p33": t["p33"], "p66": t["p66"],
               "t1": t["t1"], "t2": t["t2"], "t3": t["t3"]}
        for role, t in tier_params.items()
    },
}

with open(curves_path, "w") as f:
    json.dump(curves_out, f, indent=2)

print(f"\nCurves saved -> {curves_path}")
print("Run scripts/apply_salary_curves.py to apply to all players.")
