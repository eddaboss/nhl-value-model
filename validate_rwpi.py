"""RWPI Steps 2-5 validation script."""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, 'src')
from models.rwpi import (
    assign_roles, compute_branch_scores, compute_rwpi_score,
    fit_salary_curve, ROLE_NAMES, MIN_GP,
    _build_off_features, _build_def_features, _pct_rank, _toi_hours_82,
)

# ── Load and merge ─────────────────────────────────────────────────────────────
pred = pd.read_csv('data/processed/predictions.csv')
pred['player_id'] = pred['player_id'].astype(str)

with open('data/raw/supplemental_stats_cache.json') as f:
    supp_raw = json.load(f)
supp_rows = [
    {'player_id': pid, **vals}
    for pid, vals in supp_raw['seasons'].get('20252026', {}).items()
]
supp = pd.DataFrame(supp_rows)
supp['player_id'] = supp['player_id'].astype(str)

mp_raw = pd.read_csv('data/raw/moneypuck_2025.csv', low_memory=False)
mp_all = mp_raw[mp_raw['situation'] == 'all'].copy()
mp_all['player_id'] = mp_all['playerId'].astype(str)
ict = mp_all['icetime'].clip(lower=1)
hours = ict / 3600
mp_all['fenwick_rel'] = (
    mp_all['onIce_fenwickPercentage'] - mp_all['offIce_fenwickPercentage']
).round(4)
mp_all['xga60'] = (mp_all['OnIce_A_xGoals'] / hours).round(4)
mp_all['xgf60'] = (mp_all['I_F_xGoals'] / hours).round(4)
mp_all['sca_rel'] = (
    mp_all['onIce_xGoalsPercentage'] - mp_all['offIce_xGoalsPercentage']
).round(4)
mp_all['gp_mp'] = mp_all['games_played'].clip(lower=1)
scale = 82 / mp_all['gp_mp']
for col_src, col_dst in [
    ('I_F_primaryAssists', 'a1'),
    ('I_F_secondaryAssists', 'a2'),
    ('I_F_highDangerxGoals', 'hdxg'),
]:
    mp_all[col_dst] = (mp_all[col_src] * scale).round(2)
oz_starts = mp_all['I_F_oZoneShiftStarts'].fillna(0)
total_shifts = mp_all['I_F_shifts'].clip(lower=1)
mp_all['oz_pct'] = (oz_starts / total_shifts).round(4)
mp_all.loc[mp_all['I_F_shifts'].fillna(0) == 0, 'oz_pct'] = np.nan
mp_slim = mp_all[['player_id', 'fenwick_rel', 'xga60', 'xgf60', 'a1', 'a2', 'hdxg', 'oz_pct', 'sca_rel']]

df = pred.merge(supp[['player_id', 'hits', 'blocks', 'pp_toi', 'pk_toi']], on='player_id', how='left')
df = df.merge(mp_slim, on='player_id', how='left')

# ── Steps 1-4 ─────────────────────────────────────────────────────────────────
df = assign_roles(df)
df = compute_branch_scores(df)
df = compute_rwpi_score(df)

elig = df[df['rwpi_score'].notna()].copy()
off_feats = _build_off_features(elig)
def_feats = _build_def_features(elig)
off_pct = off_feats.apply(_pct_rank)
def_pct = def_feats.apply(_pct_rank)

# ══════════════════════════════════════════════════════════════════════════════
# AMADIO FULL DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════════
print('=== MICHAEL AMADIO FULL DIAGNOSTIC ===')
amadio = elig[elig['name'] == 'Michael Amadio']
if not amadio.empty:
    idx = amadio.index[0]
    r = amadio.iloc[0]

    print(f"Role: {r['rwpi_role']}  |  GP: {r['gp']}  |  TOI/G: {r['toi_per_g']:.2f}  |  Cap: ${r['cap_hit']:,.0f}")
    print(f"off_score={r['rwpi_off_score']:.1f}  def_score={r['rwpi_def_score']:.1f}  rwpi={r['rwpi_score']:.1f}")
    print()
    print('DEFENSIVE BRANCH:')
    print(f"  pk_toi_pct  : {def_feats.loc[idx,'pk_toi_pct']:+.4f}  -> p{def_pct.loc[idx,'pk_toi_pct']:.1f}  (wt=0.80)  [pk_toi/toi_per_g]")
    print(f"  pk_toi      : {def_feats.loc[idx,'pk_toi']:+.4f}  -> p{def_pct.loc[idx,'pk_toi']:.1f}  (wt=0.70)  [per-game PK min]")
    print(f"  dz_pct      : {def_feats.loc[idx,'dz_pct']:+.4f}  -> p{def_pct.loc[idx,'dz_pct']:.1f}  (wt=0.60)  [raw oz_pct={r['oz_pct']:.4f}]")
    print(f"  sca_rel     : {def_feats.loc[idx,'sca_rel']:+.4f}  -> p{def_pct.loc[idx,'sca_rel']:.1f}  (wt=0.50)  [on-off xGoals%]")
    print(f"  blocks60    : {def_feats.loc[idx,'blocks60']:+.4f}  -> p{def_pct.loc[idx,'blocks60']:.1f}  (wt=0.35)  [raw blocks={r['blocks']}]")
    print(f"  hits60      : {def_feats.loc[idx,'hits60']:+.4f}  -> p{def_pct.loc[idx,'hits60']:.1f}  (wt=0.25)  [raw hits={r['hits']}]")
    print()

    pk_sorted = elig[['name', 'pos', 'rwpi_role', 'pk_toi', 'gp']].sort_values('pk_toi', ascending=False).head(20)
    print('TOP 20 by pk_toi (league context):')
    print(pk_sorted.to_string(index=False))
else:
    print('  NOT FOUND')

print()
print('=' * 60)
print()

# ══════════════════════════════════════════════════════════════════════════════
# SALARY CURVE
# ══════════════════════════════════════════════════════════════════════════════
CAP_CEILING = 95_500_000
model, info = fit_salary_curve(df, CAP_CEILING)

print('=== SALARY CURVE VALIDATION ===')
print(f"Model type        : {info['model_type']}")
if 'poly_cv_r2_mean' in info:
    print(f"Poly CV R2 (rejected): {info['poly_cv_r2_mean']:.4f} +/- {info['poly_cv_r2_std']:.4f}")
print(f"Training set size : {info['n_train']}")
print(f"CV R2 scores      : {[round(x, 4) for x in info['cv_r2_scores']]}")
print(f"CV R2 mean +/- std: {info['cv_r2_mean']:.4f} +/- {info['cv_r2_std']:.4f}")
print(f"Train R2          : {info['train_r2']:.4f}")
if 'coef' in info:
    print(f"Poly intercept    : {info['intercept']:.6f}")
    print(f"Poly coefs        : {[round(c, 6) for c in info['coef']]}")
print()

# Predicted salary at benchmark levels (age_at_signing=29)
print('Predicted salary at age_at_signing=29:')
print(f"  {'RWPI':>6}  {'cap%':>7}  {'salary':>12}")
for rwpi_val in [0, 25, 50, 75, 100]:
    x = np.array([[rwpi_val, 29.0]])
    cap_pct = float(np.clip(model.predict(x)[0], 0.005, 0.20))
    print(f"  {rwpi_val:>6}  {cap_pct:>7.4f}  ${cap_pct * CAP_CEILING:>11,.0f}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# SLAVIN
# ══════════════════════════════════════════════════════════════════════════════
slavin = elig[elig['name'] == 'Jaccob Slavin']
if not slavin.empty:
    r = slavin.iloc[0]
    idx = slavin.index[0]
    print(f"Jaccob Slavin: off={r['rwpi_off_score']:.1f}  def={r['rwpi_def_score']:.1f}  rwpi={r['rwpi_score']:.1f}")
    print(
        f"  DEF: pk_toi_pct={def_feats.loc[idx,'pk_toi_pct']:.3f}"
        f"(p{def_pct.loc[idx,'pk_toi_pct']:.0f})"
        f"  pk_toi={def_feats.loc[idx,'pk_toi']:.2f}"
        f"(p{def_pct.loc[idx,'pk_toi']:.0f})"
        f"  dz_pct={def_feats.loc[idx,'dz_pct']:.3f}"
        f"(p{def_pct.loc[idx,'dz_pct']:.0f})"
        f"  sca_rel={def_feats.loc[idx,'sca_rel']:+.3f}"
        f"(p{def_pct.loc[idx,'sca_rel']:.0f})"
        f"  blocks60={def_feats.loc[idx,'blocks60']:.2f}"
        f"(p{def_pct.loc[idx,'blocks60']:.0f})"
    )
print()

# ══════════════════════════════════════════════════════════════════════════════
# TOP / BOTTOM 10
# ══════════════════════════════════════════════════════════════════════════════
cols = ['name', 'pos', 'rwpi_role', 'rwpi_off_score', 'rwpi_def_score', 'rwpi_score', 'cap_hit']
print('--- Top 10 RWPI ---')
print(elig.nlargest(10, 'rwpi_score')[cols].to_string(index=False))
print()
print('--- Bottom 10 RWPI ---')
print(elig.nsmallest(10, 'rwpi_score')[cols].to_string(index=False))
print()

# Salary bin check
td = info['train_df'].copy()
td['bin'] = pd.cut(td['rwpi_score'], bins=[0, 20, 40, 60, 80, 100])
print('Mean actual cap_hit by RWPI bin (training set):')
print(td.groupby('bin', observed=True)['cap_hit'].agg(['mean', 'count']).round(0))
