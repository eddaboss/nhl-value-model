import sys
sys.path.insert(0, ".")
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from src.data.load import load_and_merge
from src.features.build import build_features
from src.models.comps import build_ufa_comp_pool, find_comps

df_raw, ctx = load_and_merge()
df = build_features(df_raw)
comp_pool = build_ufa_comp_pool(df)

name_query = sys.argv[1] if len(sys.argv) > 1 else "panarin"
matches = df[df["name"].str.lower().str.contains(name_query.lower())]
if matches.empty:
    print(f"No player found matching '{name_query}'")
    sys.exit(1)

player = matches.iloc[0]
print(f"\n{player['name']}  ({player['team']}, {player['pos']}, age {player['age']:.1f})")
print(f"  Cluster:     {player['cluster_label']}")
print(f"  Perf Score:  {player['performance_score']:+.1f}")
print(f"  p60 (cur):   {player['p60']:.2f}")
print(f"  p60_24:      {player['p60_24']:.2f}")
print(f"  g60:         {player['g60']:.2f}")
print(f"  pp_pts:      {player['pp_pts']:.1f}")
print(f"  toi_per_g:   {player['toi_per_g']:.2f}")
print(f"  plus_minus:  {player['plus_minus']:+.0f}")
print(f"  Cap Hit:     ${player['cap_hit']:,.0f}")
print()

comps = find_comps(player, comp_pool, n=10)
print(f"Top 10 comps (pool size: {len(comp_pool)} contracted players):")
print(f"  {'Name':<28} {'Score':>7}  {'AAV':>12}  {'p60':>5}  {'pp_pts':>6}  {'Age':>5}  Cluster")
print("  " + "-" * 90)
for _, r in comps.iterrows():
    print(f"  {r['name']:<28} {r['performance_score']:>+7.1f}  ${r['cap_hit']:>11,.0f}  {r['p60']:>5.2f}  {r['pp_pts']:>6.1f}  {r['age']:>5.1f}  {r['cluster_label']}")
