"""
Verify new MoneyPuck fields are populated after merging.
Usage:  py -3 scripts/verify_new_fields.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import numpy as np

from src.data.load import load_and_merge

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.2f}".format)

NEW_MP_FIELDS = [
    "a1", "a2", "hits", "blocks",
    "xgf60", "xga60", "cf_pct", "oz_pct",
    "pp_toi", "pk_toi",
]

SAMPLE_NAMES = [
    "connor mcdavid",
    "quinton byfield",
    "anze kopitar",
    "drew doughty",
    "macklin celebrini",
    "brandt clarke",
    "artemi panarin",
    "adrian kempe",
    # defensive forward and defensive D filled by name search below
]

DISPLAY_COLS = [
    "name", "team", "pos",
    "g", "a1", "a2",
    "xgf60", "xga60",
    "pp_toi", "pk_toi",
    "shots", "shooting_pct",
    "hits", "blocks",
    "cf_pct", "oz_pct",
    "toi_per_g",
    "cap_hit", "expiry_status",
]

def main():
    print("=" * 70)
    print("Loading full dataset (NHL API + contracts + MoneyPuck)...")
    print("=" * 70)
    df, ctx = load_and_merge()

    # ── Field coverage report ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FIELD COVERAGE REPORT")
    print("=" * 70)
    n = len(df)
    all_fields = NEW_MP_FIELDS + ["g", "a", "shots", "shooting_pct",
                                   "toi_per_g", "pp_pts", "cap_hit",
                                   "expiry_status"]
    rows = []
    for col in all_fields:
        if col in df.columns:
            filled = df[col].notna().sum()
            source = "MoneyPuck" if col in NEW_MP_FIELDS else "NHL API / contracts.db"
            rows.append({"field": col, "populated": filled, "total": n,
                         "pct": f"{filled/n*100:.1f}%", "source": source})
        else:
            rows.append({"field": col, "populated": 0, "total": n,
                         "pct": "MISSING", "source": "—"})
    cov = pd.DataFrame(rows)
    print(cov.to_string(index=False))

    # ── Sample table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAMPLE: 10 TARGET PLAYERS")
    print("=" * 70)

    df["name_lower"] = df["name"].str.lower().str.strip()

    # Find by name; also find a known defensive forward and D
    target_rows = []
    found_names = set()

    for nm in SAMPLE_NAMES:
        match = df[df["name_lower"] == nm]
        if not match.empty:
            target_rows.append(match.iloc[0])
            found_names.add(nm)
        else:
            # Partial match fallback
            partial = df[df["name_lower"].str.contains(nm.split()[-1], na=False)]
            if not partial.empty:
                target_rows.append(partial.iloc[0])
                found_names.add(nm)
            else:
                print(f"  WARNING: '{nm}' not found in dataset")

    # Defensive forward: high pk_toi, low oz_pct, forward
    if "pk_toi" in df.columns and "oz_pct" in df.columns:
        def_fwd = (
            df[df["pos"].isin(["C", "L", "R"]) & df["pk_toi"].notna() & df["oz_pct"].notna()]
            .nlargest(20, "pk_toi")
            .nsmallest(1, "oz_pct")
        )
        if not def_fwd.empty:
            row = def_fwd.iloc[0]
            print(f"  Defensive forward selected: {row['name']} (PK TOI {row['pk_toi']:.2f} min/g, OZ% {row['oz_pct']:.3f})")
            target_rows.append(row)

    # Defensive D: high pk_toi, low oz_pct, defenseman
    if "pk_toi" in df.columns and "oz_pct" in df.columns:
        def_d = (
            df[df["pos"] == "D" & df["pk_toi"].notna() & df["oz_pct"].notna()]
            .nlargest(20, "pk_toi")
            .nsmallest(1, "oz_pct")
        ) if False else (
            df[(df["pos"] == "D") & df["pk_toi"].notna() & df["oz_pct"].notna()]
            .nlargest(20, "pk_toi")
            .nsmallest(1, "oz_pct")
        )
        if not def_d.empty:
            row = def_d.iloc[0]
            print(f"  Defensive defenseman selected: {row['name']} (PK TOI {row['pk_toi']:.2f} min/g, OZ% {row['oz_pct']:.3f})")
            target_rows.append(row)

    if not target_rows:
        print("No target players found!")
        return

    sample = pd.DataFrame(target_rows)
    present_cols = [c for c in DISPLAY_COLS if c in sample.columns]
    out = sample[present_cols].copy()

    # Format for display
    for col in ["cap_hit"]:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda v: f"${v/1e6:.2f}M" if pd.notna(v) else "UFA"
            )
    for col in ["xgf60", "xga60", "cf_pct", "oz_pct"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
    for col in ["pp_toi", "pk_toi", "toi_per_g"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    for col in ["shooting_pct"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    for col in ["g", "a1", "a2", "hits", "blocks", "shots"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda v: f"{v:.0f}" if pd.notna(v) else "—")

    print(out.to_string(index=False))

    # ── MoneyPuck coverage summary ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MONEYPUCK JOIN SUMMARY")
    print("=" * 70)
    if "a1" in df.columns:
        n_mp = df["a1"].notna().sum()
        print(f"Players with MoneyPuck data: {n_mp} / {n} ({n_mp/n*100:.1f}%)")
        print(f"Players WITHOUT MoneyPuck:   {n - n_mp}")
        if n - n_mp > 0:
            missing_mp = df[df["a1"].isna()][["name", "team", "pos", "gp"]].head(15)
            print("\nSample of players missing MoneyPuck data:")
            print(missing_mp.to_string(index=False))
    else:
        print("  a1 column not found — MoneyPuck merge may have failed")


if __name__ == "__main__":
    main()
