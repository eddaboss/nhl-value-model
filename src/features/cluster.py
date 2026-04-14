"""
Step 1 — K-means role clustering (k=7) using deployment + production features.
Step 2 — Performance scoring within cluster (-100 to 100).

K-means discovers 7 natural market tiers.  After fitting, clusters are
auto-labeled to match the real NHL salary market:

  1. Elite               — McDavid / Makar tier.  Highest TOI + scoring + PP.
  2. Top-Line F          — 1st/2nd line forwards.  High TOI, strong G60, PP time.
  3. Middle-Six F        — 3rd liners / lower 2nd liners.  Moderate stats.
  4. Bottom-Six F        — 4th liners / energy guys.  Low everything.
  5. Top-Four D          — High-TOI D with PP time.
  6. Bottom-Pair D       — Depth D / 7th D.
  7. Two-Way / Shutdown  — High TOI but low scoring.  Defensive specialists.

Called from build_features() to add cluster_label and performance_score.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

N_CLUSTERS = 7

# Performance scoring features per cluster label.
# Each cluster is scored on the stats that matter for that market tier.
CLUSTER_SCORE_FEATURES: dict[str, list[str]] = {
    "Elite":              ["g60", "p60", "p60_24", "pp_pts", "toi_per_g", "plus_minus"],
    "Top-Line F":         ["g60", "p60", "p60_24", "pp_pts", "shooting_pct", "shots", "plus_minus"],
    "Middle-Six F":       ["g60", "p60", "pp_pts", "shooting_pct", "shots"],
    "Bottom-Six F":       ["p60", "toi_per_g", "plus_minus"],
    "Top-Four D":         ["toi_per_g", "pp_pts", "plus_minus", "p60", "shooting_pct"],
    "Bottom-Pair D":      ["toi_per_g", "plus_minus"],
    "Two-Way / Shutdown": ["toi_per_g", "plus_minus"],
}

# Fallback for any label not in the map above
_DEFAULT_SCORE_FEATURES = ["p60", "toi_per_g", "pp_pts", "plus_minus"]


# ── Feature matrix for K-means ───────────────────────────────────────────────

def _cluster_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Build numeric matrix for KMeans.

    Features chosen so K-means can separate the 7 market tiers:
      - toi_per_g:   usage level (Elite/Top vs Bottom, Top-Four D vs Bottom-Pair)
      - p60:         scoring rate (Elite vs Top-Line vs Mid vs Bottom)
      - p60_sq:      p60 squared — amplifies the gap between elite (4+) and
                     top-line (2-3) so K-means can discover a small elite tier
      - g60:         goal-scoring rate (further separates offensive tiers)
      - pp_pts:      PP contribution (offensive vs defensive roles)
      - is_defense:  position flag so D clusters separate from F clusters
    """
    X = pd.DataFrame(index=df.index)
    X["toi_per_g"] = pd.to_numeric(df["toi_per_g"], errors="coerce").fillna(0.0)
    p60            = pd.to_numeric(df["p60"],        errors="coerce").fillna(0.0)
    X["p60"]       = p60
    X["p60_sq"]    = p60 ** 2      # quadratic — stretches elite away from rest
    X["g60"]       = pd.to_numeric(df["g60"],        errors="coerce").fillna(0.0)
    X["pp_pts"]    = pd.to_numeric(df["pp_pts"],     errors="coerce").fillna(0.0)

    # Single binary flag: defense vs forward
    pos = df["pos"].astype(str) if "pos" in df.columns else pd.Series("L", index=df.index)
    X["is_defense"] = (pos == "D").astype(float)

    return X.values


# ── Cluster labeling ──────────────────────────────────────────────────────────

def _label_clusters(df: pd.DataFrame, cluster_ids: np.ndarray) -> dict[int, str]:
    """
    Map K-means cluster IDs → descriptive market-tier labels.

    Strategy:
      1. Detect Two-Way / Shutdown FIRST — it can be F or D.  Signature:
         above-median TOI but below-median p60 (lots of ice time, little scoring).
      2. Separate remaining clusters into D-majority and F-majority.
      3. D clusters: rank by TOI → Top-Four D, Bottom-Pair D.
      4. F clusters: rank by offensive production →
         Elite, Top-Line F, Middle-Six F, Bottom-Six F.
    """
    tmp = df.copy()
    tmp["_c"] = cluster_ids

    rows = []
    for c in sorted(set(cluster_ids)):
        sub = tmp[tmp["_c"] == c]
        pos = sub["pos"].astype(str) if "pos" in sub else pd.Series(["L"] * len(sub))
        rows.append({
            "cluster": c,
            "n":       len(sub),
            "toi":     pd.to_numeric(sub["toi_per_g"], errors="coerce").mean(),
            "p60":     pd.to_numeric(sub["p60"],       errors="coerce").mean(),
            "pp":      pd.to_numeric(sub["pp_pts"],    errors="coerce").mean(),
            "g60":     pd.to_numeric(sub["g60"],       errors="coerce").mean(),
            "d_pct":   (pos == "D").mean(),
        })

    stats = pd.DataFrame(rows).set_index("cluster")
    label_map: dict[int, str] = {}

    # ── 1. Two-Way / Shutdown (position-agnostic) ─────────────────────────────
    # High ice time + low scoring = defensive specialist / checking-line player.
    median_toi = stats["toi"].median()
    median_p60 = stats["p60"].median()
    tws_candidates = stats[(stats["toi"] > median_toi) & (stats["p60"] < median_p60)]
    if not tws_candidates.empty:
        tws_id = int((tws_candidates["toi"] / tws_candidates["p60"].clip(lower=0.1)).idxmax())
        label_map[tws_id] = "Two-Way / Shutdown"
        stats = stats.drop(tws_id)

    # ── 2. Split remaining into D-majority and F-majority ─────────────────────
    d_clusters = stats[stats["d_pct"] > 0.60].index.tolist()
    f_clusters = stats[stats["d_pct"] <= 0.60].index.tolist()

    # ── 3. D clusters — rank by TOI ──────────────────────────────────────────
    d_sorted = stats.loc[d_clusters].sort_values("toi", ascending=False)
    d_names = ["Top-Four D", "Bottom-Pair D"]
    for i, c in enumerate(d_sorted.index):
        label_map[int(c)] = d_names[min(i, len(d_names) - 1)]

    # ── 4. F clusters — rank by offensive production ─────────────────────────
    f_stats = stats.loc[f_clusters].copy()
    f_stats["rank_score"] = f_stats["p60"] * 10 + f_stats["pp"] * 0.5 + f_stats["toi"] * 0.3
    f_sorted = f_stats.sort_values("rank_score", ascending=False)
    f_names = ["Elite", "Top-Line F", "Middle-Six F", "Bottom-Six F"]
    for i, c in enumerate(f_sorted.index):
        label_map[int(c)] = f_names[min(i, len(f_names) - 1)]

    return label_map


# ── Performance scoring ───────────────────────────────────────────────────────

def _add_performance_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add performance_score (-100 to 100) to each player.

    Uses cluster-specific scoring features: Elite is scored on different
    stats than Bottom-Six F or Two-Way/Shutdown.

    Within each cluster:
      1. Compute z-score for each scoring feature vs cluster-mates.
      2. Average z-scores using the cluster's feature set.
      3. Min-max scale the composite to [-100, 100].
    """
    df = df.copy()

    # Collect all features we might need across all clusters
    all_feats: set[str] = set()
    for feats in CLUSTER_SCORE_FEATURES.values():
        all_feats.update(feats)
    all_feats.update(_DEFAULT_SCORE_FEATURES)
    available = [f for f in all_feats if f in df.columns]

    # Per-cluster z-scores for every scoring feature
    for feat in available:
        zname = f"_z_{feat}"
        df[zname] = np.nan
        raw = pd.to_numeric(df[feat], errors="coerce")
        for label in df["cluster_label"].unique():
            mask = df["cluster_label"] == label
            vals = raw[mask]
            std = vals.std()
            if std > 1e-9:
                df.loc[mask, zname] = (vals - vals.mean()) / std
            else:
                df.loc[mask, zname] = 0.0

    # Per-cluster composite using cluster-specific features
    df["_composite"] = np.nan
    for label in df["cluster_label"].unique():
        mask = df["cluster_label"] == label
        feats = CLUSTER_SCORE_FEATURES.get(label, _DEFAULT_SCORE_FEATURES)
        z_cols = [f"_z_{f}" for f in feats if f in df.columns]
        if z_cols:
            df.loc[mask, "_composite"] = df.loc[mask, z_cols].mean(axis=1)
        else:
            df.loc[mask, "_composite"] = 0.0

    # Scale to -100 to 100 within each cluster
    df["performance_score"] = np.nan
    for label in df["cluster_label"].unique():
        mask = df["cluster_label"] == label
        vals = df.loc[mask, "_composite"]
        mn, mx = vals.min(), vals.max()
        if mx > mn:
            scaled = (vals - mn) / (mx - mn) * 200.0 - 100.0
        else:
            scaled = pd.Series(0.0, index=vals.index)
        df.loc[mask, "performance_score"] = scaled.round(1)

    # Drop temporary columns
    drop_cols = [c for c in df.columns if c.startswith("_z_") or c == "_composite"]
    df.drop(columns=drop_cols, inplace=True)

    return df


# ── Public API ────────────────────────────────────────────────────────────────

_MIN_GP_FOR_CLUSTERING = 20   # ignore part-time players for K-means fitting


def fit_and_apply(df: pd.DataFrame) -> tuple[pd.DataFrame, KMeans | None, StandardScaler | None]:
    """
    Fit K-means on players with enough games played, then predict cluster
    membership for ALL players (including low-GP ones).
    """
    df = df.copy()

    # Determine which players are eligible for fitting
    gp_col = pd.to_numeric(df.get("gp_actual", pd.Series(82, index=df.index)),
                           errors="coerce").fillna(0)
    fit_mask = gp_col >= _MIN_GP_FOR_CLUSTERING

    X_all = _cluster_matrix(df)
    scaler = StandardScaler()

    # Fit scaler + KMeans on eligible players only
    X_fit = X_all[fit_mask.values]
    scaler.fit(X_fit)
    X_fit_scaled = scaler.transform(X_fit)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
    kmeans.fit(X_fit_scaled)

    # Predict cluster for ALL players (including low-GP)
    X_all_scaled = scaler.transform(X_all)
    cluster_ids = kmeans.predict(X_all_scaled)

    label_map = _label_clusters(df[fit_mask], kmeans.labels_)

    df["cluster_id"]    = cluster_ids
    df["cluster_label"] = [label_map.get(int(c), "Unknown") for c in cluster_ids]

    df = _add_performance_score(df)

    return df, kmeans, scaler
