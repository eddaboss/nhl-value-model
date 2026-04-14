"""
Step 1 — K-means role clustering (k=6) using deployment features.
Step 2 — Performance scoring within cluster (-100 to 100).

Called from build_features() to add cluster_label and performance_score
columns to the dataframe. Saves fitted artifacts to models/kmeans_cluster.pkl.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path(__file__).parents[2] / "models"
N_CLUSTERS = 6

# Performance scoring features by position group
FWD_SCORE_FEATURES = ["g60", "p60_24", "pp_pts", "shooting_pct", "shots", "plus_minus"]
DEF_SCORE_FEATURES = ["toi_per_g", "pp_pts", "plus_minus", "gp", "shooting_pct"]


# ── Feature preparation ────────────────────────────────────────────────────────

def _cluster_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Build numeric matrix for KMeans from deployment features.
    faceoff_pct is set to 0 for non-centers (they genuinely take no faceoffs).
    pos is one-hot encoded.
    """
    X = pd.DataFrame(index=df.index)
    X["toi_per_g"]  = pd.to_numeric(df["toi_per_g"],  errors="coerce").fillna(0.0)
    X["pp_pts"]     = pd.to_numeric(df["pp_pts"],     errors="coerce").fillna(0.0)
    X["gp"]         = pd.to_numeric(df["gp"],         errors="coerce").fillna(0.0)
    X["plus_minus"] = pd.to_numeric(df["plus_minus"], errors="coerce").fillna(0.0)

    # faceoff_pct: real value for centers, 0 for everyone else
    fo = pd.to_numeric(df.get("faceoff_pct", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    if "pos" in df.columns:
        fo = fo.where(df["pos"].astype(str) == "C", 0.0)
    X["faceoff_pct"] = fo

    # One-hot encode position
    pos = df["pos"].astype(str) if "pos" in df.columns else pd.Series("L", index=df.index)
    for p in ["C", "L", "R", "D"]:
        X[f"pos_{p}"] = (pos == p).astype(float)

    return X.values


# ── Cluster labeling ───────────────────────────────────────────────────────────

def _label_clusters(df: pd.DataFrame, cluster_ids: np.ndarray) -> dict[int, str]:
    """
    Auto-assign descriptive names to cluster IDs based on centroid stats.
    Separates D-majority from F-majority clusters, then ranks within each group.
    """
    tmp = df.copy()
    tmp["_c"] = cluster_ids

    rows = []
    for c in range(N_CLUSTERS):
        sub = tmp[tmp["_c"] == c]
        pos = sub["pos"].astype(str) if "pos" in sub else pd.Series(["L"] * len(sub))
        rows.append({
            "cluster": c,
            "n":       len(sub),
            "toi":     pd.to_numeric(sub["toi_per_g"], errors="coerce").mean(),
            "pp":      pd.to_numeric(sub["pp_pts"],    errors="coerce").mean(),
            "d_pct":   (pos == "D").mean(),
            "c_pct":   (pos == "C").mean(),
        })

    stats = pd.DataFrame(rows).set_index("cluster")

    d_clusters = stats[stats["d_pct"] > 0.50].index.tolist()
    f_clusters  = stats[stats["d_pct"] <= 0.50].index.tolist()

    label_map: dict[int, str] = {}

    # D clusters — rank by average TOI
    d_sorted = stats.loc[d_clusters].sort_values("toi", ascending=False)
    d_names  = ["Top-Pair D", "Bottom-Pair D", "3rd-Pair D"]
    for i, c in enumerate(d_sorted.index):
        label_map[int(c)] = d_names[min(i, len(d_names) - 1)]

    # F clusters — rank by PP production + TOI (proxy for offensive role)
    f_stats = stats.loc[f_clusters].copy()
    f_stats["rank_score"] = f_stats["pp"] * 2.0 + f_stats["toi"]
    f_sorted = f_stats.sort_values("rank_score", ascending=False)

    # Names in order of descending offensive deployment
    f_names = ["Top-Line C/F", "Top-Six F", "PP Specialist", "Checking C", "Bottom-Six F"]
    for i, c in enumerate(f_sorted.index):
        label_map[int(c)] = f_names[min(i, len(f_names) - 1)]

    return label_map


# ── Performance scoring ────────────────────────────────────────────────────────

def _add_performance_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add performance_score (-100 to 100) to each player.

    Within each cluster:
      1. Compute z-score for each scoring feature vs cluster-mates.
      2. Average z-scores for the player's position group (F or D).
      3. Min-max scale the composite to [-100, 100].

    Result: a player at the cluster median scores near 0;
            cluster leader ≈ 100; cluster bottom ≈ -100.
    """
    df = df.copy()
    is_d = df["pos"].astype(str) == "D"

    all_feats = list(dict.fromkeys(FWD_SCORE_FEATURES + DEF_SCORE_FEATURES))
    available = [f for f in all_feats if f in df.columns]

    # Per-cluster z-scores for every scoring feature
    for feat in available:
        zname = f"_z_{feat}"
        df[zname] = np.nan
        raw = pd.to_numeric(df[feat], errors="coerce")
        for cid in df["cluster_id"].unique():
            mask = df["cluster_id"] == cid
            vals = raw[mask]
            std  = vals.std()
            if std > 1e-9:
                df.loc[mask, zname] = (vals - vals.mean()) / std
            else:
                df.loc[mask, zname] = 0.0

    fwd_z = [f"_z_{f}" for f in FWD_SCORE_FEATURES if f in df.columns]
    def_z = [f"_z_{f}" for f in DEF_SCORE_FEATURES if f in df.columns]

    fwd_composite = df[fwd_z].mean(axis=1)
    def_composite = df[def_z].mean(axis=1)
    df["_composite"] = np.where(is_d, def_composite, fwd_composite)

    # Scale to -100 to 100 within each cluster
    df["performance_score"] = np.nan
    for cid in df["cluster_id"].unique():
        mask = df["cluster_id"] == cid
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


# ── Public API ─────────────────────────────────────────────────────────────────

def fit_and_apply(df: pd.DataFrame) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Fit K-means on all players, then add cluster_id, cluster_label,
    and performance_score columns to df. Saves fitted artifacts to disk.
    """
    X = _cluster_matrix(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
    kmeans.fit(X_scaled)
    cluster_ids = kmeans.labels_

    label_map = _label_clusters(df, cluster_ids)

    df = df.copy()
    df["cluster_id"]    = cluster_ids
    df["cluster_label"] = [label_map[int(c)] for c in cluster_ids]

    df = _add_performance_score(df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"kmeans": kmeans, "scaler": scaler, "label_map": label_map},
        MODELS_DIR / "kmeans_cluster.pkl",
    )

    return df, kmeans, scaler
