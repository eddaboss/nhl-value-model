"""
NHL Player Value Model — Streamlit App
Tabs: League Overview | Leaderboards | LA Kings | Player Search | Model Insights
"""
import json
import sys
import threading
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# Ensure repo root is on sys.path (needed on Streamlit Cloud)
_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Background refresh state (module-level, shared across Streamlit reruns) ────
_refresh_status: dict = {"running": False, "done": False, "error": None, "started_at": None, "completed": False}
_refresh_lock = threading.Lock()


def _run_pipeline_background(processed_dir: Path) -> None:
    """Run load+predict in a background thread and write predictions.csv when done."""
    global _refresh_status
    try:
        import warnings
        warnings.filterwarnings("ignore")

        from src.data.load import load_and_merge, CAP_CEILING
        from src.features.build import build_features, resign_label
        from src.models.comps import run_comps_model

        df_raw, ctx = load_and_merge()
        df = build_features(df_raw)  # also applies clustering + performance_score

        # Comps model — primary predictions (same as pipeline.py)
        df, _ = run_comps_model(df)

        # Suppress predictions for low-GP players (inflated rate stats)
        _MIN_GP = 20
        gp_actual = pd.to_numeric(df.get("gp_actual", pd.Series(82, index=df.index)),
                                   errors="coerce").fillna(0)
        low_gp = gp_actual < _MIN_GP
        df.loc[low_gp, "predicted_value"] = None
        df.loc[low_gp, "value_delta"] = None

        df["resign_signal"] = df.apply(resign_label, axis=1)

        keep = [
            "name", "team", "pos", "age",
            "cap_hit", "predicted_value", "value_delta",
            "expiry_status", "expiry_year", "years_left", "length_of_contract",
            "gp", "gp_actual", "g", "a", "p", "ppg",
            "toi_per_g", "plus_minus", "pim",
            "g60", "p60", "pp_pts", "shots", "shooting_pct",
            "cluster_id", "cluster_label", "performance_score",
            "faceoff_pct",
            "resign_signal", "player_id",
            "has_contract_data", "has_prior_market_data", "is_estimated",
            "has_extension", "extension_cap_hit", "extension_start_year",
            "extension_expiry_year", "extension_length", "extension_expiry_status",
        ]
        out = df[[c for c in keep if c in df.columns]].copy()
        for col in ["cap_hit", "predicted_value", "value_delta"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").round(0)
        if "is_estimated" not in out.columns:
            out["is_estimated"] = False
        out["is_estimated"] = out["is_estimated"].fillna(False).astype(bool)

        # Write to a temp file then rename (atomic on most filesystems)
        tmp = processed_dir / "predictions_tmp.csv"
        out.to_csv(tmp, index=False)
        tmp.replace(processed_dir / "predictions.csv")

        # Also refresh season_context.json (include cap_ceiling so app never needs it hardcoded)
        import json as _json
        ctx_out = {**ctx, "cap_ceiling": CAP_CEILING}
        (processed_dir / "season_context.json").write_text(
            _json.dumps(ctx_out, indent=2), encoding="utf-8"
        )

        _refresh_status["done"] = True
        _refresh_status["completed"] = True
        _refresh_status["error"] = None
    except Exception as e:
        _refresh_status["error"] = str(e)
    finally:
        _refresh_status["running"] = False


_PREDICTIONS_MAX_AGE_HOURS = 12   # skip background refresh if file is this fresh


def start_background_refresh(processed_dir: Path) -> None:
    """Launch background pipeline thread once per process lifetime.

    Skipped when predictions.csv already exists and is recent (< MAX_AGE_HOURS).
    This prevents the thread from overwriting committed, validated predictions
    with a fresh K-means run that might produce different cluster assignments.
    On Streamlit Cloud the committed file is always available at boot, so the
    thread only fires when data is genuinely stale.
    """
    pred_path = processed_dir / "predictions.csv"
    if pred_path.exists():
        age_hours = (time.time() - pred_path.stat().st_mtime) / 3600
        if age_hours < _PREDICTIONS_MAX_AGE_HOURS:
            # Mark completed so we don't re-launch, but leave done=False so
            # the main thread doesn't call st.rerun() and cause an infinite loop.
            _refresh_status["done"] = False
            _refresh_status["completed"] = True
            return

    with _refresh_lock:
        if _refresh_status["running"] or _refresh_status["completed"]:
            return
        _refresh_status["running"] = True
        _refresh_status["done"] = False
        _refresh_status["error"] = None
        _refresh_status["started_at"] = time.time()
    t = threading.Thread(
        target=_run_pipeline_background, args=(processed_dir,), daemon=True
    )
    t.start()

# ── Feature label map ──────────────────────────────────────────────────────────
_FEATURE_LABELS: dict[str, str] = {
    "toi_per_g":          "Ice Time / Game",
    "toi_per_g_24":       "Ice Time / Game (Prior Season)",
    "pp_pts":             "Power Play Points",
    "pp_pts_24":          "Power Play Points (Prior Season)",
    "ppg":                "Points Per Game",
    "ppg_24":             "Points Per Game (Prior Season)",
    "shooting_pct":       "Shooting %",
    "shooting_pct_24":    "Shooting % (Prior Season)",
    "length_of_contract": "Contract Length",
    "draft_position":     "Draft Position",
    "draft_year":         "Draft Year",
    "year_of_contract":   "Contract Year",
    "plus_minus":         "Plus / Minus",
    "plus_minus_24":      "Plus / Minus (Prior Season)",
    "hits":               "Hits",
    "hits_24":            "Hits (Prior Season)",
    "blocks":             "Blocked Shots",
    "blocks_24":          "Blocked Shots (Prior Season)",
    "fenwick_pct":        "Fenwick %",
    "oz_start_pct":       "Offensive Zone Start %",
    "xg":                 "Expected Goals",
    "xg_24":              "Expected Goals (Prior Season)",
    "pk_toi":             "Penalty Kill TOI / Game",
    "pp_toi":             "Power Play TOI / Game",
    "faceoff_pct":        "Faceoff Win %",
    "gp":                 "Games Played",
    "age":                "Age",
    "g":                  "Goals",
    "a":                  "Assists",
    "p":                  "Points",
    "g60":                "Goals / 60",
    "p60":                "Points / 60",
    "shots":              "Shots",
    "pim":                "Penalty Minutes",
}

def _label(feature: str) -> str:
    """Return a human-readable label for a SHAP feature column name."""
    return _FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def _driver_tooltip(feat: str, player: pd.Series, name: str, positive: bool) -> str:
    """Return a plain-English sentence explaining why a feature drives value up or down."""
    import textwrap

    def _get(col, fmt=".1f"):
        v = player.get(col)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        try:
            return format(float(v), fmt)
        except Exception:
            return str(v)

    toi      = _get("toi_per_g")
    goals    = _get("g", ".0f")
    ppg_v    = _get("ppg")
    age_v    = _get("age", ".0f")
    pm       = _get("plus_minus", "+.0f")
    spct     = _get("shooting_pct")
    hits_v   = _get("hits", ".0f")
    blk_v    = _get("blocks", ".0f")
    xg_v     = _get("xg")
    fen_v    = _get("fenwick_pct")
    oz_v     = _get("oz_start_pct")
    pp_toi_v = _get("pp_toi")
    pk_toi_v = _get("pk_toi")
    loc_raw  = player.get("length_of_contract")
    loc_v    = int(float(loc_raw)) if loc_raw and not (isinstance(loc_raw, float) and pd.isna(loc_raw)) else None
    dp_v     = _get("draft_position", ".0f")

    is_prior  = feat.endswith("_24")
    base_feat = feat[:-3] if is_prior else feat

    tips: dict[str, str] = {
        "toi_per_g": (
            f"{name} averages {toi} minutes per game — one of the highest totals in the league. Coaches allocate ice time purely on merit, making this one of the most honest signals of how much a team values a player."
            if positive else
            f"{name} averages {toi} minutes per game, indicating a depth role. Ice time is allocated by coaches based on performance and trust — lower minutes reflect a more limited role in the lineup."
        ) if toi else (
            f"{name} logs elite ice time. Coaches allocate ice time purely on merit, making this one of the most honest signals of how much a team values a player."
            if positive else
            f"{name} logs limited ice time, indicating a depth role. Ice time is allocated by coaches based on performance and trust."
        ),
        "g": (
            f"{name} has scored {goals} goals this season — an elite pace that directly signals offensive value to any team."
            if positive else
            f"{name} has scored {goals} goals this season. Lower goal totals reduce predicted value, though role and deployment context always matters when interpreting raw counts."
        ) if goals else (
            f"{name}'s goal-scoring pace directly signals elite offensive value."
            if positive else
            f"{name}'s lower goal totals reduce predicted value, though role and deployment context always matters."
        ),
        "ppg": (
            f"{name} is producing at {ppg_v} points per game — an elite offensive rate that commands a premium on the open market."
            if positive else
            f"{name} is producing at {ppg_v} points per game, below average for a player at this usage level. Consistent point production is one of the strongest drivers of market value."
        ) if ppg_v else (
            f"{name} is producing at an elite offensive rate that commands a premium on the open market."
            if positive else
            f"{name}'s production is below average for their usage level. Consistent point production is one of the strongest drivers of market value."
        ),
        "pp_toi": (
            f"{name} averages {pp_toi_v} minutes of power play time per game. Power play deployment is one of the clearest signals of offensive trust — coaches only put their best offensive players on the ice with the man advantage."
            if positive else
            f"{name} receives minimal power play time. Since power play deployment reflects a coach's offensive trust, limited PP time suggests a more defensive or depth role in the lineup."
        ) if pp_toi_v else (
            f"{name} receives significant power play time — one of the clearest signals of offensive trust from coaching staff."
            if positive else
            f"{name} receives minimal power play time, suggesting a more defensive or depth role in the lineup."
        ),
        "pk_toi": (
            f"{name} averages {pk_toi_v} minutes of PK time per game. Elite penalty killers are genuinely valued across the league — coaches only deploy players they trust completely in high-pressure defensive situations."
            if positive else
            f"{name} sees little penalty kill time. Not all valuable players kill penalties, but PK deployment adds demonstrable two-way value that teams factor into contracts."
        ) if pk_toi_v else (
            f"{name} logs significant penalty kill time. Elite PK players are genuinely valued — coaches only deploy players they trust completely in high-pressure defensive situations."
            if positive else
            f"{name} sees little penalty kill time. Not all valuable players kill penalties, but PK deployment adds demonstrable two-way value that teams factor into contracts."
        ),
        "fenwick_pct": (
            f"When {name} is on the ice, their team controls {fen_v}% of unblocked shot attempts. Fenwick% is a strong indicator of puck possession and shot attempt control, though zone starts and quality of competition also affect this number."
            if positive else
            f"The team is outshot when {name} is on the ice at {fen_v}% Fenwick. This is worth noting but should be read with context — players deployed primarily in defensive zone situations will naturally have lower Fenwick% numbers regardless of their individual quality."
        ) if fen_v else (
            f"When {name} is on the ice, their team controls a strong share of unblocked shot attempts — a good indicator of puck possession and shot attempt control."
            if positive else
            f"The team is outshot when {name} is on the ice. Players deployed primarily in defensive situations will naturally have lower Fenwick% regardless of their individual quality."
        ),
        "xg": (
            f"{name} has generated {xg_v} expected goals based on shot quality and location — a more reliable measure of offensive threat than raw goals, since it removes the influence of shooting luck."
            if positive else
            f"{name} has generated {xg_v} expected goals. Lower xG suggests fewer high danger scoring chances, which is a more reliable negative signal than a low goal total alone since it accounts for shot quality."
        ) if xg_v else (
            f"{name} generates strong expected goals numbers — a more reliable offensive measure than raw goals since it accounts for shot quality and removes shooting luck."
            if positive else
            f"{name}'s lower xG suggests fewer high-danger scoring chances, a more reliable signal than raw goal totals since it accounts for shot quality."
        ),
        "oz_start_pct": (
            f"{name} starts {oz_v}% of shifts in the offensive zone. High offensive zone deployment reflects a coach's decision to use this player as an offensive weapon."
            if positive else
            f"{name} starts only {oz_v}% of shifts in the offensive zone. This is often intentional — shutdown forwards and defensive defensemen are deliberately deployed in their own end to protect leads and neutralize top opposing lines. Low OZ% can signal defensive value rather than poor play."
        ) if oz_v else (
            f"{name} starts a high share of shifts in the offensive zone, reflecting a coach's decision to use them as an offensive weapon."
            if positive else
            f"{name} starts few shifts in the offensive zone — often intentional. Shutdown forwards and defensive defensemen are deliberately deployed in their own end, so low OZ% can signal defensive value rather than poor play."
        ),
        "plus_minus": (
            f"{name} has a plus/minus of {pm}. This traditional stat counts goals for and against while a player is on the ice at 5v5, but is heavily influenced by teammates, goaltending, and deployment — most modern analytics departments treat it as a weak individual performance signal."
            if positive else
            f"{name} has a plus/minus of {pm}. Important context: plus/minus is considered one of the least reliable individual stats in modern hockey analytics because it depends heavily on teammates, goaltending quality, and deployment. A negative number does not necessarily reflect poor individual performance."
        ) if pm else (
            f"{name}'s plus/minus is positive, though this stat is heavily influenced by teammates, goaltending, and deployment — most modern analytics departments treat it as a weak individual signal."
            if positive else
            f"{name}'s plus/minus is negative. Important context: plus/minus is considered one of the least reliable individual stats in modern analytics — it depends heavily on teammates and goaltending, not just individual play."
        ),
        "length_of_contract": (
            f"{loc_v} years remaining on this contract. Players can only negotiate a new contract when their current deal expires — years remaining reflects how long ago this contract was signed and what the market valued this player at that point in time."
            if positive else
            "Fewer years remaining means this contract is near expiration. The cap hit reflects what the market valued this player at when the deal was signed — which may be higher or lower than their current production level."
        ) if loc_v else (
            "More years remaining means this contract was signed more recently and likely reflects current market value more accurately."
            if positive else
            "Fewer years remaining means this contract is near expiration and may not reflect current market value."
        ),
        "draft_position": (
            f"Selected {dp_v}th overall. High draft position reflects strong organizational investment early in a player's career and still factors into market pricing, particularly for younger players who have not yet established a long NHL track record."
            if positive else
            "A later draft position carries less historical market premium. For veterans with long NHL careers, draft position becomes increasingly irrelevant as their track record speaks for itself."
        ) if dp_v else (
            "High draft pedigree factors into market pricing, particularly for younger players who have not yet established a long NHL track record."
            if positive else
            "A later draft position carries less historical market premium, though this becomes increasingly irrelevant as a player builds their NHL track record."
        ),
        "age": (
            f"At {age_v} years old, {name} is in their prime earning years. NHL players typically peak between ages 24 and 29, and the market prices players in this window at a premium."
            if positive else
            f"At {age_v} years old, {name} is past the typical NHL prime earning window. The market generally discounts players over 30 due to expected performance decline, though elite players regularly outperform age-based expectations."
        ) if age_v else (
            f"{name} is in their prime earning years. NHL players typically peak between ages 24 and 29, and the market prices players in this window at a premium."
            if positive else
            f"{name} is past the typical NHL prime earning window. The market generally discounts older players due to expected decline, though elite players regularly outperform age-based expectations."
        ),
        "shooting_pct": (
            f"{name} converts {spct}% of their shots this season. Strong shooting percentage adds offensive value, though elite playmakers sometimes have lower shooting percentages because they shoot less frequently and focus on setting up teammates."
            if positive else
            f"{name} converts {spct}% of their shots. Low shooting percentage can reflect poor finishing, a primary playmaking role rather than a shooting role, or natural variance over a season — it should not be read in isolation."
        ) if spct else (
            f"{name}'s strong shooting percentage adds offensive value, though elite playmakers sometimes shoot less because they focus on setting up teammates."
            if positive else
            f"{name}'s low shooting percentage can reflect poor finishing, a primary playmaking role, or natural variance over a season."
        ),
        "hits": (
            f"{name} has recorded {hits_v} hits this season. Physical play is valued in certain systems and by certain organizations, though modern NHL analytics research shows that hits correlate weakly with team success and winning percentage."
            if positive else
            f"{name} has recorded {hits_v} hits. Hits are weighted lightly in this model because analytics research shows limited correlation between physical play and team performance. Low hit totals often simply reflect an offensive or perimeter playing style rather than lack of value."
        ) if hits_v else (
            f"{name}'s physical play is valued in certain systems, though modern analytics research shows hits correlate weakly with team success."
            if positive else
            f"{name}'s hit totals are weighted lightly in this model. Low hit totals often reflect an offensive playing style rather than lack of value."
        ),
        "blocks": (
            f"{name} has blocked {blk_v} shots — a clear signal of defensive commitment and willingness to sacrifice their body in the defensive zone to protect their goaltender."
            if positive else
            "Fewer blocked shots typically reflects offensive zone deployment rather than lack of defensive effort. Forwards and offensive defensemen naturally block fewer shots because they spend less time defending in their own end."
        ) if blk_v else (
            f"{name}'s shot-blocking is a clear signal of defensive commitment and willingness to sacrifice their body in the defensive zone."
            if positive else
            "Fewer blocked shots typically reflects offensive deployment rather than lack of defensive effort. Forwards and offensive defensemen naturally block fewer shots."
        ),
        "ppg_24": (
            f"{name} produced at {_get('ppg')} points per game last season. Strong prior year production validates current performance and shows consistent output over time rather than a one-season outlier."
            if positive else
            f"{name} produced at {_get('ppg')} points per game last season. Lower prior year production factors into the model as an indicator of whether current output represents genuine improvement or a one-season spike."
        ),
    }

    # For any _24 feature not explicitly listed, use the base feature text with a prior-season prefix
    sentence = tips.get(feat) or tips.get(base_feat)
    if sentence is None:
        lbl = _label(feat)
        sentence = (
            f"{lbl} is a positive factor in {name}'s market value."
            if positive else
            f"{lbl} is pulling {name}'s estimated value down."
        )
    if is_prior and feat not in tips:
        sentence = f"Prior season: {sentence}"

    return "<br>".join(textwrap.wrap(sentence, width=68))



# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NHL Value Model",
    page_icon=":material/sports_hockey:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cap ceiling — read from season_context.json written by pipeline.py ─────────
# load.py is the single definition; pipeline.py writes it to season_context.json.
# Fallback only fires before the first pipeline run.
def _load_cap_ceiling() -> int:
    p = Path(__file__).parents[2] / "data" / "processed" / "season_context.json"
    if p.exists():
        try:
            return int(json.loads(p.read_text(encoding="utf-8")).get("cap_ceiling", 95_500_000))
        except Exception:
            pass
    return 95_500_000

CAP_CEILING = _load_cap_ceiling()

# ── Kings brand colors ─────────────────────────────────────────────────────────
KINGS_BLACK  = "#040404"
KINGS_GOLD   = "#C8A84B"
KINGS_SILVER = "#8A9499"
KINGS_WHITE  = "#E8E4DC"

# ── Runtime theme colour tokens — populated by _set_theme() ────────────────────
_T: dict = {}


def _set_theme(dark: bool) -> None:
    """Populate _T with palette used by Plotly charts and dynamic inline HTML."""
    global _T
    if dark:
        _T.update({
            "page_text":     "#F2EEE5",
            "plot_paper":    "#0E1013",
            "plot_bg":       "#0E1013",
            "plot_font":     "#8A8F99",
            "grid":          "#262B33",
            "grid_alt":      "#262B33",
            "zero":          "#343A44",
            "legend_bg":     "#1A1E24",
            "card_bg":       "#1A1E24",
            "card_border":   "#262B33",
            "card_text":     "#F2EEE5",
            "card_subtext":  "#8A8F99",
            "card_header":   "#14171C",
            "row_divider":   "#262B33",
            "accent":        "#F2EEE5",
            "accent_ice":    "#4FD1C5",
            "accent_purple": "#A78BFA",
            "gold":          "#E5B664",
            "positive":      "#7DD87A",
            "negative":      "#F07A7A",
        })
    else:
        _T.update({
            "page_text":     "#0B0D10",
            "plot_paper":    "rgba(0,0,0,0)",
            "plot_bg":       "#F5F2EC",
            "plot_font":     "#6A6F78",
            "grid":          "#E4DFD5",
            "grid_alt":      "#E4DFD5",
            "zero":          "#CFC9BD",
            "legend_bg":     "#FFFFFF",
            "card_bg":       "#FFFFFF",
            "card_border":   "#E4DFD5",
            "card_text":     "#0B0D10",
            "card_subtext":  "#6A6F78",
            "card_header":   "#F9F6F0",
            "row_divider":   "#E4DFD5",
            "accent":        "#2E6FA8",
            "positive":      "#338C5B",
            "negative":      "#C64525",
        })

# ── Resign signal palettes ─────────────────────────────────────────────────────
RESIGN_PALETTE = {
    "Must Sign":         "#0E7A3A",
    "Priority RFA":      "#1254A0",
    "Locked In (Value)": "#6B21C8",
    "Fair Deal":         "#2A3A40",
    "Let Walk":          "#8B2200",
    "Buyout Candidate":  "#7B0D42",
    "Monitor":           "#2E4A54",
    "Extension Signed":  "#0D3348",
}

KINGS_SIGNAL_PALETTE = {
    "Extension Now":    "#0A5C30",
    "Lock Up":          "#0E7A3A",
    "Priority Re-sign": "#1254A0",
    "Extension Signed": "#0D3348",
    "Fair Deal":        "#2A3A40",
    "Monitor":          "#2E4A54",
    "Let Walk":         "#8B2200",
    "Buyout Candidate": "#7B0D42",
    "UFA":              "#2A1A10",
}

TEAM_NAMES = {
    "ANA": "Anaheim Ducks",       "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres",      "CGY": "Calgary Flames",
    "CAR": "Carolina Hurricanes", "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche",  "CBJ": "Columbus Blue Jackets",
    "DAL": "Dallas Stars",        "DET": "Detroit Red Wings",
    "EDM": "Edmonton Oilers",     "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings",   "MIN": "Minnesota Wild",
    "MTL": "Montréal Canadiens",  "NSH": "Nashville Predators",
    "NJD": "New Jersey Devils",   "NYI": "New York Islanders",
    "NYR": "New York Rangers",    "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers", "PIT": "Pittsburgh Penguins",
    "SEA": "Seattle Kraken",      "SJS": "San Jose Sharks",
    "STL": "St. Louis Blues",     "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs", "UTA": "Utah Hockey Club",
    "VAN": "Vancouver Canucks",   "VGK": "Vegas Golden Knights",
    "WSH": "Washington Capitals", "WPG": "Winnipeg Jets",
}

# primary = main accent (borders, highlights, chart bars)
# secondary = secondary chart bar color
TEAM_COLORS = {
    "ANA": {"primary": "#FC4C02", "secondary": "#A2AAAD"},
    "BOS": {"primary": "#FCB514", "secondary": "#1a1a2e"},
    "BUF": {"primary": "#FCB514", "secondary": "#003087"},
    "CGY": {"primary": "#C8102E", "secondary": "#F1BE48"},
    "CAR": {"primary": "#CC0000", "secondary": "#A4A9AD"},
    "CHI": {"primary": "#CF0A2C", "secondary": "#FF671B"},
    "COL": {"primary": "#6F263D", "secondary": "#236192"},
    "CBJ": {"primary": "#CE1126", "secondary": "#002654"},
    "DAL": {"primary": "#006847", "secondary": "#8F8F8C"},
    "DET": {"primary": "#CE1126", "secondary": "#FFFFFF"},
    "EDM": {"primary": "#FF4C00", "secondary": "#041E42"},
    "FLA": {"primary": "#C8102E", "secondary": "#B9975B"},
    "LAK": {"primary": "#C8A84B", "secondary": "#8A9499"},
    "MIN": {"primary": "#A6192E", "secondary": "#154734"},
    "MTL": {"primary": "#AF1E2D", "secondary": "#192168"},
    "NSH": {"primary": "#FFB81C", "secondary": "#041E42"},
    "NJD": {"primary": "#CE1126", "secondary": "#000000"},
    "NYI": {"primary": "#F47D30", "secondary": "#00539B"},
    "NYR": {"primary": "#0038A8", "secondary": "#CE1126"},
    "OTT": {"primary": "#C52032", "secondary": "#C69214"},
    "PHI": {"primary": "#F74902", "secondary": "#1a1a2e"},
    "PIT": {"primary": "#FCB514", "secondary": "#1a1a2e"},
    "SEA": {"primary": "#99D9D9", "secondary": "#001628"},
    "SJS": {"primary": "#006D75", "secondary": "#EA7200"},
    "STL": {"primary": "#003087", "secondary": "#FCB514"},
    "TBL": {"primary": "#002868", "secondary": "#FFFFFF"},
    "TOR": {"primary": "#00205B", "secondary": "#FFFFFF"},
    "UTA": {"primary": "#6CAEDF", "secondary": "#D09C47"},
    "VAN": {"primary": "#00843D", "secondary": "#00205B"},
    "VGK": {"primary": "#B4975A", "secondary": "#333F42"},
    "WSH": {"primary": "#C8102E", "secondary": "#041E42"},
    "WPG": {"primary": "#004C97", "secondary": "#AC162C"},
}

# ── Cluster constants ──────────────────────────────────────────────────────────
CLUSTER_ORDER = [
    "Elite", "Top-Line F", "Middle-Six F", "Bottom-Six F",
    "Top-Four D", "Bottom-Pair D", "Two-Way / Shutdown",
]


PERCENTILE_STATS = [
    ("ppg",       "Points/Game"),
    ("p60",       "Points/60 min"),
    ("g60",       "Goals/60 min"),
    ("toi_per_g", "TOI/Game (min)"),
    ("pp_pts",    "PP Points"),
    ("shots",     "Shots"),
    ("gp",        "Games Played"),
]

CAP_CEILING = 95_500_000

# ── CSS ────────────────────────────────────────────────────────────────────────
_FONT_LINK = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
"""

# ── Inline SVG icons (currentColor — use in st.markdown with unsafe_allow_html) ─
_ICONS = {
    "globe":     "<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round' style='display:inline;vertical-align:middle;margin-right:6px;'><circle cx='12' cy='12' r='10'/><line x1='2' y1='12' x2='22' y2='12'/><path d='M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z'/></svg>",
    "bar_chart": "<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round' style='display:inline;vertical-align:middle;margin-right:6px;'><line x1='18' y1='20' x2='18' y2='10'/><line x1='12' y1='20' x2='12' y2='4'/><line x1='6' y1='20' x2='6' y2='14'/></svg>",
    "shield":    "<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round' style='display:inline;vertical-align:middle;margin-right:6px;'><path d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'/></svg>",
    "search":    "<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round' style='display:inline;vertical-align:middle;margin-right:6px;'><circle cx='11' cy='11' r='8'/><line x1='21' y1='21' x2='16.65' y2='16.65'/></svg>",
    "scatter":   "<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round' style='display:inline;vertical-align:middle;margin-right:6px;'><circle cx='7' cy='17' r='2'/><circle cx='17' cy='7' r='2'/><circle cx='17' cy='17' r='2'/><line x1='8.7' y1='15.7' x2='15.3' y2='8.3'/><line x1='15' y1='17' x2='9' y2='17'/></svg>",
    "moon":      "<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round' style='display:inline;vertical-align:middle;margin-right:6px;'><path d='M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z'/></svg>",
    "sun":       "<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round' style='display:inline;vertical-align:middle;margin-right:6px;'><circle cx='12' cy='12' r='5'/><line x1='12' y1='1' x2='12' y2='3'/><line x1='12' y1='21' x2='12' y2='23'/><line x1='4.22' y1='4.22' x2='5.64' y2='5.64'/><line x1='18.36' y1='18.36' x2='19.78' y2='19.78'/><line x1='1' y1='12' x2='3' y2='12'/><line x1='21' y1='12' x2='23' y2='12'/><line x1='4.22' y1='19.78' x2='5.64' y2='18.36'/><line x1='18.36' y1='5.64' x2='19.78' y2='4.22'/></svg>",
    "calendar":  "<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round' style='display:inline;vertical-align:middle;margin-right:6px;'><rect x='3' y='4' width='18' height='18'/><line x1='16' y1='2' x2='16' y2='6'/><line x1='8' y1='2' x2='8' y2='6'/><line x1='3' y1='10' x2='21' y2='10'/></svg>",
    "diamond":   "<svg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='display:inline;vertical-align:middle;margin-right:4px;'><polygon points='12 2 22 12 12 22 2 12'/></svg>",
    "refresh":   "<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round' style='display:inline;vertical-align:middle;margin-right:6px;'><polyline points='23 4 23 10 17 10'/><polyline points='1 20 1 14 7 14'/><path d='M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15'/></svg>",
}

_DARK_CSS  = """<style>
  :root {
    --font: 'Inter', sans-serif; --font-mono: 'JetBrains Mono', monospace;
    --ink:#0E1013; --panel:#1A1E24; --panel-2:#14171C;
    --line:#262B33; --line-2:#343A44;
    --text:#F2EEE5; --text-2:#B8BDC5; --muted:#8A8F99; --muted-2:#5A5F69;
    --accent:#4FD1C5; --accent-2:#A78BFA; --gold:#E5B664;
    --green:#7DD87A; --red:#F07A7A;
  }
  footer { visibility: hidden; }
  .block-container { padding-top: 2.5rem !important; padding-bottom: 2rem !important; }

  /* ── Page surfaces ── */
  .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] { background-color: #0E1013 !important; }

  /* ── Base text color (overrides Streamlit's light-theme body color) ── */
  /* Without this, any element without an explicit color rule inherits
     Streamlit's default rgb(28,28,28) — near-black on our dark bg = invisible. */
  body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
  [data-testid="stSidebar"], [data-testid="stMainBlockContainer"] {
    color: #F2EEE5;
  }

  /* ── Base typography ── */
  html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
  p, label, [data-testid="stMarkdownContainer"], [class*="css"] { font-family: 'Inter', sans-serif !important; }
  h1, h2, h3,
  [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {
      font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important;
      line-height: 1.15 !important; letter-spacing: -0.01em !important; color: #F2EEE5 !important;
  }

  /* ── Metric tiles ── */
  [data-testid="stMetric"] {
      background: #1A1E24 !important; border: 1px solid #262B33 !important;
      border-radius: 10px !important; padding: 18px 20px !important; box-shadow: none !important;
  }
  [data-testid="stMetricLabel"] {
      font-family: 'JetBrains Mono', monospace !important; font-size: 0.72rem !important;
      letter-spacing: 0.08em !important; text-transform: uppercase !important; color: #8A8F99 !important;
  }
  [data-testid="stMetricValue"] {
      font-family: 'Space Grotesk', sans-serif !important; font-size: 1.9rem !important;
      line-height: 1.0 !important; letter-spacing: -0.01em !important; font-weight: 500 !important; color: #F2EEE5 !important;
  }
  [data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', monospace !important; font-size: 0.78rem !important; }

  /* ── Tabs ── */
  [data-baseweb="tab-list"] {
      gap: 4px !important; background: transparent !important; padding-bottom: 0 !important;
      margin-bottom: 24px !important; border-bottom: 1px solid #262B33 !important;
  }
  [data-baseweb="tab"] {
      font-family: 'Space Grotesk', sans-serif !important; font-size: 0.95rem !important;
      letter-spacing: 0.01em !important; text-transform: none !important; font-weight: 500 !important;
      padding: 0 18px !important; height: 46px !important; border-radius: 0 !important;
      background: transparent !important; border-bottom: 2px solid transparent !important;
      margin-bottom: -1px !important; color: #8A8F99 !important;
  }
  [data-baseweb="tab"]:hover { color: #F2EEE5 !important; }
  [aria-selected="true"][data-baseweb="tab"] {
      color: #F2EEE5 !important; border-bottom: 2px solid #4FD1C5 !important;
      background: transparent !important;
      font-style: normal !important; font-family: 'Space Grotesk', sans-serif !important;
  }

  /* ── Cards ── */
  .player-card { border-radius: 12px; padding: 20px 22px; margin-bottom: 12px; background: #1A1E24; border: 1px solid #262B33; }
  .kings-card  { border-radius: 10px; padding: 18px 22px; margin-bottom: 8px; background: #1A1E24; border: 1px solid #262B33; transition: border-color 140ms, background 140ms; }
  .kings-card:hover { background: #14171C; border-color: #4FD1C5; }

  /* ── Typography classes ── */
  .stat-label { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; color: #8A8F99; }
  .stat-value { font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; color: #F2EEE5; font-feature-settings: "tnum" 1; }
  .delta-pos  { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; color: #7DD87A; font-feature-settings: "tnum" 1; }
  .delta-neg  { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; color: #F07A7A; font-feature-settings: "tnum" 1; }
  .pct-pos    { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #7DD87A; }
  .pct-neg    { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #F07A7A; }
  .section-header { font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 500; letter-spacing: -0.01em; line-height: 1.15; margin-bottom: 8px; color: #F2EEE5; }
  .group-label { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; margin: 24px 0 12px; padding-left: 10px; border-left-width: 2px; border-left-style: solid; color: #8A8F99; }
  .signal-badge { display: inline-block; padding: 3px 10px; border-radius: 999px; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: #fff !important; }
  .kings-gold { color: #E5B664; font-weight: 600; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"], section[data-testid="stSidebar"] { background-color: #0E1013 !important; border-right: 1px solid #262B33 !important; }
  [data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] { color: #F2EEE5 !important; }
  [data-testid="stSidebar"] .stCaptionContainer p { color: #8A8F99 !important; }

  /* ── Sidebar expand / collapse buttons (Streamlit hardcodes icon to near-black) ── */
  [data-testid="stExpandSidebarButton"],
  [data-testid="stSidebarCollapseButton"],
  [data-testid="stExpandSidebarButton"] button,
  [data-testid="stSidebarCollapseButton"] button { background: #14171C !important; border: 1px solid #262B33 !important; border-radius: 8px !important; }
  [data-testid="stExpandSidebarButton"] *,
  [data-testid="stSidebarCollapseButton"] *,
  [data-testid="stIconMaterial"] { color: #F2EEE5 !important; fill: #F2EEE5 !important; }

  /* ── Expanders ── */
  [data-testid="stExpander"],
  [data-testid="stExpander"] > *,
  [data-testid="stExpander"] details,
  [data-testid="stExpander"] details > div { border: none !important; outline: none !important; box-shadow: none !important; }
  [data-testid="stExpander"] { background-color: #1A1E24 !important; border-radius: 10px !important; }
  [data-testid="stExpander"] details { background-color: #1A1E24 !important; }
  [data-testid="stExpander"] details summary { background-color: #14171C !important; color: #F2EEE5 !important; border: none !important; }
  [data-testid="stExpander"] details > div { background-color: #1A1E24 !important; }
  details summary p { font-family: 'Inter', sans-serif !important; font-size: 0.85rem !important; letter-spacing: 0.04em !important; }

  /* ── Captions & text ── */
  [data-testid="stCaptionContainer"] p { font-family: 'JetBrains Mono', monospace !important; font-size: 0.78rem !important; color: #8A8F99 !important; letter-spacing: 0.04em !important; }
  [data-testid="stMarkdownContainer"] p { color: #F2EEE5 !important; }
  [data-testid="stMainBlockContainer"] p, [data-testid="stMainBlockContainer"] li { color: #F2EEE5 !important; }
  .stCaptionContainer p { color: #8A8F99 !important; }
  [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color: #8A8F99 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.72rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }

  /* ── Inputs ── */
  input, [data-baseweb="input"] input {
      border-radius: 8px !important; font-family: 'Inter', sans-serif !important;
      background: #1A1E24 !important; border-color: #262B33 !important; color: #F2EEE5 !important;
  }
  input:focus, [data-baseweb="input"] input:focus { border-color: #4FD1C5 !important; box-shadow: none !important; }
  ::-webkit-scrollbar { width: 10px; height: 10px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { border-radius: 10px; background: #262B33; }
  ::-webkit-scrollbar-thumb:hover { background: #343A44; }
  hr { margin: 24px 0 !important; border-color: #262B33 !important; }

  /* ── Buttons ── */
  [data-testid="stButton"] button {
      background: #1A1E24 !important; color: #F2EEE5 !important;
      border: 1px solid #262B33 !important; border-radius: 8px !important;
      font-family: 'Space Grotesk', sans-serif !important; font-weight: 500 !important;
      padding: 6px 16px !important;
      transition: border-color 140ms, color 140ms !important;
  }
  [data-testid="stButton"] button:hover { border-color: #4FD1C5 !important; color: #4FD1C5 !important; }

  /* ── Selectbox ── */
  [data-baseweb="select"] > div { background: #1A1E24 !important; border-color: #262B33 !important; border-radius: 8px !important; }
  [data-baseweb="select"] span, [data-baseweb="select"] div { color: #F2EEE5 !important; }
  [data-baseweb="select"] svg { fill: #8A8F99 !important; }
  [data-baseweb="popover"] { background: #1A1E24 !important; }
  [data-baseweb="menu"] { background: #1A1E24 !important; border: 1px solid #262B33 !important; border-radius: 8px !important; }
  [data-baseweb="menu-item"], [role="option"] { color: #F2EEE5 !important; background: #1A1E24 !important; }
  [data-baseweb="menu-item"]:hover, [role="option"]:hover { background: #14171C !important; }

  /* ── Slider ── */
  [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child { background: #262B33 !important; }
  [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background: #4FD1C5 !important; border-color: #4FD1C5 !important; }
  [data-testid="stSlider"] [data-testid="stTickBarMin"],
  [data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #8A8F99 !important; }

  /* ── Checkbox ── text only; let Streamlit handle the box visual ── */
  [data-testid="stCheckbox"] label p,
  [data-testid="stCheckbox"] label > div { color: #F2EEE5 !important; font-family: 'Inter', sans-serif !important; font-size: 0.82rem !important; letter-spacing: 0 !important; text-transform: none !important; }

  /* ── Radio ── */
  [data-testid="stRadio"] label,
  [data-testid="stRadio"] label *,
  [data-baseweb="radio"],
  [data-baseweb="radio"] * { color: #F2EEE5 !important; }

  /* ── Widget labels (Streamlit uses classes not testids for these) ── */
  .stSelectbox label, .stMultiSelect label, .stTextInput label,
  .stNumberInput label, .stCheckbox label, .stRadio label,
  .stSlider label, .stDateInput label, .stTimeInput label,
  .stTextArea label, [data-testid="stWidgetLabel"],
  [data-testid="stWidgetLabel"] * {
    color: #B8BDC5 !important;
  }

  /* ── Selectbox / multiselect rendered values + popover options ── */
  [data-baseweb="select"] *, [data-baseweb="popover"] *,
  [data-baseweb="menu"] *, ul[role="listbox"] *, [role="option"] {
    color: #F2EEE5 !important;
  }
  [data-baseweb="select"] svg, [role="option"] svg { fill: #8A8F99 !important; }

  /* ── Multiselect chips (fallback — we now use checkboxes for Role Cluster) ── */
  [data-baseweb="tag"] { background: #1B1F26 !important; border: 1px solid #343A44 !important; color: #F2EEE5 !important; }
  [data-baseweb="tag"] span, [data-baseweb="tag"] svg { color: #F2EEE5 !important; fill: #F2EEE5 !important; }

  /* ── Number input ── */
  [data-testid="stNumberInput"] input {
      background: #1A1E24 !important; color: #F2EEE5 !important;
      border-color: #262B33 !important; border-radius: 8px !important;
      font-family: 'JetBrains Mono', monospace !important;
  }
  [data-testid="stNumberInput"] input:focus { border-color: #4FD1C5 !important; box-shadow: none !important; }
  [data-testid="stNumberInput"] button { background: #1A1E24 !important; border-color: #262B33 !important; color: #8A8F99 !important; }
  [data-testid="stNumberInput"] button svg { fill: #8A8F99 !important; }
  [data-testid="stNumberInput"] button:hover { border-color: #4FD1C5 !important; color: #4FD1C5 !important; }
  [data-testid="stNumberInput"] button:hover svg { fill: #4FD1C5 !important; }

  /* ── Verdict/signal pills (design system) ── */
  .verdict-pill { display:inline-block; padding:3px 10px; border-radius:999px; font-family:'JetBrains Mono',monospace; font-size:0.7rem; letter-spacing:0.08em; text-transform:uppercase; }
  .verdict-pill.underpaid { background:rgba(125,216,122,0.13); color:#7DD87A; border:1px solid rgba(125,216,122,0.35); }
  .verdict-pill.overpaid  { background:rgba(240,122,122,0.13); color:#F07A7A; border:1px solid rgba(240,122,122,0.35); }
  .verdict-pill.fair      { background:rgba(138,143,153,0.13); color:#8A8F99; border:1px solid rgba(138,143,153,0.35); }
  .verdict-pill.ice       { background:rgba(79,209,197,0.13);  color:#4FD1C5; border:1px solid rgba(79,209,197,0.35); }

  /* ── Brand header ── */
  .rink-brand { display:flex; align-items:baseline; gap:14px; padding:0 0 12px; border-bottom:1px solid #262B33; margin-bottom:24px; }
  .rink-brand .logo { font-family:'Space Grotesk',sans-serif; font-size:1.5rem; font-weight:700; letter-spacing:0.08em; color:#F2EEE5; }
  .rink-brand .logo .dot { color:#4FD1C5; }
  .rink-brand .tag { font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#8A8F99; text-transform:uppercase; letter-spacing:0.1em; }
  .rink-brand .season { margin-left:auto; font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#8A8F99; letter-spacing:0.08em; }

  /* ── Footer ── */
  .rink-footer { margin-top:3rem; padding-top:1.25rem; border-top:1px solid #262B33; color:#8A8F99; font-family:'JetBrains Mono',monospace; font-size:0.72rem; letter-spacing:0.06em; display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap; }
  .rink-footer .dot { color:#4FD1C5; }

  /* ── Stat card (generic content tile) ── */
  .rink-card { background:#1A1E24; border:1px solid #262B33; border-radius:12px; padding:18px 20px; margin-bottom:10px; }
  .rink-card .label { font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#8A8F99; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px; }
  .rink-card .value { font-family:'Space Grotesk',sans-serif; font-size:1.45rem; color:#F2EEE5; font-weight:500; }
  .rink-card .sub   { font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#8A8F99; margin-top:8px; letter-spacing:0.04em; }

  /* ── Player hero ── */
  .player-hero { background:linear-gradient(135deg,#1A1E24 0%,#14171C 100%); border:1px solid #262B33; border-radius:14px; padding:28px 32px; margin-bottom:20px; }
  .player-hero .name { font-family:'Space Grotesk',sans-serif; font-size:2.6rem; font-weight:600; color:#F2EEE5; line-height:1.05; margin-bottom:8px; letter-spacing:-0.015em; }
  .player-hero .meta { font-family:'JetBrains Mono',monospace; font-size:0.82rem; color:#8A8F99; letter-spacing:0.06em; text-transform:uppercase; }

  /* ── Dataframe ── */
  [data-testid="stDataFrameContainer"], [data-testid="stDataFrame"],
  .stDataFrame { filter: invert(0.88) hue-rotate(180deg) !important; background: #f5f5f2 !important; }

  /* ── Tooltip card (value driver) ── */
  .vd-tip { font-family: 'Inter', sans-serif !important; }

  /* ── Tab icons (inactive gray / active light) ── */
  [data-baseweb="tab"]:nth-child(1)::before { content: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%238A8F99' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='10'/><line x1='2' y1='12' x2='22' y2='12'/><path d='M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z'/></svg>"); display:inline-block; margin-right:7px; vertical-align:-3px; }
  [data-baseweb="tab"]:nth-child(2)::before { content: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%238A8F99' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><line x1='18' y1='20' x2='18' y2='10'/><line x1='12' y1='20' x2='12' y2='4'/><line x1='6' y1='20' x2='6' y2='14'/></svg>"); display:inline-block; margin-right:7px; vertical-align:-3px; }
  [data-baseweb="tab"]:nth-child(3)::before { content: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%238A8F99' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><path d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'/></svg>"); display:inline-block; margin-right:7px; vertical-align:-3px; }
  [data-baseweb="tab"]:nth-child(4)::before { content: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%238A8F99' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><circle cx='11' cy='11' r='8'/><line x1='21' y1='21' x2='16.65' y2='16.65'/></svg>"); display:inline-block; margin-right:7px; vertical-align:-3px; }
  [data-baseweb="tab"]:nth-child(5)::before { content: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%238A8F99' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><circle cx='7' cy='17' r='2'/><circle cx='17' cy='7' r='2'/><circle cx='17' cy='17' r='2'/><line x1='8.7' y1='15.7' x2='15.3' y2='8.3'/><line x1='15' y1='17' x2='9' y2='17'/></svg>"); display:inline-block; margin-right:7px; vertical-align:-3px; }
  [aria-selected="true"][data-baseweb="tab"]:nth-child(1)::before { content: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%234FD1C5' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='10'/><line x1='2' y1='12' x2='22' y2='12'/><path d='M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z'/></svg>"); }
  [aria-selected="true"][data-baseweb="tab"]:nth-child(2)::before { content: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%234FD1C5' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><line x1='18' y1='20' x2='18' y2='10'/><line x1='12' y1='20' x2='12' y2='4'/><line x1='6' y1='20' x2='6' y2='14'/></svg>"); }
  [aria-selected="true"][data-baseweb="tab"]:nth-child(3)::before { content: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%234FD1C5' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><path d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'/></svg>"); }
  [aria-selected="true"][data-baseweb="tab"]:nth-child(4)::before { content: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%234FD1C5' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><circle cx='11' cy='11' r='8'/><line x1='21' y1='21' x2='16.65' y2='16.65'/></svg>"); }
  [aria-selected="true"][data-baseweb="tab"]:nth-child(5)::before { content: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%234FD1C5' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><circle cx='7' cy='17' r='2'/><circle cx='17' cy='7' r='2'/><circle cx='17' cy='17' r='2'/><line x1='8.7' y1='15.7' x2='15.3' y2='8.3'/><line x1='15' y1='17' x2='9' y2='17'/></svg>"); }

  /* ── Theme toggle button: moon icon (dark mode active) ── */
  [data-testid="stSidebar"] [data-testid="stButton"]:first-child button::before { content: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%234FD1C5' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><path d='M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z'/></svg>"); display:inline-block; margin-right:8px; vertical-align:-2px; }

  /* ══════════ Word-break normalization ══════════
     Prevents text from rendering one character per line when Streamlit
     columns get squeezed. Browsers default to break-all in some locales;
     this pins to word boundaries everywhere inside the app. */
  [data-testid="stMainBlockContainer"] *,
  [data-testid="stMetric"] *,
  .rink-card, .rink-card *, .player-card, .player-card *,
  .kings-card, .kings-card *, .rink-brand, .rink-brand * {
    word-break: normal !important;
    overflow-wrap: break-word !important;
    hyphens: manual !important;
  }
  .stat-value, .delta-pos, .delta-neg, .pct-pos, .pct-neg,
  [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
    white-space: nowrap !important;
    word-break: keep-all !important;
  }

  /* Safe-area padding for iOS (home bar / notch) */
  .stApp, [data-testid="stAppViewContainer"] {
    padding-bottom: env(safe-area-inset-bottom) !important;
    padding-left: env(safe-area-inset-left) !important;
    padding-right: env(safe-area-inset-right) !important;
  }

  /* ══════════ Responsive breakpoints (from RINK-1 styles.css) ══════════
     Port of the @media rules shipped in the RINK-1 design bundle
     (.design-ref/project/styles.css). Selectors adapted from the
     React class names (.kpi-strip, .lb-row, .player-hero, .comps-grid,
     .split, .container) to their Streamlit equivalents
     ([data-testid=stHorizontalBlock], [data-testid=stColumn],
      .block-container). Breakpoints kept at 900/720/480 to match. */

  /* @900px in RINK-1: kpi-strip 4->2 col, split 1.4fr|1fr -> 1fr, tabs drawer, brand-meta hidden */
  @media (max-width: 900px) {
    /* Streamlit st.columns(N) stays N-wide by default; this reflows them. */
    [data-testid="stHorizontalBlock"] {
      flex-wrap: wrap !important;
      gap: 12px !important;
      row-gap: 12px !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
      flex: 1 1 220px !important;   /* 2-col layout when viewport allows */
      min-width: 0 !important;
      width: auto !important;
    }
    /* RINK-1 line 188: .brand-meta { display: none; } */
    .rink-brand .season { margin-left: 0; flex-basis: 100%; font-size: 0.68rem; opacity: 0.8; }
    .rink-brand { flex-wrap: wrap; row-gap: 4px; }
    .rink-footer { flex-direction: column; gap: 6px; }
    /* RINK-1 line 167: .tabs { display: none; } then hamburger drawer.
       Streamlit can't swap to a drawer; horizontal scroll is the best
       equivalent that preserves tab switching on touch. */
    [data-baseweb="tab-list"] { overflow-x: auto; overflow-y: hidden; flex-wrap: nowrap !important; -webkit-overflow-scrolling: touch; scrollbar-width: none; }
    [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
    [data-baseweb="tab"] { padding: 0 14px !important; font-size: 0.9rem !important; height: 44px !important; flex-shrink: 0 !important; }
    [data-baseweb="tab"]::before { display: none !important; margin-right: 0 !important; }
  }

  /* @720px: phone. Reposition st.tabs as a fixed bottom nav. */
  @media (max-width: 720px) {
    /* RINK-1 line 95: .container { padding: 0 18px; } + bottom-nav clearance */
    .block-container { padding-top: 1rem !important; padding-bottom: calc(84px + env(safe-area-inset-bottom)) !important; padding-left: 18px !important; padding-right: 18px !important; }
    .rink-brand { padding: 0 0 10px; margin-bottom: 18px; }

    /* ── Bottom tab nav (replaces st.tabs top strip) ── */
    [data-baseweb="tab-list"] {
      position: fixed !important;
      bottom: 0 !important; left: 0 !important; right: 0 !important;
      z-index: 100 !important;
      background: #0E1013 !important;
      border-top: 1px solid #262B33 !important;
      border-bottom: none !important;
      box-shadow: 0 -8px 24px rgba(0,0,0,0.4) !important;
      height: calc(62px + env(safe-area-inset-bottom)) !important;
      padding: 0 4px env(safe-area-inset-bottom) !important;
      margin: 0 !important;
      display: flex !important;
      flex-direction: row !important;
      flex-wrap: nowrap !important;
      justify-content: space-around !important;
      align-items: stretch !important;
      gap: 0 !important;
      overflow: visible !important;
    }
    [data-baseweb="tab"] {
      flex: 1 1 0 !important;
      min-width: 0 !important;
      height: 62px !important;
      padding: 6px 2px 4px !important;
      font-family: 'Inter', sans-serif !important;
      font-size: 0.62rem !important;
      font-weight: 500 !important;
      letter-spacing: 0.04em !important;
      text-transform: uppercase !important;
      display: flex !important;
      flex-direction: column !important;
      align-items: center !important;
      justify-content: center !important;
      gap: 5px !important;
      color: #8A8F99 !important;
      border-bottom: none !important;
      border-top: 2px solid transparent !important;
      border-radius: 0 !important;
      background: transparent !important;
      text-overflow: ellipsis !important;
      overflow: hidden !important;
      white-space: nowrap !important;
    }
    /* Restore tab icons — stacked above label, enlarged */
    [data-baseweb="tab"]::before {
      display: block !important;
      margin: 0 !important;
      transform: scale(1.3) !important;
      vertical-align: baseline !important;
      line-height: 0 !important;
    }
    [aria-selected="true"][data-baseweb="tab"] {
      color: #4FD1C5 !important;
      border-top: 2px solid #4FD1C5 !important;
      border-bottom: none !important;
      background: rgba(79,209,197,0.07) !important;
      font-family: 'Inter', sans-serif !important;
      font-style: normal !important;
    }

    /* Column reflow — 2 per row at phone width */
    [data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex: 1 1 160px !important; }
    [data-testid="stMetric"] { padding: 14px 16px !important; }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.66rem !important; letter-spacing: 0.1em !important; }
    [data-testid="stMetricDelta"] { font-size: 0.72rem !important; }
    .player-hero { padding: 22px 20px; border-radius: 12px; }
    .player-hero .name { font-size: 1.9rem; line-height: 1.1; }
    .player-hero .meta { font-size: 0.74rem; }
    .rink-card { padding: 14px 16px; }
    .rink-card .value { font-size: 1.25rem; }
    .player-card, .kings-card { padding: 14px 16px; border-radius: 10px; }
    .lb-bar-wrap, .lb-pct { display: none !important; }
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.3rem !important; }
    h3 { font-size: 1.05rem !important; }
    .js-plotly-plot, .plot-container { width: 100% !important; }
    /* Sidebar collapse nudge on mobile — Streamlit's default is open. Hide the
       toggle-arrow shadow that lingers over content. */
    section[data-testid="stSidebar"] { z-index: 99 !important; }
  }

  /* @480px in RINK-1: kpi-value 26px. Phone-portrait small screens. */
  @media (max-width: 480px) {
    .block-container { padding-left: 14px !important; padding-right: 14px !important; }
    .rink-brand .logo { font-size: 1.15rem; }
    .rink-brand .tag { font-size: 0.64rem; }
    [data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex: 1 1 100% !important; }
    [data-testid="stMetric"] { padding: 12px 14px !important; }
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }  /* RINK-1: 26px ≈ 1.4rem */
    [data-testid="stMetricLabel"] { font-size: 0.62rem !important; }
    .player-hero { padding: 18px 18px; }
    .player-hero .name { font-size: 1.55rem; }
    .signal-badge, .verdict-pill { font-size: 0.62rem; padding: 2px 7px; }
  }

  /* Bottom nav: short labels (Overview/Leaders/Teams/Search/Insights)
     scale smoothly. With 8-char max labels, a floor of 0.58rem (~9px)
     is comfortable at 360px+, still no truncation. */
  @media (max-width: 720px) {
    [data-baseweb="tab"] {
      font-size: clamp(0.58rem, 1.4vw, 0.78rem) !important;
      letter-spacing: 0 !important;
      padding: 0 4px !important;
      gap: 4px !important;
      text-transform: none !important;
    }
    [data-baseweb="tab"]::before { transform: scale(1.15) !important; margin: 0 !important; }
  }
  /* Very narrow phones still fall back to icon-only. */
  @media (max-width: 340px) {
    [data-baseweb="tab"] { font-size: 0 !important; gap: 0 !important; padding: 0 !important; }
    [data-baseweb="tab"]::before { transform: scale(1.6) !important; }
  }
</style>"""
_LIGHT_CSS = """<style>
  :root {
    --font: 'Inter', sans-serif; --font-mono: 'JetBrains Mono', monospace;
    --ink:#F5F2EC; --panel:#FFFFFF; --panel-2:#F9F6F0;
    --line:#E4DFD5; --line-2:#CFC9BD;
    --text:#0B0D10; --text-2:#2A2D33; --muted:#6A6F78; --muted-2:#9EA3AB;
    --green:#338C5B; --red:#C64525; --ice:#2E6FA8;
  }
  footer { visibility: hidden; }
  .block-container { padding-top: 2.5rem !important; padding-bottom: 2rem !important; }

  /* ── Page surfaces ── */
  .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] { background-color: #F5F2EC !important; }

  /* ── Base text color ── */
  body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
  [data-testid="stSidebar"], [data-testid="stMainBlockContainer"] {
    color: #0B0D10;
  }

  /* ── Base typography ── */
  html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
  p, label, [data-testid="stMarkdownContainer"], [class*="css"] { font-family: 'Inter', sans-serif !important; }
  h1, h2, h3,
  [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {
      font-family: 'Space Grotesk', sans-serif !important; font-weight: 500 !important;
      font-variation-settings: "opsz" 144 !important;
      line-height: 1.06 !important; letter-spacing: -0.02em !important; color: #0B0D10 !important;
  }

  /* ── Metric tiles ── */
  [data-testid="stMetric"] {
      background: transparent !important; border: none !important;
      border-radius: 0 !important; padding: 16px 2px 12px !important; box-shadow: none !important;
  }
  [data-testid="stMetricLabel"] {
      font-family: 'JetBrains Mono', monospace !important; font-size: 0.66rem !important;
      letter-spacing: 0.14em !important; text-transform: uppercase !important; color: #6A6F78 !important;
  }
  [data-testid="stMetricValue"] {
      font-family: 'Space Grotesk', sans-serif !important; font-size: 2.5rem !important;
      font-variation-settings: "opsz" 144 !important;
      line-height: 1.0 !important; letter-spacing: -0.02em !important; font-weight: 500 !important; color: #0B0D10 !important;
  }
  [data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', monospace !important; font-size: 0.78rem !important; }

  /* ── Tabs ── */
  [data-baseweb="tab-list"] {
      gap: 0 !important; background: transparent !important; padding-bottom: 0 !important;
      margin-bottom: 24px !important; border-bottom: 1px solid #E4DFD5 !important;
  }
  [data-baseweb="tab"] {
      font-family: 'Inter', sans-serif !important; font-size: 0.82rem !important;
      letter-spacing: 0.1em !important; text-transform: uppercase !important;
      padding: 12px 22px !important; border-radius: 0 !important;
      background: transparent !important; border-bottom: 2px solid transparent !important;
      margin-bottom: -1px !important; color: #6A6F78 !important;
  }
  [data-baseweb="tab"]:hover { color: #2A2D33 !important; }
  [aria-selected="true"][data-baseweb="tab"] {
      color: #0B0D10 !important; border-bottom: 2px solid #0B0D10 !important;
      font-style: italic !important; font-family: 'Space Grotesk', sans-serif !important;
      font-variation-settings: "opsz" 14 !important;
  }

  /* ── Cards ── */
  .player-card { border-radius: 8px; padding: 20px 24px; margin-bottom: 14px; background: #FFFFFF; border: 1px solid #E4DFD5; }
  .kings-card  { border-radius: 4px; padding: 18px 22px; margin-bottom: 6px; background: #FFFFFF; border: 1px solid #E4DFD5; transition: border-color 140ms, background 140ms; }
  .kings-card:hover { background: #F9F6F0; border-color: #CFC9BD; }

  /* ── Typography classes ── */
  .stat-label { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em; color: #6A6F78; }
  .stat-value { font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; color: #0B0D10; font-feature-settings: "tnum" 1; }
  .delta-pos  { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; color: #338C5B; font-feature-settings: "tnum" 1; }
  .delta-neg  { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; color: #C64525; font-feature-settings: "tnum" 1; }
  .pct-pos    { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #338C5B; }
  .pct-neg    { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #C64525; }
  .section-header { font-family: 'Space Grotesk', sans-serif; font-variation-settings: "opsz" 144; font-size: 1.8rem; font-weight: 500; letter-spacing: -0.015em; line-height: 1.1; margin-bottom: 8px; color: #0B0D10; }
  .group-label { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 500; letter-spacing: 0.14em; text-transform: uppercase; margin: 24px 0 12px; padding-left: 10px; border-left-width: 2px; border-left-style: solid; color: #6A6F78; }
  .signal-badge { display: inline-block; padding: 3px 8px; border-radius: 3px; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; color: #fff !important; }
  .kings-gold { color: #2E6FA8; font-weight: 600; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"], section[data-testid="stSidebar"] { background-color: #F5F2EC !important; border-right: 1px solid #E4DFD5 !important; }
  [data-testid="stSidebar"] * { color: #0B0D10 !important; }
  [data-testid="stSidebar"] .stCaptionContainer p { color: #6A6F78 !important; }

  /* ── Sidebar expand / collapse buttons ── */
  [data-testid="stExpandSidebarButton"],
  [data-testid="stSidebarCollapseButton"],
  [data-testid="stExpandSidebarButton"] button,
  [data-testid="stSidebarCollapseButton"] button { background: #FFFFFF !important; border: 1px solid #E4DFD5 !important; border-radius: 8px !important; }
  [data-testid="stExpandSidebarButton"] *,
  [data-testid="stSidebarCollapseButton"] *,
  [data-testid="stIconMaterial"] { color: #0B0D10 !important; fill: #0B0D10 !important; }

  /* ── Expanders ── */
  [data-testid="stExpander"] { border-radius: 8px !important; border: 1px solid #E4DFD5 !important; background-color: #FFFFFF !important; }
  [data-testid="stExpander"] details { background-color: #FFFFFF !important; }
  [data-testid="stExpander"] details summary { background-color: #F9F6F0 !important; color: #0B0D10 !important; border: none !important; }
  details summary p { font-family: 'Inter', sans-serif !important; font-size: 0.85rem !important; letter-spacing: 0.04em !important; color: #0B0D10 !important; }

  /* ── Flip inline dark backgrounds to light (legacy fallbacks) ── */
  [style*="background:#1A1E24"] { background: #FFFFFF !important; }
  [style*="background: #1A1E24"] { background: #FFFFFF !important; }
  [style*="background:#0E1013"] { background: #F5F2EC !important; }
  [style*="border:1px solid #262B33"] { border-color: #E4DFD5 !important; }
  [style*="border: 1px solid #262B33"] { border-color: #E4DFD5 !important; }

  /* ── Captions & text ── */
  [data-testid="stCaptionContainer"] p { font-family: 'JetBrains Mono', monospace !important; font-size: 0.78rem !important; color: #6A6F78 !important; letter-spacing: 0.04em !important; }
  [data-testid="stMarkdownContainer"] p { color: #0B0D10 !important; }
  [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 { color: #0B0D10 !important; }
  [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color: #6A6F78 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.72rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; }
  .stCaptionContainer p { color: #6A6F78 !important; }

  /* ── Inputs ── */
  input, [data-baseweb="input"] input {
      border-radius: 4px !important; font-family: 'Inter', sans-serif !important;
      background: #FFFFFF !important; border-color: #E4DFD5 !important; color: #0B0D10 !important;
  }
  input:focus, [data-baseweb="input"] input:focus { border-color: #CFC9BD !important; }
  ::-webkit-scrollbar { width: 10px; height: 10px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { border-radius: 10px; background: #E4DFD5; }
  ::-webkit-scrollbar-thumb:hover { background: #CFC9BD; }
  hr { margin: 24px 0 !important; border-color: #E4DFD5 !important; }

  /* ── Buttons ── */
  [data-testid="stButton"] button {
      background: #FFFFFF !important; color: #0B0D10 !important;
      border: 1px solid #E4DFD5 !important; border-radius: 4px !important;
      font-family: 'Inter', sans-serif !important; font-weight: 500 !important;
      transition: border-color 140ms, background 140ms !important;
  }
  [data-testid="stButton"] button:hover { background: #F9F6F0 !important; border-color: #CFC9BD !important; }

  /* ── Selectbox ── */
  [data-baseweb="select"] > div { background: #FFFFFF !important; border-color: #E4DFD5 !important; border-radius: 4px !important; }
  [data-baseweb="select"] span { color: #0B0D10 !important; }
  [data-baseweb="popover"] { background: #FFFFFF !important; }
  [data-baseweb="menu"] { background: #FFFFFF !important; border: 1px solid #E4DFD5 !important; border-radius: 4px !important; }
  [data-baseweb="menu-item"], [role="option"] { color: #0B0D10 !important; background: #FFFFFF !important; }
  [data-baseweb="menu-item"]:hover, [role="option"]:hover { background: #F9F6F0 !important; }

  /* ── Slider ── */
  [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child { background: #E4DFD5 !important; }
  [data-testid="stSlider"] [data-testid="stTickBarMin"],
  [data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #6A6F78 !important; }

  /* ── Checkbox ── text only; let Streamlit handle the box visual ── */
  [data-testid="stCheckbox"] label p,
  [data-testid="stCheckbox"] label > div { color: #0B0D10 !important; font-family: 'Inter', sans-serif !important; font-size: 0.82rem !important; letter-spacing: 0 !important; text-transform: none !important; }

  /* ── Radio ── */
  [data-testid="stRadio"] label,
  [data-testid="stRadio"] label *,
  [data-baseweb="radio"],
  [data-baseweb="radio"] * { color: #0B0D10 !important; }

  /* ── Widget labels ── */
  .stSelectbox label, .stMultiSelect label, .stTextInput label,
  .stNumberInput label, .stCheckbox label, .stRadio label,
  .stSlider label, .stDateInput label, .stTimeInput label,
  .stTextArea label, [data-testid="stWidgetLabel"],
  [data-testid="stWidgetLabel"] * {
    color: #2A2D33 !important;
  }

  /* ── Selectbox / multiselect rendered values + popover options ── */
  [data-baseweb="select"] *, [data-baseweb="popover"] *,
  [data-baseweb="menu"] *, ul[role="listbox"] *, [role="option"] {
    color: #0B0D10 !important;
  }
  [data-baseweb="select"] svg, [role="option"] svg { fill: #6A6F78 !important; }

  /* ── Multiselect chips ── */
  [data-baseweb="tag"] { background: #F9F6F0 !important; border: 1px solid #E4DFD5 !important; color: #0B0D10 !important; }
  [data-baseweb="tag"] span, [data-baseweb="tag"] svg { color: #0B0D10 !important; fill: #0B0D10 !important; }

  /* ── Number input ── */
  [data-testid="stNumberInput"] input {
      background: #FFFFFF !important; color: #0B0D10 !important;
      border-color: #E4DFD5 !important; border-radius: 4px !important;
      font-family: 'JetBrains Mono', monospace !important;
  }
  [data-testid="stNumberInput"] input:focus { border-color: #CFC9BD !important; box-shadow: none !important; }
  [data-testid="stNumberInput"] button { background: #FFFFFF !important; border-color: #E4DFD5 !important; }
  [data-testid="stNumberInput"] button svg { fill: #6A6F78 !important; }
  [data-testid="stNumberInput"] button:hover { border-color: #CFC9BD !important; }
  [data-testid="stNumberInput"] button:hover svg { fill: #0B0D10 !important; }

  /* ── Tooltip card (value driver) ── */
  .vd-tip { font-family: 'Inter', sans-serif !important; }

  /* ── Tab icons (inactive gray / active navy) ── */
  [data-baseweb="tab"]:nth-child(1)::before { content: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%236A6F78' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='10'/><line x1='2' y1='12' x2='22' y2='12'/><path d='M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z'/></svg>"); display:inline-block; margin-right:7px; vertical-align:-3px; }
  [data-baseweb="tab"]:nth-child(2)::before { content: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%236A6F78' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><line x1='18' y1='20' x2='18' y2='10'/><line x1='12' y1='20' x2='12' y2='4'/><line x1='6' y1='20' x2='6' y2='14'/></svg>"); display:inline-block; margin-right:7px; vertical-align:-3px; }
  [data-baseweb="tab"]:nth-child(3)::before { content: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%236A6F78' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><path d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'/></svg>"); display:inline-block; margin-right:7px; vertical-align:-3px; }
  [data-baseweb="tab"]:nth-child(4)::before { content: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%236A6F78' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><circle cx='11' cy='11' r='8'/><line x1='21' y1='21' x2='16.65' y2='16.65'/></svg>"); display:inline-block; margin-right:7px; vertical-align:-3px; }
  [data-baseweb="tab"]:nth-child(5)::before { content: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%236A6F78' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><circle cx='7' cy='17' r='2'/><circle cx='17' cy='7' r='2'/><circle cx='17' cy='17' r='2'/><line x1='8.7' y1='15.7' x2='15.3' y2='8.3'/><line x1='15' y1='17' x2='9' y2='17'/></svg>"); display:inline-block; margin-right:7px; vertical-align:-3px; }
  [aria-selected="true"][data-baseweb="tab"]:nth-child(1)::before { content: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%232E6FA8' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='10'/><line x1='2' y1='12' x2='22' y2='12'/><path d='M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z'/></svg>"); }
  [aria-selected="true"][data-baseweb="tab"]:nth-child(2)::before { content: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%232E6FA8' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><line x1='18' y1='20' x2='18' y2='10'/><line x1='12' y1='20' x2='12' y2='4'/><line x1='6' y1='20' x2='6' y2='14'/></svg>"); }
  [aria-selected="true"][data-baseweb="tab"]:nth-child(3)::before { content: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%232E6FA8' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><path d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'/></svg>"); }
  [aria-selected="true"][data-baseweb="tab"]:nth-child(4)::before { content: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%232E6FA8' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><circle cx='11' cy='11' r='8'/><line x1='21' y1='21' x2='16.65' y2='16.65'/></svg>"); }
  [aria-selected="true"][data-baseweb="tab"]:nth-child(5)::before { content: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%232E6FA8' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><circle cx='7' cy='17' r='2'/><circle cx='17' cy='7' r='2'/><circle cx='17' cy='17' r='2'/><line x1='8.7' y1='15.7' x2='15.3' y2='8.3'/><line x1='15' y1='17' x2='9' y2='17'/></svg>"); }

  /* ── Theme toggle button: sun icon (light mode active) ── */
  [data-testid="stSidebar"] [data-testid="stButton"]:first-child button::before { content: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%230B0D10' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='5'/><line x1='12' y1='1' x2='12' y2='3'/><line x1='12' y1='21' x2='12' y2='23'/><line x1='4.22' y1='4.22' x2='5.64' y2='5.64'/><line x1='18.36' y1='18.36' x2='19.78' y2='19.78'/><line x1='1' y1='12' x2='3' y2='12'/><line x1='21' y1='12' x2='23' y2='12'/><line x1='4.22' y1='19.78' x2='5.64' y2='18.36'/><line x1='18.36' y1='5.64' x2='19.78' y2='4.22'/></svg>"); display:inline-block; margin-right:8px; vertical-align:-2px; }

  /* ══════════ Word-break normalization ══════════ */
  [data-testid="stMainBlockContainer"] *,
  [data-testid="stMetric"] *,
  .rink-card, .rink-card *, .player-card, .player-card *,
  .kings-card, .kings-card *, .rink-brand, .rink-brand * {
    word-break: normal !important;
    overflow-wrap: break-word !important;
    hyphens: manual !important;
  }
  .stat-value, .delta-pos, .delta-neg, .pct-pos, .pct-neg,
  [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
    white-space: nowrap !important;
    word-break: keep-all !important;
  }

  /* Safe-area padding for iOS */
  .stApp, [data-testid="stAppViewContainer"] {
    padding-bottom: env(safe-area-inset-bottom) !important;
    padding-left: env(safe-area-inset-left) !important;
    padding-right: env(safe-area-inset-right) !important;
  }

  /* ══════════ Responsive (port of RINK-1 styles.css @media rules) ══════════ */
  @media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 12px !important; row-gap: 12px !important; }
    [data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex: 1 1 220px !important; min-width: 0 !important; width: auto !important; }
    .rink-brand .season { margin-left: 0; flex-basis: 100%; font-size: 0.68rem; opacity: 0.8; }
    .rink-brand { flex-wrap: wrap; row-gap: 4px; }
    .rink-footer { flex-direction: column; gap: 6px; }
    [data-baseweb="tab-list"] { overflow-x: auto; overflow-y: hidden; flex-wrap: nowrap !important; -webkit-overflow-scrolling: touch; scrollbar-width: none; }
    [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
    [data-baseweb="tab"] { padding: 0 14px !important; font-size: 0.9rem !important; height: 44px !important; flex-shrink: 0 !important; }
    [data-baseweb="tab"]::before { display: none !important; margin-right: 0 !important; }
  }
  @media (max-width: 720px) {
    .block-container { padding-top: 1rem !important; padding-bottom: calc(84px + env(safe-area-inset-bottom)) !important; padding-left: 18px !important; padding-right: 18px !important; }
    .rink-brand { padding: 0 0 10px; margin-bottom: 18px; }

    /* Bottom tab nav (light) */
    [data-baseweb="tab-list"] {
      position: fixed !important;
      bottom: 0 !important; left: 0 !important; right: 0 !important;
      z-index: 100 !important;
      background: #F5F2EC !important;
      border-top: 1px solid #E4DFD5 !important;
      border-bottom: none !important;
      box-shadow: 0 -8px 24px rgba(0,0,0,0.08) !important;
      height: calc(62px + env(safe-area-inset-bottom)) !important;
      padding: 0 4px env(safe-area-inset-bottom) !important;
      margin: 0 !important;
      display: flex !important;
      flex-direction: row !important;
      flex-wrap: nowrap !important;
      justify-content: space-around !important;
      align-items: stretch !important;
      gap: 0 !important;
      overflow: visible !important;
    }
    [data-baseweb="tab"] {
      flex: 1 1 0 !important;
      min-width: 0 !important;
      height: 62px !important;
      padding: 6px 2px 4px !important;
      font-family: 'Inter', sans-serif !important;
      font-size: 0.62rem !important;
      font-weight: 500 !important;
      letter-spacing: 0.04em !important;
      text-transform: uppercase !important;
      display: flex !important;
      flex-direction: column !important;
      align-items: center !important;
      justify-content: center !important;
      gap: 5px !important;
      color: #6A6F78 !important;
      border-bottom: none !important;
      border-top: 2px solid transparent !important;
      border-radius: 0 !important;
      background: transparent !important;
      text-overflow: ellipsis !important;
      overflow: hidden !important;
      white-space: nowrap !important;
    }
    [data-baseweb="tab"]::before {
      display: block !important;
      margin: 0 !important;
      transform: scale(1.3) !important;
      vertical-align: baseline !important;
      line-height: 0 !important;
    }
    [aria-selected="true"][data-baseweb="tab"] {
      color: #2E6FA8 !important;
      border-top: 2px solid #2E6FA8 !important;
      border-bottom: none !important;
      background: rgba(46,111,168,0.07) !important;
      font-family: 'Inter', sans-serif !important;
      font-style: normal !important;
    }

    [data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex: 1 1 160px !important; }
    [data-testid="stMetric"] { padding: 14px 16px !important; }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.66rem !important; letter-spacing: 0.1em !important; }
    [data-testid="stMetricDelta"] { font-size: 0.72rem !important; }
    .player-hero { padding: 22px 20px; border-radius: 12px; }
    .player-hero .name { font-size: 1.9rem; line-height: 1.1; }
    .player-hero .meta { font-size: 0.74rem; }
    .rink-card { padding: 14px 16px; }
    .rink-card .value { font-size: 1.25rem; }
    .player-card, .kings-card { padding: 14px 16px; border-radius: 10px; }
    .lb-bar-wrap, .lb-pct { display: none !important; }
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.3rem !important; }
    h3 { font-size: 1.05rem !important; }
    .js-plotly-plot, .plot-container { width: 100% !important; }
    section[data-testid="stSidebar"] { z-index: 99 !important; }
  }
  @media (max-width: 480px) {
    .block-container { padding-left: 14px !important; padding-right: 14px !important; }
    .rink-brand .logo { font-size: 1.15rem; }
    .rink-brand .tag { font-size: 0.64rem; }
    [data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex: 1 1 100% !important; }
    [data-testid="stMetric"] { padding: 12px 14px !important; }
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.62rem !important; }
    .player-hero { padding: 18px 18px; }
    .player-hero .name { font-size: 1.55rem; }
    .signal-badge, .verdict-pill { font-size: 0.62rem; padding: 2px 7px; }
  }

  /* Bottom nav: smooth scaling (light theme mirrors dark) */
  /* Bottom nav: smooth label scale so it never truncates.
     5 tabs × viewport/5 gives each ~vw/5 px of width. Longest label is
     'LEAGUE OVERVIEW' ≈ 9-10 chars at ~0.55em each uppercase, so
     font-size needs to be ≤ (vw/5 − padding) / 10 ≈ 1.2vw. */
  @media (max-width: 720px) {
    [data-baseweb="tab"] {
      font-size: clamp(0.38rem, 1.18vw, 0.72rem) !important;
      letter-spacing: 0 !important;
      padding: 0 2px !important;
      gap: 3px !important;
      text-transform: none !important;
    }
    [data-baseweb="tab"]::before { transform: scale(1.1) !important; margin: 0 !important; }
  }
  /* At true phone widths labels don't fit legibly — icon only. */
  @media (max-width: 480px) {
    [data-baseweb="tab"] { font-size: 0 !important; gap: 0 !important; padding: 0 !important; }
    [data-baseweb="tab"]::before { transform: scale(1.6) !important; }
  }
</style>"""



def _inject_css(dark: bool = True) -> None:
    """Inject theme CSS. Called at start of main() based on session_state."""
    st.markdown(_FONT_LINK, unsafe_allow_html=True)
    st.markdown(_DARK_CSS if dark else _LIGHT_CSS, unsafe_allow_html=True)


PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"


# ── Data loaders ───────────────────────────────────────────────────────────────
def _predictions_mtime() -> float:
    """Return mtime of predictions.csv — used as cache key so cache busts on file change."""
    p = PROCESSED_DIR / "predictions.csv"
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data
def _load_predictions_for_mtime(_mtime: float) -> pd.DataFrame:
    path = PROCESSED_DIR / "predictions.csv"
    if not path.exists():
        st.error("predictions.csv not found — run `py -3 pipeline.py` first.")
        st.stop()
    df = pd.read_csv(path)
    for col in ["cap_hit", "predicted_value", "value_delta"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "is_estimated" not in df.columns:
        df["is_estimated"] = False
    df["is_estimated"] = df["is_estimated"].fillna(False).astype(bool)
    return df


def load_predictions() -> pd.DataFrame:
    """Load predictions — cache busts automatically whenever the file is updated."""
    return _load_predictions_for_mtime(_predictions_mtime())


@st.cache_data
def load_shap_summary() -> pd.DataFrame:
    p = PROCESSED_DIR / "shap_summary.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_shap_values() -> pd.DataFrame:
    p = PROCESSED_DIR / "shap_values.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_season_context() -> dict:
    p = PROCESSED_DIR / "season_context.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data
def load_last_updated() -> dict:
    p = PROCESSED_DIR / "last_updated.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ── Comp pool helpers ──────────────────────────────────────────────────────────
def _build_comp_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Build the comp pool from loaded predictions — cached per session."""
    from src.models.comps import build_ufa_comp_pool
    return build_ufa_comp_pool(df)


def _get_player_comps(player: pd.Series, comp_pool: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Get the actual 5 comps for a player from the comps engine."""
    from src.models.comps import find_comps
    comps = find_comps(player, comp_pool, n=n)
    if comps.empty:
        return pd.DataFrame()
    keep = ["name", "team", "pos", "age", "cap_hit", "performance_score",
            "cluster_label", "cluster_id", "p60", "_dist", "_weight", "_same"]
    return comps[[c for c in keep if c in comps.columns]].copy()


# ── Helpers ────────────────────────────────────────────────────────────────────
def fmt_m(v) -> str:
    """Format as $X.XXM."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"${v/1_000_000:.2f}M"


def fmt_delta(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    s = "+" if v >= 0 else ""
    return f"{s}${v/1_000_000:.2f}M"


def fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    s = "+" if v >= 0 else ""
    return f"{s}{v:.1f}%"


def pct_rank(series, val) -> int:
    clean = series.dropna()
    if len(clean) == 0:
        return 50
    return round((clean < val).sum() / len(clean) * 100)


def _season_str(ctx: dict) -> str:
    """Convert season_context current_season_id to display string e.g. '2025-26'."""
    sid = str(ctx.get("current_season_id", 20252026))
    return f"{sid[:4]}-{sid[6:]}"


def headshot_url(player_id, team: str = "") -> str:
    ctx = load_season_context()
    season_id = ctx.get("current_season_id", 20252026)
    team = str(team).upper().strip() if team else ""
    if team:
        return f"https://assets.nhle.com/mugs/nhl/{season_id}/{team}/{int(player_id)}.png"
    return f"https://assets.nhle.com/mugs/nhl/{season_id}/{int(player_id)}.png"


def team_logo_url(team_abbrev: str) -> str:
    return f"https://assets.nhle.com/logos/nhl/svg/{team_abbrev}_light.svg"


def _mini_player_cards(players_df: pd.DataFrame, delta_col: str = "value_delta",
                       show_pred: bool = False) -> None:
    """Responsive grid of mini headshot cards — auto-wraps when container narrows.
    Uses CSS grid (minmax 140px) instead of st.columns() so the cards reflow based
    on available width regardless of sidebar state or viewport — prevents the
    one-character-per-line collapse when space is tight."""
    _card_bg  = _T["card_bg"]
    _card_bd  = _T["card_border"]
    _card_txt = _T["card_text"]
    _card_sub = _T["card_subtext"]

    cards_html = ""
    for _, row in players_df.iterrows():
        pid   = row.get("player_id")
        name  = row.get("name", "?")
        team  = row.get("team", "?")
        pos   = row.get("pos", "?")
        age   = row.get("age")
        delta = row.get(delta_col)
        pv    = row.get("predicted_value")

        clr     = _T.get("positive", "#2A7A4B") if (delta or 0) >= 0 else _T.get("negative", "#C0392B")
        val_str = fmt_delta(delta) if not show_pred else fmt_m(pv)

        hs_html = ""
        if pid and pd.notna(pid):
            hs_url  = headshot_url(pid, team)
            hs_html = (
                f"<img src='{hs_url}' width='56' height='56' "
                f"style='object-fit:cover;display:block;margin:0 auto 8px;border-radius:4px;' "
                f"onerror=\"this.style.display='none'\">"
            )

        cards_html += (
            f"<div style='background:{_card_bg};padding:14px 10px;"
            f"text-align:center;border:1px solid {_card_bd};border-top:3px solid {clr};"
            f"border-radius:8px;min-width:0;'>"
            f"  {hs_html}"
            f"  <div style='font-family:\"Space Grotesk\",sans-serif;font-weight:500;color:{_card_txt};font-size:.95rem;"
            f"    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{name}</div>"
            f"  <div style='color:{_card_sub};font-size:.7rem;margin:4px 0;"
            f"    font-family:\"Inter\",sans-serif;letter-spacing:.06em;text-transform:uppercase;"
            f"    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>"
            f"    {team} · {pos}</div>"
            f"  <div style='color:{clr};font-size:.85rem;font-weight:500;"
            f"    margin-top:6px;font-family:\"JetBrains Mono\",monospace;"
            f"    white-space:nowrap;'>{val_str}</div>"
            f"</div>"
        )

    st.markdown(
        f"<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));"
        f"gap:10px;margin-bottom:12px;'>{cards_html}</div>",
        unsafe_allow_html=True,
    )


def signal_badge(signal: str, palette: dict) -> str:
    color = palette.get(signal, "#2C3A40")
    return (f"<span class='signal-badge' "
            f"style='background:{color};color:#fff;'>{signal}</span>")


def delta_color_html(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "<span style='color:#888'>N/A</span>"
    cls = "delta-pos" if v >= 0 else "delta-neg"
    return f"<span class='{cls}'>{fmt_delta(v)}</span>"


def add_delta_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Add value_delta_pct column = delta / cap_hit * 100."""
    df = df.copy()
    df["value_delta_pct"] = np.where(
        df["cap_hit"].notna() & (df["cap_hit"] > 0),
        df["value_delta"] / df["cap_hit"] * 100,
        np.nan,
    )
    return df


# ── Kings-specific resign signal ───────────────────────────────────────────────
def kings_resign_signal(row) -> str:
    age      = row.get("age") if pd.notna(row.get("age")) else 30
    delta    = row.get("value_delta") if pd.notna(row.get("value_delta")) else 0
    _yl      = row.get("years_left")
    yrs_left = int(_yl) if pd.notna(_yl) else 0
    _ch      = row.get("cap_hit")
    cap_hit  = float(_ch) if pd.notna(_ch) else 0.0
    has_data = bool(row.get("has_contract_data"))

    # Extension already signed — highest priority override
    if row.get("has_extension"):
        return "Extension Signed"

    if not has_data:
        return "UFA"

    expiring      = yrs_left <= 1
    is_elc        = cap_hit < 1_000_000 and age < 25
    severely_over = delta < -1_500_000
    overpaid      = delta < -500_000
    underpaid     = delta > 200_000
    young         = age < 26
    veteran       = age >= 33
    many_yrs_left = yrs_left >= 3

    if is_elc and underpaid:
        return "Extension Now"
    if expiring and underpaid and young:
        return "Lock Up"
    if expiring and underpaid and not veteran:
        return "Priority Re-sign"
    if expiring and overpaid and veteran:
        return "Let Walk"
    if not expiring and severely_over and many_yrs_left:
        return "Buyout Candidate"
    if overpaid:
        return "Monitor"
    return "Fair Deal"


# ── Sidebar ────────────────────────────────────────────────────────────────────
def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        # ── Dark/Light mode toggle ──────────────────────────────────────
        _dark = st.session_state.get("dark_mode", True)
        _label = "Dark Mode" if _dark else "Light Mode"
        if st.button(_label, key="_theme_btn", use_container_width=True):
            st.session_state["dark_mode"] = not _dark
            st.rerun()
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        _sb_sub = _T["card_subtext"]; _sb_txt = _T["page_text"]

        def _sb_section(label):
            _cb = _T["card_border"]
            st.markdown(
                f"<div style='font-family:\"Inter\",sans-serif;font-size:.65rem;"
                f"font-weight:500;letter-spacing:.15em;text-transform:uppercase;"
                f"color:{_sb_sub};border-top:1px solid {_cb};"
                f"padding-top:10px;margin-top:4px;margin-bottom:6px;'>{label}</div>",
                unsafe_allow_html=True,
            )

        def _sb_mono(text):
            st.markdown(
                f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:.78rem;"
                f"color:{_sb_txt};line-height:1.7;'>{text}</div>",
                unsafe_allow_html=True,
            )

        _sb_section("NHL Value Model")

        _sb_section("Filters")
        positions = ["All", "C", "L", "R", "D", "F (all)"]
        pos_sel  = st.selectbox("Position", positions, key="sb_pos")
        teams_list = ["All"] + sorted(df["team"].dropna().unique().tolist())
        team_sel = st.selectbox("Team", teams_list, key="sb_team")
        # Cluster filter — checkbox list instead of cramped multiselect chips.
        cluster_opts = [c for c in CLUSTER_ORDER if c in df["cluster_label"].unique()]
        # Clear stale session-state keys if cluster labels changed
        for _cl in list(st.session_state.keys()):
            if _cl.startswith("sb_cl_") and _cl[6:] not in cluster_opts:
                del st.session_state[_cl]

        st.markdown(
            f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:.72rem;"
            f"color:{_sb_sub};letter-spacing:.08em;text-transform:uppercase;"
            f"margin:8px 0 6px;'>Role Cluster</div>",
            unsafe_allow_html=True,
        )

        # Quick toggles — use on_click callbacks so state mutations happen
        # BEFORE the checkboxes re-render on the same run.
        def _bulk_set(val: bool):
            for _cl in cluster_opts:
                st.session_state[f"sb_cl_{_cl}"] = val

        _c1, _c2 = st.columns(2)
        _c1.button("All",  key="sb_cl_all_btn",  use_container_width=True, on_click=_bulk_set, args=(True,))
        _c2.button("None", key="sb_cl_none_btn", use_container_width=True, on_click=_bulk_set, args=(False,))

        _cl_cols = st.columns(2)
        cluster_sel = []
        for _i, _cl in enumerate(cluster_opts):
            _key = f"sb_cl_{_cl}"
            _default = st.session_state.get(_key, True)
            if _cl_cols[_i % 2].checkbox(_cl, value=_default, key=_key):
                cluster_sel.append(_cl)

        import math
        age_min = math.floor(df["age"].dropna().min())
        age_max = math.ceil(df["age"].dropna().max())
        st.markdown(
            f"<div style='font-size:.65rem;color:{_sb_sub};margin-bottom:4px;"
            f"font-family:\"Inter\",sans-serif;letter-spacing:.12em;text-transform:uppercase;'>"
            f"Age range</div>",
            unsafe_allow_html=True,
        )
        _ac1, _ac2 = st.columns(2)
        age_lo = _ac1.number_input("Min age", min_value=age_min, max_value=age_max, value=age_min, step=1, label_visibility="collapsed", key="age_lo")
        age_hi = _ac2.number_input("Max age", min_value=age_min, max_value=age_max, value=age_max, step=1, label_visibility="collapsed", key="age_hi")
        age_r = (int(age_lo), int(age_hi))

        filt = df.copy()
        if pos_sel == "F (all)":
            filt = filt[filt["pos"].isin(["C", "L", "R"])]
        elif pos_sel != "All":
            filt = filt[filt["pos"] == pos_sel]
        if team_sel != "All":
            filt = filt[filt["team"] == team_sel]
        # NaN-age players always pass through — don't drop them
        age_mask = filt["age"].isna() | ((filt["age"] >= age_r[0]) & (filt["age"] <= age_r[1]))
        filt = filt[age_mask]
        # Cluster filter
        if cluster_sel:
            filt = filt[filt["cluster_label"].isin(cluster_sel)]

        # Season context
        ctx = load_season_context()
        if ctx:
            _sb_section("Season")
            _sb_mono(ctx.get("description", ""))

        # Model stats
        n_players = df["has_contract_data"].fillna(False).sum()
        _sb_section("Model")
        _sb_mono(
            f"Comps Model &nbsp;·&nbsp; k=7 Clusters<br>"
            f"XGB Benchmark R² 0.83 &nbsp;·&nbsp; {n_players} players"
        )

        # Data freshness
        if _refresh_status["error"]:
            _sb_section("Data")
            _sb_mono(f"Error: {_refresh_status['error'][:50]}")
        else:
            pred_path = PROCESSED_DIR / "predictions.csv"
            if pred_path.exists():
                from datetime import datetime
                mtime = datetime.fromtimestamp(pred_path.stat().st_mtime)
                label = "Updating..." if _refresh_status["running"] else mtime.strftime('%b %d %Y, %I:%M %p')
                _sb_section("Data")
                _sb_mono(label)

    return filt


# ── Tab 1: League Overview ─────────────────────────────────────────────────────
def tab_overview(df: pd.DataFrame, full_df: pd.DataFrame):
    df_all = df[df["predicted_value"].notna()].copy()
    df_all = add_delta_pct(df_all)
    df_c   = df_all[df_all["cap_hit"].notna()].copy()
    df_ufa = df_all[df_all["cap_hit"].isna()].copy()

    _txt = _T["page_text"]; _sub = _T["card_subtext"]; _cbg = _T["card_bg"]; _cbd = _T["card_border"]
    _pos = _T["positive"]; _neg = _T["negative"]
    _accent = _T.get("accent", "#F5F5F2")

    # ── A. Hero Section ──────────────────────────────────────────────────────
    st.markdown(
        f"<div style='padding:0 0 24px;'>"
        f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:3rem;"
        f"font-weight:400;color:{_txt};line-height:1.05;margin-bottom:10px;'>"
        f"Predicting NHL Player<br>Market Value</div>"
        f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:.7rem;"
        f"letter-spacing:.12em;text-transform:uppercase;color:{_sub};margin-bottom:16px;'>"
        f"{_season_str(load_season_context())} &nbsp;·&nbsp; "
        f"Comps-Based Valuation &nbsp;·&nbsp; K-Means Clustering &nbsp;·&nbsp; Live Data</div>"
        f"<div style='font-family:\"Inter\",sans-serif;font-size:.82rem;"
        f"color:{_sub};line-height:1.7;max-width:680px;'>"
        f"This model estimates each NHL skater's fair market value by clustering players into "
        f"positional roles, scoring their performance within those roles, and finding the five "
        f"most comparable contracts. The result: a transparent, comps-based valuation for every "
        f"player in the league.</div>"
        f"<div style='height:1px;background:{_cbd};margin-top:24px;'></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── B. Methodology (collapsed by default) ───────────────────────────────
    _abg = _cbg; _abd = _cbd; _atxt = _T["card_text"]; _asub = _sub
    _pos_clr = _pos
    with st.expander("How the model works", expanded=False):
        _steps = [
            ("Data Ingestion", "NHL API + PuckPedia — roster stats, contracts, prior-season stats; all normalized to 82-game pace."),
            ("K-Means Clustering", "k=7 unsupervised groups players by deployment (TOI, PP pts, ±, faceoff%, position). Auto-labels roles."),
            ("Performance Scoring", "Z-scored within each cluster. FWD: G/60 · P/60 · PPP · shots · sh% · ±. DEF: TOI · PPP · ± · sh%."),
            ("Comps Engine", "5 nearest neighbors. Weighted distance: P/60 (45%) + Score (30%) + Age (25%). UFAs weighted 1.5×; same-cluster priority."),
            ("Predicted Value", "Weighted average of 5 comps' AAVs. Delta = predicted − actual cap hit."),
        ]
        _rows = ""
        for _i, (_t, _d) in enumerate(_steps, start=1):
            _rows += (
                f"<div style='display:grid;grid-template-columns:32px 1fr;gap:14px;"
                f"padding:10px 0;border-bottom:1px solid {_abd};'>"
                f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:.7rem;"
                f"color:{_asub};letter-spacing:.08em;padding-top:2px;'>0{_i}</div>"
                f"<div>"
                f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:.95rem;"
                f"font-weight:600;color:{_atxt};margin-bottom:2px;'>{_t}</div>"
                f"<div style='font-family:\"Inter\",sans-serif;font-size:.8rem;"
                f"color:{_asub};line-height:1.55;'>{_d}</div>"
                f"</div></div>"
            )
        st.markdown(f"<div>{_rows}</div>", unsafe_allow_html=True)

    # Validation benchmark bar
    st.markdown(
        f"<div style='background:{_abg};border:1px solid {_abd};border-left:3px solid {_asub};"
        f"padding:10px 16px;margin:8px 0 28px;display:flex;align-items:center;gap:12px;'>"
        f"<span style='font-size:.68rem;color:{_asub};font-family:\"Inter\",sans-serif;"
        f"letter-spacing:.1em;text-transform:uppercase;'>Validation Benchmark</span>"
        f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:.8rem;color:{_atxt};'>"
        f"XGBoost + SHAP &nbsp;·&nbsp; 5-Fold CV R² ≈ 0.83 &nbsp;·&nbsp; RMSE ≈ $1.2M"
        f"</span></div>",
        unsafe_allow_html=True,
    )

    # ── C. Key Findings ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Skaters", f"{len(df_all):,}",
              delta=f"{len(df_ufa)} UFA/unsigned", delta_color="off")
    avg_cap = df_c["cap_hit"].mean() if len(df_c) else 0
    c2.metric("Avg Cap Hit (contracted)", fmt_m(avg_cap))
    if len(df_c):
        best  = df_c.nlargest(1,  "value_delta").iloc[0]
        worst = df_c.nsmallest(1, "value_delta").iloc[0]
        c3.metric("Most Underpaid", best["name"],  delta=fmt_delta(best["value_delta"]))
        c4.metric("Most Overpaid",  worst["name"], delta=fmt_delta(worst["value_delta"]),
                  delta_color="inverse")

    # ── D. Value Scatter ─────────────────────────────────────────────────────
    st.markdown(
        f"<div style='height:1px;background:{_cbd};margin:20px 0 16px;'></div>",
        unsafe_allow_html=True,
    )
    show_ufa = st.session_state.get("overview_show_ufa", False)

    fig = go.Figure()

    # Contracted players colored by delta on RdYlGn scale
    if not df_c.empty:
        fig.add_trace(go.Scatter(
            x=df_c["predicted_value"], y=df_c["cap_hit"],
            mode="markers", name="Contracted",
            marker=dict(
                size=7, opacity=0.82,
                color=df_c["value_delta"], colorscale="RdYlGn", cmid=0,
                line=dict(width=0.4, color="#000"),
                colorbar=dict(title="Delta ($)", tickformat="$,.0f",
                              len=0.65, thickness=12),
            ),
            customdata=df_c[["name", "team", "pos", "cap_hit",
                              "predicted_value", "value_delta", "value_delta_pct"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]} · %{customdata[2]}<br>"
                "Cap Hit: $%{customdata[3]:,.0f}<br>"
                "Predicted: $%{customdata[4]:,.0f}<br>"
                "Delta: $%{customdata[5]:,.0f} (%{customdata[6]:.1f}%)<br>"
                "<extra></extra>"
            ),
        ))

    if show_ufa and not df_ufa.empty:
        fig.add_trace(go.Scatter(
            x=df_ufa["predicted_value"], y=[0] * len(df_ufa),
            mode="markers", name="UFA / Unsigned",
            marker=dict(symbol="diamond", size=11, opacity=0.85,
                        color=_accent, line=dict(width=0)),
            customdata=df_ufa[["name", "team", "pos", "predicted_value"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b> — UFA / Unsigned<br>"
                "%{customdata[1]} · %{customdata[2]}<br>"
                "Predicted Market Value: $%{customdata[3]:,.0f}"
                "<extra></extra>"
            ),
        ))

    if not df_c.empty:
        lo = min(df_c["predicted_value"].min(), df_c["cap_hit"].min()) * 0.95
        hi = max(df_c["predicted_value"].max(), df_c["cap_hit"].max()) * 1.02
        _fvline_color = "#4FD1C5" if st.session_state.get("dark_mode", True) else "#0B0D10"
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                      line=dict(dash="dot", color=_fvline_color, width=2))
        fig.add_annotation(x=hi * 0.72, y=hi * 0.78, text="Fair value",
                           showarrow=False, font=dict(color=_fvline_color, size=12,
                                                      family="'Inter', sans-serif"))

    fig.update_layout(
        paper_bgcolor=_T["plot_paper"], plot_bgcolor=_T["plot_bg"],
        font=dict(family="'Inter', sans-serif", color=_T["plot_font"]),
        xaxis=dict(tickformat="$,.0f", title="Predicted Market Value",
                   gridcolor=_T["grid"], zeroline=False),
        yaxis=dict(tickformat="$,.0f", title="Actual Cap Hit (0 = UFA/Unsigned)",
                   gridcolor=_T["grid"]),
        showlegend=False,
        height=560,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    event = st.plotly_chart(fig, use_container_width=True,
                            on_select="rerun", key="overview_scatter")

    _ufa_label = "Hide UFA / Unsigned" if show_ufa else "Show UFA / Unsigned"
    if st.button(_ufa_label, key="overview_ufa_toggle"):
        st.session_state["overview_show_ufa"] = not show_ufa
        st.rerun()
    st.markdown(
        f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:0.72rem;color:{_sub};line-height:2;'>"
        f"<span style='display:inline-block;width:8px;height:8px;background:{_pos};vertical-align:middle;margin-right:5px;'></span>Below dashed line = underpaid"
        f"&nbsp;&nbsp;·&nbsp;&nbsp;"
        f"<span style='display:inline-block;width:8px;height:8px;background:{_neg};vertical-align:middle;margin-right:5px;'></span>Above = overpaid"
        f"&nbsp;&nbsp;·&nbsp;&nbsp;"
        f"Click any marker for details.</div>",
        unsafe_allow_html=True,
    )

    if event and event.get("selection") and event["selection"].get("points"):
        pts = event["selection"]["points"]
        if pts:
            cd = pts[0].get("customdata", [])
            clicked_name = cd[0] if cd else ""
            match = full_df[full_df["name"] == clicked_name]
            if not match.empty:
                p = match.iloc[0]
                st.markdown(f"**Selected: {clicked_name}**")
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                mc1.metric("Team / Pos", f"{p.get('team','?')} · {p.get('pos','?')}")
                mc2.metric("Cap Hit",         fmt_m(p.get("cap_hit")))
                mc3.metric("Predicted Value", fmt_m(p.get("predicted_value")))
                dv = p.get("value_delta")
                mc4.metric("Delta", fmt_delta(dv),
                           delta_color="normal" if (dv or 0) >= 0 else "inverse")
                _p_cl = p.get("cluster_label", "—")
                _p_ps = p.get("performance_score")
                _ps_s = f"{_p_ps:+.1f}" if pd.notna(_p_ps) else "—"
                mc5.metric("Role / Score", f"{_p_cl} · {_ps_s}")

    # ── E. Cluster Economics + Top 5 (stacked) ──────────────────────────────
    if not df_c.empty and "cluster_label" in df_c.columns:
        st.markdown(f"<div style='height:1px;background:{_cbd};margin:24px 0 20px;'></div>",
                    unsafe_allow_html=True)

        # --- Cluster Economics (full width) ---
        st.markdown(
            f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:1.5rem;"
            f"color:{_txt};margin-bottom:14px;font-weight:500;'>Cluster Economics</div>",
            unsafe_allow_html=True,
        )
        _cl_dot = _T.get("accent_ice", _T.get("card_subtext", "#8A8F99"))

        # Column widths: cluster name flexes; numeric cells fixed. Min-width
        # of the inner table = 520px — below that, outer wrapper scrolls.
        _grid_cols = "minmax(130px,1.6fr) 44px 54px 72px 72px 82px 60px"
        _hdr_cell = (f"font-family:\"JetBrains Mono\",monospace;font-size:.66rem;"
                     f"letter-spacing:.1em;text-transform:uppercase;color:{_txt};"
                     f"font-weight:500;padding:10px 6px;text-align:right;white-space:nowrap;")

        _rows = (
            f"<div style='background:{_cbg};border:1px solid {_cbd};border-radius:8px;"
            f"overflow-x:auto;overflow-y:hidden;'>"
            f"<div style='min-width:520px;'>"
            f"<div style='display:grid;grid-template-columns:{_grid_cols};"
            f"gap:0;border-bottom:2px solid {_cbd};background:{_T['card_header']};'>"
            f"<div style='{_hdr_cell}text-align:left;'>Cluster</div>"
            f"<div style='{_hdr_cell}'>N</div>"
            f"<div style='{_hdr_cell}'>Age</div>"
            f"<div style='{_hdr_cell}'>Cap</div>"
            f"<div style='{_hdr_cell}'>Pred</div>"
            f"<div style='{_hdr_cell}'>Δ</div>"
            f"<div style='{_hdr_cell}'>Score</div>"
            f"</div>"
        )

        _row_cell_val = (f"font-family:\"JetBrains Mono\",monospace;font-size:.78rem;"
                         f"color:{_txt};padding:11px 6px;text-align:right;"
                         f"white-space:nowrap;display:flex;align-items:center;justify-content:flex-end;")
        _row_cell_name = (f"font-family:\"Inter\",sans-serif;font-size:.8rem;font-weight:500;"
                          f"color:{_txt};padding:11px 10px;text-align:left;"
                          f"white-space:nowrap;display:flex;align-items:center;gap:8px;"
                          f"overflow:hidden;text-overflow:ellipsis;")

        for cl in CLUSTER_ORDER:
            cl_sub = df_c[df_c["cluster_label"] == cl]
            if cl_sub.empty:
                continue
            avg_d = cl_sub["value_delta"].mean()
            _d_clr = _pos if avg_d >= 0 else _neg
            avg_ps = cl_sub["performance_score"].dropna().mean()
            _row_bord = f"border-bottom:1px solid {_cbd};"
            _rows += (
                f"<div style='display:grid;grid-template-columns:{_grid_cols};"
                f"gap:0;{_row_bord}'>"
                f"<div style='{_row_cell_name}'>"
                f"<span style='display:inline-block;width:8px;height:8px;background:{_cl_dot};"
                f"border-radius:2px;flex-shrink:0;'></span>"
                f"<span style='overflow:hidden;text-overflow:ellipsis;'>{cl}</span>"
                f"</div>"
                f"<div style='{_row_cell_val}'>{len(cl_sub)}</div>"
                f"<div style='{_row_cell_val}'>{cl_sub['age'].mean():.1f}</div>"
                f"<div style='{_row_cell_val}'>{fmt_m(cl_sub['cap_hit'].mean())}</div>"
                f"<div style='{_row_cell_val}'>{fmt_m(cl_sub['predicted_value'].dropna().mean())}</div>"
                f"<div style='{_row_cell_val}color:{_d_clr};font-weight:600;'>{fmt_delta(avg_d)}</div>"
                f"<div style='{_row_cell_val}'>{avg_ps:+.1f}</div>"
                f"</div>"
            )
        _rows += "</div></div>"

        st.markdown(_rows, unsafe_allow_html=True)

        # --- Top 5 Underpaid / Overpaid BELOW, side-by-side (wraps on narrow) ---
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        _top_left, _top_right = st.columns(2)
        with _top_left:
            st.markdown(
                f"<div style='font-family:\"Inter\",sans-serif;font-size:.65rem;font-weight:500;"
                f"letter-spacing:.12em;text-transform:uppercase;color:{_pos};margin-bottom:10px;"
                f"border-left:2px solid {_pos};padding-left:8px;'>Top 5 Underpaid</div>",
                unsafe_allow_html=True,
            )
            _mini_player_cards(df_c.nlargest(5, "value_delta"))
        with _top_right:
            st.markdown(
                f"<div style='font-family:\"Inter\",sans-serif;font-size:.65rem;font-weight:500;"
                f"letter-spacing:.12em;text-transform:uppercase;color:{_neg};margin-bottom:10px;"
                f"border-left:2px solid {_neg};padding-left:8px;'>Top 5 Overpaid</div>",
                unsafe_allow_html=True,
            )
            _mini_player_cards(df_c.nsmallest(5, "value_delta"))


# ── Tab 2: Leaderboards ────────────────────────────────────────────────────────
def tab_leaderboards(df: pd.DataFrame):
    st.markdown(f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:2.2rem;color:{_T['page_text']};letter-spacing:0;margin:0 0 20px;font-weight:400;'>Value Leaderboards</div>", unsafe_allow_html=True)

    df_c   = df[df["cap_hit"].notna() & df["value_delta"].notna()].copy()
    df_c   = add_delta_pct(df_c)
    df_ufa = df[df["cap_hit"].isna() & df["predicted_value"].notna()].copy()

    lc1, lc2, lc3 = st.columns([1, 1, 2])
    n = int(lc1.number_input("Show top N", min_value=5, max_value=30, value=15, step=1, key="lb_n"))
    sort_by = lc2.radio("Sort by", ["% Delta", "$ Delta"], horizontal=True, key="lb_sort")
    pos_opts = ["All", "C", "L", "R", "D", "F (all)"]
    pos_f = lc3.radio("Position", pos_opts, horizontal=True, key="lb_pos")

    # Cluster filter
    _lb_cl_opts = ["All Clusters"] + [c for c in CLUSTER_ORDER if c in df_c["cluster_label"].unique()]
    _lb_cl_sel = st.selectbox("Filter by Role Cluster", _lb_cl_opts, key="lb_cluster")

    if pos_f == "F (all)":
        df_c = df_c[df_c["pos"].isin(["C", "L", "R"])]
    elif pos_f != "All":
        df_c = df_c[df_c["pos"] == pos_f]

    if _lb_cl_sel != "All Clusters":
        df_c = df_c[df_c["cluster_label"] == _lb_cl_sel]

    sort_col = "value_delta_pct" if sort_by == "% Delta" else "value_delta"

    col1, col2 = st.columns(2)

    def bar_chart(data, title, scale, ascending):
        x_col  = sort_col
        x_fmt  = ":.1f" if sort_by == "% Delta" else ":$,.0f"
        x_lbl  = "% Delta" if sort_by == "% Delta" else "$ Delta"
        fig = px.bar(
            data, x=x_col, y="name", orientation="h",
            color=x_col, color_continuous_scale=scale,
            hover_data={
                "team": True, "pos": True, "age": ":.0f",
                "cap_hit": ":$,.0f", "predicted_value": ":$,.0f",
                "value_delta": ":$,.0f", "value_delta_pct": ":.1f",
            },
            labels={x_col: x_lbl, "name": ""},
            height=max(360, n * 28),
        )
        fig.update_layout(
            paper_bgcolor=_T["plot_paper"], plot_bgcolor=_T["plot_bg"],
            font=dict(family="'Inter', sans-serif", color=_T["plot_font"]), showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
            xaxis=dict(
                tickformat=".1f%" if sort_by == "% Delta" else "$,.0f",
                title=x_lbl, gridcolor=_T["grid"],
            ),
            margin=dict(l=0, r=10, t=30, b=10),
            title=dict(text=title, font=dict(family="'Inter', sans-serif", size=14, color=_T["plot_font"])),
        )
        return fig

    with col1:
        gems = df_c.nlargest(n, sort_col)
        st.plotly_chart(bar_chart(gems, "Hidden Gems — Most Underpaid", "Greens", False),
                        use_container_width=True)

    with col2:
        over = df_c.nsmallest(n, sort_col)
        st.plotly_chart(bar_chart(over, "Most Overpaid", "Reds_r", True),
                        use_container_width=True)

    st.markdown("---")

    # Headshot row for top 5 gems and top 5 overpaid
    hs1, hs2 = st.columns(2)
    with hs1:
        st.markdown(
            f"<div style='font-family:\"Inter\",sans-serif;font-size:.7rem;font-weight:500;"
            f"letter-spacing:.12em;text-transform:uppercase;color:{_T['positive']};margin-bottom:10px;"
            f"border-left:2px solid {_T['positive']};padding-left:8px;'>Top 5 Underpaid</div>",
            unsafe_allow_html=True,
        )
        _mini_player_cards(df_c.nlargest(5, sort_col))
    with hs2:
        st.markdown(
            f"<div style='font-family:\"Inter\",sans-serif;font-size:.7rem;font-weight:500;"
            f"letter-spacing:.12em;text-transform:uppercase;color:{_T['negative']};margin-bottom:10px;"
            f"border-left:2px solid {_T['negative']};padding-left:8px;'>Top 5 Overpaid</div>",
            unsafe_allow_html=True,
        )
        _mini_player_cards(df_c.nsmallest(5, sort_col))

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    _TABLE_CSS = (
        f"width:100%;border-collapse:collapse;font-family:\"Inter\",sans-serif;"
        f"font-size:.8rem;background:{_T['card_bg']};"
    )
    _TH_CSS = (
        f"padding:8px 12px;text-align:left;color:{_T['card_text']};font-weight:500;"
        f"font-family:\"Inter\",sans-serif;font-size:.7rem;letter-spacing:.12em;"
        f"text-transform:uppercase;border-bottom:2px solid {_T['card_border']};"
        f"background:{_T['card_header']};"
    )
    _TD_CSS = (
        f"padding:8px 12px;color:{_T['card_text']};border-bottom:1px solid {_T['row_divider']};"
        f"font-family:\"Inter\",sans-serif;"
    )

    def _html_table(data, delta_col="value_delta"):
        cols_order = ["name", "team", "pos", "age", "cluster_label", "cap_hit",
                      "predicted_value", "value_delta", "value_delta_pct"]
        col_labels = {
            "name": "Player", "team": "Team", "pos": "Pos", "age": "Age",
            "cluster_label": "Role",
            "cap_hit": "Cap Hit", "predicted_value": "Pred. Value",
            "value_delta": "$ Delta", "value_delta_pct": "% Delta",
        }
        cols = [c for c in cols_order if c in data.columns]
        rows_html = ""
        for _, row in data.iterrows():
            delta_v = row.get(delta_col, 0) or 0
            _rbg = _T["card_bg"]
            _ctxt = _T["card_text"]
            row_bg  = f"background:{_rbg};"
            cells = ""
            for c in cols:
                v = row.get(c)
                if c == "age":
                    txt = f"{v:.0f}" if pd.notna(v) else "?"
                elif c == "cap_hit":
                    txt = fmt_m(v)
                elif c == "predicted_value":
                    txt = fmt_m(v)
                elif c == "value_delta":
                    color = _T.get("positive", "#2A7A4B") if (v or 0) >= 0 else _T.get("negative", "#C0392B")
                    txt = f"<span style='color:{color};font-weight:600;font-family:\"JetBrains Mono\",monospace;'>{fmt_delta(v)}</span>"
                elif c == "value_delta_pct":
                    color = _T.get("positive", "#2A7A4B") if (v or 0) >= 0 else _T.get("negative", "#C0392B")
                    txt = f"<span style='color:{color};font-family:\"JetBrains Mono\",monospace;'>{fmt_pct(v)}</span>"
                elif c == "name":
                    txt = f"<span style='color:{_ctxt};font-weight:500;font-family:\"Inter\",sans-serif;'>{v}</span>"
                elif c == "team":
                    _team_accent = _T.get("accent", "#1A1A2E")
                    txt = f"<span style='color:{_team_accent};font-family:\"JetBrains Mono\",monospace;'>{v}</span>"
                elif c == "cluster_label":
                    _cl_c = _T.get("card_text", "#F2EEE5")
                    txt = f"<span style='color:{_cl_c};font-family:\"Inter\",sans-serif;font-size:.72rem;font-weight:500;'>{v}</span>"
                else:
                    txt = str(v) if pd.notna(v) else "?"
                cells += f"<td style='{_TD_CSS}'>{txt}</td>"
            rows_html += f"<tr style='{row_bg}'>{cells}</tr>"
        header = "".join(f"<th style='{_TH_CSS}'>{col_labels.get(c, c)}</th>" for c in cols)
        _cbg = _T["card_bg"]; _cbd = _T["card_border"]
        return (
            f"<div style='background:{_cbg};border:1px solid {_cbd};border-radius:0;"
            f"overflow:hidden;margin-bottom:4px;'>"
            f"<table style='{_TABLE_CSS}'>"
            f"<thead><tr>{header}</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            f"</table></div>"
        )

    t1, t2 = st.columns(2)
    with t1:
        st.markdown(
            f"<div style='font-family:\"Inter\",sans-serif;font-size:.7rem;font-weight:500;"
            f"letter-spacing:.12em;text-transform:uppercase;color:{_T['positive']};margin-bottom:8px;"
            f"border-left:2px solid {_T['positive']};padding-left:8px;'>"
            "Hidden Gems</div>",
            unsafe_allow_html=True,
        )
        st.markdown(_html_table(df_c.nlargest(n, sort_col)), unsafe_allow_html=True)
    with t2:
        st.markdown(
            f"<div style='font-family:\"Inter\",sans-serif;font-size:.7rem;font-weight:500;"
            f"letter-spacing:.12em;text-transform:uppercase;color:{_T['negative']};margin-bottom:8px;"
            f"border-left:2px solid {_T['negative']};padding-left:8px;'>"
            "Overpaid</div>",
            unsafe_allow_html=True,
        )
        st.markdown(_html_table(df_c.nsmallest(n, sort_col)), unsafe_allow_html=True)

    # ── UFA / Unsigned players ────────────────────────────────────────────────
    if not df_ufa.empty:
        ufa_filtered = df_ufa
        if pos_f == "F (all)":
            ufa_filtered = df_ufa[df_ufa["pos"].isin(["C", "L", "R"])]
        elif pos_f != "All":
            ufa_filtered = df_ufa[df_ufa["pos"] == pos_f]

        if not ufa_filtered.empty:
            st.markdown("---")
            _ufa_txt = _T["page_text"]; _ufa_sub = _T["card_subtext"]
            st.markdown(
                f"<div style='margin-bottom:12px;'>"
                f"<span style='font-family:\"Space Grotesk\",sans-serif;font-size:1.5rem;"
                f"font-weight:400;color:{_ufa_txt};'>UFA / Unsigned Players</span>"
                f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:.72rem;"
                f"color:{_ufa_sub};margin-left:12px;letter-spacing:.06em;'>"
                f"No current contract — predicted value only</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            top_ufa = ufa_filtered.nlargest(min(n, len(ufa_filtered)), "predicted_value")
            ufa_disp = top_ufa[["name", "team", "pos", "age", "predicted_value"]].copy()

            def _ufa_html_table(data):
                cols = ["name", "team", "pos", "age", "predicted_value"]
                col_labels = {"name": "Player", "team": "Team", "pos": "Pos",
                              "age": "Age", "predicted_value": "Predicted Value"}
                rows_html = ""
                for _, row in data.iterrows():
                    cells = ""
                    for c in cols:
                        v = row.get(c)
                        if c == "age":
                            txt = f"{v:.0f}" if pd.notna(v) else "?"
                        elif c == "predicted_value":
                            _pv_txt = _T["card_text"]
                            txt = f"<span style='color:{_pv_txt};font-family:\"JetBrains Mono\",monospace;font-weight:500;'>{fmt_m(v)}</span>"
                        elif c == "name":
                            _ct = _T["card_text"]
                            txt = f"<span style='color:{_ct};font-weight:500;font-family:\"Inter\",sans-serif;'>{v}</span>"
                        elif c == "team":
                            _ta = _T.get("accent", "#1A1A2E")
                            txt = f"<span style='color:{_ta};font-family:\"JetBrains Mono\",monospace;'>{v}</span>"
                        else:
                            txt = str(v) if pd.notna(v) else "?"
                        cells += f"<td style='{_TD_CSS}'>{txt}</td>"
                    rows_html += f"<tr>{cells}</tr>"
                header = "".join(f"<th style='{_TH_CSS}'>{col_labels.get(c, c)}</th>" for c in cols)
                _cbg2 = _T["card_bg"]; _cbd2 = _T["card_border"]
                return (
                    f"<div style='background:{_cbg2};border:1px solid {_cbd2};border-radius:0;"
                    f"overflow:hidden;margin-bottom:4px;'>"
                    f"<table style='{_TABLE_CSS}'>"
                    f"<thead><tr>{header}</tr></thead>"
                    f"<tbody>{rows_html}</tbody>"
                    f"</table></div>"
                )

            st.markdown(_ufa_html_table(ufa_disp), unsafe_allow_html=True)
            st.caption(
                f"Showing {len(top_ufa)} of {len(ufa_filtered)} UFA/unsigned players — "
                "sorted by predicted market value. These players have no active cap charge."
            )

    # ── Value Leaders by Role ─────────────────────────────────────────────────
    # Use the original (pre-filter) contracted data for this section
    _vlr_df = df[df["cap_hit"].notna() & df["value_delta"].notna()].copy()
    if not _vlr_df.empty and "cluster_label" in _vlr_df.columns:
        st.markdown("---")
        st.markdown(
            f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:1.7rem;"
            f"color:{_T['page_text']};margin-bottom:14px;font-weight:400;'>"
            f"Value Leaders by Role</div>",
            unsafe_allow_html=True,
        )
        _vlr_present = [c for c in CLUSTER_ORDER if c in _vlr_df["cluster_label"].values]
        for cl_name in _vlr_present:
            cl_sub = _vlr_df[_vlr_df["cluster_label"] == cl_name]
            if len(cl_sub) < 2:
                continue
            cl_clr = _T.get("card_text", "#F2EEE5")
            best = cl_sub.nlargest(1, "value_delta").iloc[0]
            worst = cl_sub.nsmallest(1, "value_delta").iloc[0]
            _v1, _v2, _v3 = st.columns([1.5, 2, 2])
            _v1.markdown(
                f"<div style='padding:8px 0;'>"
                f"<span style='color:{cl_clr};font-family:\"Inter\",sans-serif;"
                f"font-size:.82rem;font-weight:600;'>{cl_name}</span>"
                f"<br><span style='color:{_T['card_subtext']};font-family:\"JetBrains Mono\",monospace;"
                f"font-size:.68rem;'>{len(cl_sub)} players</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            _pos_c = _T.get("positive", "#2A7A4B")
            _neg_c = _T.get("negative", "#C0392B")
            _v2.markdown(
                f"<div style='background:{_T['card_bg']};border:1px solid {_T['card_border']};"
                f"border-left:3px solid {_pos_c};padding:8px 12px;'>"
                f"<span style='color:{_T['card_text']};font-family:\"Inter\",sans-serif;"
                f"font-size:.82rem;'>{best['name']}</span>"
                f"<span style='color:{_pos_c};font-family:\"JetBrains Mono\",monospace;"
                f"font-size:.78rem;margin-left:8px;'>{fmt_delta(best['value_delta'])}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            _v3.markdown(
                f"<div style='background:{_T['card_bg']};border:1px solid {_T['card_border']};"
                f"border-left:3px solid {_neg_c};padding:8px 12px;'>"
                f"<span style='color:{_T['card_text']};font-family:\"Inter\",sans-serif;"
                f"font-size:.82rem;'>{worst['name']}</span>"
                f"<span style='color:{_neg_c};font-family:\"JetBrains Mono\",monospace;"
                f"font-size:.78rem;margin-left:8px;'>{fmt_delta(worst['value_delta'])}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ── Tab 3: Teams (generic, all 32 teams) ──────────────────────────────────────
def tab_team(df: pd.DataFrame, team_code: str):
    tc        = TEAM_COLORS.get(team_code, TEAM_COLORS["LAK"])
    T_PRIMARY = tc["primary"]
    T_SECOND  = tc["secondary"]
    team_name = TEAM_NAMES.get(team_code, team_code)
    team_logo = team_logo_url(team_code)

    _ktxt = _T["card_text"]; _ksub = _T["card_subtext"]
    st.markdown(
        f"<div style='padding:0 0 16px 0;'>"
        f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:3.5rem;"
        f"font-weight:400;color:{_ktxt};line-height:1;margin-bottom:8px;'>"
        f"{team_name}</div>"
        f"<div style='color:{_ksub};font-size:.72rem;"
        f"font-family:\"JetBrains Mono\",monospace;letter-spacing:.1em;text-transform:uppercase;'>"
        f"{_season_str(load_season_context())} &nbsp;·&nbsp; Roster Analysis &nbsp;·&nbsp; Comps Model</div>"
        f"<div style='height:1px;background:{_T['card_border']};margin-top:16px;'></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    team_all = df[df["team"] == team_code].copy()
    if team_all.empty:
        st.warning(f"No {team_name} players found in predictions. Run pipeline.py first.")
        return

    team_all = add_delta_pct(team_all)
    team_all["team_signal"] = team_all.apply(kings_resign_signal, axis=1)
    team = team_all[team_all["cap_hit"].notna()].copy()

    # ── Cap summary ──────────────────────────────────────────────────────────
    total_committed   = team["cap_hit"].sum()
    cap_space         = CAP_CEILING - total_committed
    next_yr_committed = team[team["years_left"].fillna(0) >= 1]["cap_hit"].sum()
    next_yr_space     = CAP_CEILING - next_yr_committed
    n_expiring        = int((team["years_left"].fillna(0) <= 1).sum())

    st.markdown(
        f"<div style='height:1px;background:{_T['card_border']};margin-bottom:20px;'></div>",
        unsafe_allow_html=True,
    )
    cap_cols = st.columns(5)
    cap_cols[0].metric("Roster Players",      len(team_all))
    cap_cols[1].metric("Cap Committed",        fmt_m(total_committed))
    cap_cols[2].metric("Cap Space",            fmt_m(cap_space),
                        delta="+space" if cap_space > 0 else "over cap",
                        delta_color="normal" if cap_space > 0 else "inverse")
    cap_cols[3].metric("Next Season Space",    fmt_m(next_yr_space))
    cap_cols[4].metric("Expiring Contracts",   f"{n_expiring} players")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Team Cluster Composition ─────────────────────────────────────────────
    if "cluster_label" in team.columns:
        st.markdown(
            f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:1.8rem;"
            f"font-weight:400;color:{_T['page_text']};margin:16px 0 12px;'>Role Composition</div>"
            f"<div style='color:{_T['card_subtext']};font-size:.72rem;"
            f"font-family:\"JetBrains Mono\",monospace;letter-spacing:.06em;margin-bottom:12px;'>"
            f"How this team's roster is distributed across role clusters</div>",
            unsafe_allow_html=True,
        )
        cluster_counts = team.groupby("cluster_label").size().reindex(
            [c for c in CLUSTER_ORDER if c in team["cluster_label"].values], fill_value=0
        )
        cluster_counts = cluster_counts[cluster_counts > 0]

        fig_comp = go.Figure()
        for clabel in cluster_counts.index:
            cnt = cluster_counts[clabel]
            fig_comp.add_bar(
                name=clabel, y=["Roster"], x=[cnt], orientation="h",
                marker_color="#888",
                text=[f"{clabel} ({cnt})"], textposition="inside",
                textfont=dict(size=11, color="#fff", family="'Inter', sans-serif"),
                hovertemplate=f"{clabel}: {cnt} players<extra></extra>",
            )
        fig_comp.update_layout(
            barmode="stack",
            paper_bgcolor=_T["plot_paper"], plot_bgcolor=_T["plot_bg"],
            font=dict(family="'Inter', sans-serif", color=_T["plot_font"]),
            xaxis=dict(title="Players", gridcolor=_T["grid_alt"]),
            yaxis=dict(showticklabels=False),
            showlegend=False, height=120,
            margin=dict(l=10, r=10, t=10, b=30),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # ── Cluster Spending Breakdown ───────────────────────────────────────────
    if "cluster_label" in team.columns:
        cluster_spend = team.groupby("cluster_label").agg(
            cap_total=("cap_hit", "sum"),
            pred_total=("predicted_value", "sum"),
            n_players=("name", "count"),
        ).reindex([c for c in CLUSTER_ORDER if c in team["cluster_label"].values])
        cluster_spend = cluster_spend[cluster_spend["n_players"] > 0].copy()

        if not cluster_spend.empty:
            st.markdown(
                f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:1.8rem;"
                f"font-weight:400;color:{_T['page_text']};margin:20px 0 12px;'>Cluster Spending</div>"
                f"<div style='color:{_T['card_subtext']};font-size:.72rem;"
                f"font-family:\"JetBrains Mono\",monospace;letter-spacing:.06em;margin-bottom:12px;'>"
                f"Total cap committed vs. predicted market value by role</div>",
                unsafe_allow_html=True,
            )
            fig_spend = go.Figure()
            fig_spend.add_bar(
                name="Cap Committed",
                x=cluster_spend.index, y=cluster_spend["cap_total"],
                marker_color=T_SECOND, opacity=0.85,
                text=cluster_spend["cap_total"].apply(fmt_m),
                textposition="outside", textfont=dict(size=10, color=T_SECOND),
            )
            fig_spend.add_bar(
                name="Predicted Value",
                x=cluster_spend.index, y=cluster_spend["pred_total"],
                marker_color=T_PRIMARY, opacity=0.85,
                text=cluster_spend["pred_total"].apply(fmt_m),
                textposition="outside", textfont=dict(size=10, color=T_PRIMARY),
            )
            fig_spend.update_layout(
                barmode="group",
                paper_bgcolor=_T["plot_paper"], plot_bgcolor=_T["plot_bg"],
                font=dict(family="'Inter', sans-serif", color=_T["plot_font"]),
                xaxis=dict(tickangle=-20, gridcolor=_T["grid_alt"]),
                yaxis=dict(tickformat="$,.0f", title="", gridcolor=_T["grid_alt"], zeroline=False),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            font=dict(size=11)),
                height=380,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_spend, use_container_width=True)

    # ── Cap Hit vs Predicted Value chart ─────────────────────────────────────
    ufa_key = f"team_{team_code}_show_ufa"
    if ufa_key not in st.session_state:
        st.session_state[ufa_key] = False

    chart_data = team.copy()
    if st.session_state[ufa_key]:
        ufa_players = team_all[team_all["cap_hit"].isna()].copy()
        if not ufa_players.empty:
            ufa_players["cap_hit"] = 0
            chart_data = pd.concat([team, ufa_players], ignore_index=True)

    # Sort by cluster group, then by cap_hit within each group
    _cluster_rank = {c: i for i, c in enumerate(CLUSTER_ORDER)}
    chart_data["_cl_rank"] = chart_data["cluster_label"].map(_cluster_rank).fillna(99)
    sorted_data = chart_data.sort_values(["_cl_rank", "cap_hit"], ascending=[True, False])
    sorted_data = sorted_data.drop(columns=["_cl_rank"])

    fig = go.Figure()
    fig.add_bar(
        name="Cap Hit",
        x=sorted_data["name"], y=sorted_data["cap_hit"],
        marker_color=T_SECOND, opacity=0.9,
        text=sorted_data["cap_hit"].apply(lambda v: fmt_m(v) if v > 0 else "UFA"),
        textposition="outside", textfont=dict(size=11, color=T_SECOND),
    )
    fig.add_bar(
        name="Predicted Value",
        x=sorted_data["name"], y=sorted_data["predicted_value"],
        marker_color=T_PRIMARY, opacity=0.9,
    )
    fig.update_layout(
        barmode="group",
        paper_bgcolor=_T["plot_paper"], plot_bgcolor=_T["plot_bg"],
        font=dict(family="'Inter', sans-serif", color=_T["plot_font"]),
        xaxis=dict(tickangle=-40, gridcolor=_T["grid_alt"]),
        yaxis=dict(tickformat="$,.0f", title="", gridcolor=_T["grid_alt"], zeroline=False),
        title=dict(text="Cap Hit vs. Predicted Market Value (grouped by role)",
                   font=dict(color=T_PRIMARY, size=16)),
        showlegend=False,
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    ufa_count = len(team_all[team_all["cap_hit"].isna()])
    if ufa_count > 0:
        ufa_label = "Hide UFA Players" if st.session_state[ufa_key] else f"Show UFA / Unsigned ({ufa_count})"
        if st.button(ufa_label, key=f"{team_code}_ufa_btn"):
            st.session_state[ufa_key] = not st.session_state[ufa_key]
            st.rerun()
    st.markdown(
        f"<div style='font-size:.7rem;color:{_T['card_subtext']};font-family:\"Inter\",sans-serif;"
        f"letter-spacing:.06em;margin-top:4px;'>"
        f"<span style='display:inline-block;width:10px;height:10px;background:{T_SECOND};"
        f"margin-right:5px;vertical-align:middle;'></span>Cap Hit"
        f"&nbsp;&nbsp;"
        f"<span style='display:inline-block;width:10px;height:10px;background:{T_PRIMARY};"
        f"margin-right:5px;vertical-align:middle;'></span>Predicted Value"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Cap Outlook expander ──────────────────────────────────────────────────
    with st.expander("Cap Outlook — Next Season", expanded=False):
        next_yr_players = team[team["years_left"].fillna(0) >= 2].copy()
        expiring        = team[team["years_left"].fillna(0) <= 1].copy()
        ufa_team        = team_all[team_all["cap_hit"].isna()].copy()

        next_committed = next_yr_players["cap_hit"].sum()
        next_space     = CAP_CEILING - next_committed
        n_need_deals   = len(expiring) + len(ufa_team)

        oc1, oc2, oc3, oc4 = st.columns(4)
        oc1.metric("Committed Next Season", fmt_m(next_committed))
        oc2.metric("Projected Cap Space",   fmt_m(next_space))
        oc3.metric("Contracts Expiring",    len(expiring))
        oc4.metric("Deals Needed",          n_need_deals)

        if not expiring.empty:
            _exp_sub = _T["card_subtext"]
            st.markdown(
                f"<div style='font-family:\"Inter\",sans-serif;font-size:.65rem;"
                f"font-weight:500;letter-spacing:.15em;text-transform:uppercase;"
                f"color:{_exp_sub};margin:12px 0 6px;'>Expiring Contracts</div>",
                unsafe_allow_html=True,
            )
            exp_disp = expiring[["name", "pos", "age", "cap_hit",
                                  "predicted_value", "team_signal"]].copy()
            exp_disp["age"]             = exp_disp["age"].apply(lambda v: f"{v:.0f}" if pd.notna(v) else "?")
            exp_disp["cap_hit"]         = exp_disp["cap_hit"].apply(fmt_m)
            exp_disp["predicted_value"] = exp_disp["predicted_value"].apply(fmt_m)
            exp_disp = exp_disp.rename(columns={
                "name": "Player", "pos": "Pos", "age": "Age",
                "cap_hit": "Current Cap Hit", "predicted_value": "Predicted Value",
                "team_signal": "Re-sign Signal",
            })
            st.dataframe(exp_disp, use_container_width=True, hide_index=True)

    # ── Roster breakdown ──────────────────────────────────────────────────────
    st.markdown(
        f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:1.8rem;"
        f"font-weight:400;color:{_T['page_text']};margin:24px 0 16px;'>Player Breakdown</div>",
        unsafe_allow_html=True,
    )

    tbl = team_all.sort_values("value_delta", ascending=False, na_position="last").copy()

    def _render_group(group_df, group_label):
        if group_df.empty:
            return
        st.markdown(f"<div class='group-label' style='border-left-color:{T_PRIMARY};'>{group_label}</div>", unsafe_allow_html=True)
        for _, row in group_df.iterrows():
            _render_player_row(row)

    def _render_player_row(row):
        pid       = row.get("player_id")
        name      = row.get("name", "?")
        pos       = row.get("pos", "?")
        age       = row.get("age")
        age_str   = f"{age:.0f}" if pd.notna(age) else "?"
        ch        = row.get("cap_hit")
        pv        = row.get("predicted_value")
        delta     = row.get("value_delta")
        delta_pct = row.get("value_delta_pct")
        exp_yr    = row.get("expiry_year")
        exp_st    = row.get("expiry_status") or "—"
        yrs       = row.get("years_left")
        signal    = row.get("team_signal", "—")
        is_est    = bool(row.get("is_estimated", False))
        has_data  = bool(row.get("has_contract_data", False))
        cluster   = row.get("cluster_label") or "—"
        perf_raw  = row.get("performance_score")
        perf_str  = f"{perf_raw:+.1f}" if pd.notna(perf_raw) else "—"
        perf_clr  = _T.get("positive", "#2A7A4B") if (pd.notna(perf_raw) and perf_raw >= 0) else _T.get("negative", "#C0392B")

        ch_str = (fmt_m(ch) + ("*" if is_est else "")) if has_data and pd.notna(ch) else "UFA"
        pv_str = fmt_m(pv) if pd.notna(pv) else "—"
        sig_color = KINGS_SIGNAL_PALETTE.get(signal, "#2C3A40")

        delta_str = "—"
        pct_str   = ""
        if pd.notna(delta):
            sign = "+" if delta >= 0 else ""
            clr  = _T.get("positive", "#2A7A4B") if delta >= 0 else _T.get("negative", "#C0392B")
            delta_str = f"<span style='color:{clr};font-weight:700;'>{sign}{fmt_m(delta)}</span>"
            if pd.notna(delta_pct):
                pct_str = f"<span style='color:{clr};font-size:.8rem;'>({sign}{delta_pct:.1f}%)</span>"

        exp_str = f"{int(exp_yr)}" if pd.notna(exp_yr) else "—"
        yrs_str = f"{int(yrs)}" if pd.notna(yrs) else "—"

        hs_html = ""
        if pid and pd.notna(pid):
            hs_url = headshot_url(pid, team_code)
            hs_html = (
                f"<img src='{hs_url}' width='48' height='48' "
                f"style='object-fit:cover;flex-shrink:0;' "
                f"onerror=\"this.style.display='none'\">"
            )

        _bdg_bg  = _T["card_border"]
        _bdg_txt = _T["card_subtext"]
        est_badge = (
            f"<span style='background:{_bdg_bg};color:{_bdg_txt};padding:1px 5px;"
            f"font-family:\"Inter\",sans-serif;font-size:.65rem;letter-spacing:.06em;"
            f"text-transform:uppercase;margin-left:4px;' "
            f"title='Salary estimated'>est*</span>"
        ) if is_est else ""

        has_ext = bool(row.get("has_extension", False))
        ext_note = ""
        if has_ext:
            ext_ch_val  = row.get("extension_cap_hit")
            ext_start_v = row.get("extension_start_year")
            ext_len_v   = row.get("extension_length")
            ext_ch_s    = f"${ext_ch_val/1e6:.2f}M" if ext_ch_val else "?"
            ext_yr_s    = f"{int(ext_start_v)-1}-{str(int(ext_start_v))[-2:]}" if ext_start_v else "?"
            ext_len_s   = f"{int(ext_len_v)}-yr" if ext_len_v else ""
            _ext_sub = _T["card_subtext"]
            ext_note    = (
                f"<div style='margin-top:5px;font-size:.72rem;font-family:\"Inter\",sans-serif;"
                f"letter-spacing:.06em;color:{_ext_sub};text-transform:uppercase;'>"
                f"Extension Signed — {ext_len_s} {ext_ch_s}/yr starting {ext_yr_s}</div>"
            )

        _rtxt = _T["card_text"]; _rsub = _T["card_subtext"]
        st.markdown(
            f"<div class='kings-card' style='display:flex;align-items:center;gap:16px;"
            f"border-top:3px solid {T_PRIMARY};'>"
            f"  {hs_html}"
            f"  <div style='flex:1;min-width:0;'>"
            f"    <div style='display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;'>"
            f"      <span style='font-size:1.05rem;font-weight:400;color:{_rtxt};"
            f"font-family:\"Space Grotesk\",sans-serif;'>{name}</span>"
            f"      {est_badge}"
            f"      <span style='color:{_rsub};font-size:.72rem;font-family:\"Inter\",sans-serif;"
            f"letter-spacing:.06em;text-transform:uppercase;white-space:nowrap;'>{pos} · {age_str}</span>"
            f"    </div>"
            f"    <div style='display:flex;gap:22px;margin-top:8px;flex-wrap:wrap;'>"
            f"      <div><div class='stat-label'>Cap Hit</div>"
            f"           <div class='stat-value'>{ch_str}</div></div>"
            f"      <div><div class='stat-label'>Pred. Value</div>"
            f"           <div class='stat-value'>{pv_str}</div></div>"
            f"      <div><div class='stat-label'>Value Delta</div>"
            f"           <div style='font-size:.88rem;font-family:\"JetBrains Mono\",monospace;'>"
            f"             {delta_str} {pct_str}</div></div>"
            f"      <div><div class='stat-label'>Expiry</div>"
            f"           <div class='stat-value'>{exp_str} ({exp_st})</div></div>"
            f"      <div><div class='stat-label'>Yrs Left</div>"
            f"           <div class='stat-value'>{yrs_str}</div></div>"
            f"      <div><div class='stat-label'>Role</div>"
            f"           <div class='stat-value' style='font-size:.72rem;font-family:\"Inter\",sans-serif;"
            f"letter-spacing:.04em;'>{cluster}</div></div>"
            f"      <div><div class='stat-label'>Perf Score</div>"
            f"           <div style='font-size:.88rem;font-weight:700;font-family:\"JetBrains Mono\",monospace;"
            f"color:{perf_clr};'>{perf_str}</div></div>"
            f"    </div>"
            f"    {ext_note}"
            f"  </div>"
            f"  <div style='text-align:center;flex-shrink:0;'>"
            f"    <div class='stat-label'>Signal</div>"
            f"    <span class='signal-badge' style='background:{sig_color};color:#fff;"
            f"display:inline-block;margin-top:6px;'>{signal}</span>"
            f"  </div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    fwds = tbl[tbl["pos"].isin(["C", "L", "R"])]
    dmen = tbl[tbl["pos"] == "D"]
    _render_group(fwds, "Forwards")
    _render_group(dmen, "Defensemen")

    # ── Signal legend ─────────────────────────────────────────────────────────
    st.markdown("---")
    badges = "".join(
        f"<span class='signal-badge' style='background:{clr};color:#fff;"
        f"white-space:nowrap;margin:3px 4px 3px 0;display:inline-block;'>{lbl}</span>"
        for lbl, clr in KINGS_SIGNAL_PALETTE.items()
    )
    st.markdown(
        f"<div style='color:{_T['card_subtext']};font-size:.8rem;margin-bottom:6px;'>"
        f"Re-sign Signal Legend</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:4px;'>{badges}</div>",
        unsafe_allow_html=True,
    )

    if team_all["is_estimated"].any():
        st.caption("*Salary estimated from position/TOI medians — contract data pending verification")


# ── Tab 3: LA Kings ────────────────────────────────────────────────────────────
def tab_kings(df: pd.DataFrame):
    # Kings-specific header with logo
    kings_logo = team_logo_url("LAK")
    _ktxt = _T["card_text"]; _ksub = _T["card_subtext"]
    st.markdown(
        f"<div style='padding:0 0 16px 0;'>"
        f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:3.5rem;"
        f"font-weight:400;color:{_ktxt};line-height:1;margin-bottom:8px;'>"
        f"Los Angeles Kings</div>"
        f"<div style='color:{_ksub};font-size:.72rem;"
        f"font-family:\"JetBrains Mono\",monospace;letter-spacing:.1em;text-transform:uppercase;'>"
        f"{_season_str(load_season_context())} &nbsp;·&nbsp; Roster Analysis &nbsp;·&nbsp; Comps Model</div>"
        f"<div style='height:1px;background:{_T['card_border']};margin-top:16px;'></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    kings_all = df[df["team"] == "LAK"].copy()
    if kings_all.empty:
        st.warning("No Kings players found in predictions. Run pipeline.py first.")
        return

    kings_all = add_delta_pct(kings_all)
    kings_all["kings_signal"] = kings_all.apply(kings_resign_signal, axis=1)

    kings = kings_all[kings_all["cap_hit"].notna()].copy()

    # ── Cap summary ─────────────────────────────────────────────────────────
    total_committed   = kings["cap_hit"].sum()
    cap_space         = CAP_CEILING - total_committed
    next_yr_committed = kings[kings["years_left"].fillna(0) >= 1]["cap_hit"].sum()
    next_yr_space     = CAP_CEILING - next_yr_committed
    n_expiring        = int((kings["years_left"].fillna(0) <= 1).sum())

    st.markdown(
        f"<div style='height:1px;background:{_T['card_border']};margin-bottom:20px;'></div>",
        unsafe_allow_html=True,
    )
    cap_cols = st.columns(5)
    cap_cols[0].metric("Roster Players", len(kings_all))
    cap_cols[1].metric("Cap Committed",  fmt_m(total_committed))
    cap_cols[2].metric("Cap Space",      fmt_m(cap_space),
                        delta="+space" if cap_space > 0 else "over cap",
                        delta_color="normal" if cap_space > 0 else "inverse")
    cap_cols[3].metric("Next Season Space", fmt_m(next_yr_space))
    cap_cols[4].metric("Expiring Contracts", f"{n_expiring} players")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Cap Hit vs Predicted Value chart ────────────────────────────────────
    if "kings_show_ufa" not in st.session_state:
        st.session_state["kings_show_ufa"] = False

    chart_data = kings.copy()
    if st.session_state["kings_show_ufa"] and not kings_all[kings_all["cap_hit"].isna()].empty:
        ufa_kings = kings_all[kings_all["cap_hit"].isna()].copy()
        ufa_kings["cap_hit"] = 0
        chart_data = pd.concat([kings, ufa_kings], ignore_index=True)

    kings_sorted = chart_data.sort_values("cap_hit", ascending=False)
    fig = go.Figure()
    fig.add_bar(
        name="Cap Hit",
        x=kings_sorted["name"], y=kings_sorted["cap_hit"],
        marker_color=KINGS_SILVER, opacity=0.9,
        text=kings_sorted["cap_hit"].apply(lambda v: fmt_m(v) if v > 0 else "UFA"),
        textposition="outside", textfont=dict(size=11, color=KINGS_SILVER),
    )
    fig.add_bar(
        name="Predicted Value",
        x=kings_sorted["name"], y=kings_sorted["predicted_value"],
        marker_color=KINGS_GOLD, opacity=0.9,
    )
    fig.update_layout(
        barmode="group",
        paper_bgcolor=_T["plot_paper"], plot_bgcolor=_T["plot_bg"],
        font=dict(family="'Inter', sans-serif", color=_T["plot_font"]),
        xaxis=dict(tickangle=-40, gridcolor=_T["grid_alt"]),
        yaxis=dict(tickformat="$,.0f", title="",
                   gridcolor=_T["grid_alt"], zeroline=False),
        title=dict(text="Cap Hit vs. Predicted Market Value",
                   font=dict(color=_T["plot_font"], size=14)),
        showlegend=False,
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    ufa_kings_count = len(kings_all[kings_all["cap_hit"].isna()])
    if ufa_kings_count > 0:
        ufa_label = "Hide UFA Players" if st.session_state["kings_show_ufa"] else f"Show UFA / Unsigned ({ufa_kings_count})"
        if st.button(ufa_label, key="kings_ufa_toggle"):
            st.session_state["kings_show_ufa"] = not st.session_state["kings_show_ufa"]
            st.rerun()
    st.markdown(
        f"<div style='font-size:.75rem;color:{_T['card_subtext']};font-family:\"JetBrains Mono\",monospace;"
        f"letter-spacing:.06em;margin-top:2px;'>"
        f"<span style='display:inline-block;width:12px;height:12px;background:{KINGS_SILVER};"
        f"margin-right:5px;vertical-align:middle;'></span>Cap Hit"
        f"&nbsp;&nbsp;"
        f"<span style='display:inline-block;width:12px;height:12px;background:{KINGS_GOLD};"
        f"margin-right:5px;vertical-align:middle;'></span>Predicted Value"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Cap Outlook: next season ─────────────────────────────────────────────
    with st.expander("Cap Outlook — Next Season", expanded=False):
        next_yr_players = kings[kings["years_left"].fillna(0) >= 2].copy()
        expiring        = kings[kings["years_left"].fillna(0) <= 1].copy()
        ufa_kings       = kings_all[kings_all["cap_hit"].isna()].copy()

        next_committed = next_yr_players["cap_hit"].sum()
        next_space     = CAP_CEILING - next_committed
        n_need_deals   = len(expiring) + len(ufa_kings)

        oc1, oc2, oc3, oc4 = st.columns(4)
        oc1.metric("Committed Next Season", fmt_m(next_committed))
        oc2.metric("Projected Cap Space",   fmt_m(next_space))
        oc3.metric("Contracts Expiring",    len(expiring))
        oc4.metric("Deals Needed",          n_need_deals)

        if not expiring.empty:
            _kexp_sub = _T["card_subtext"]
            st.markdown(
                f"<div style='font-family:\"Inter\",sans-serif;font-size:.65rem;"
                f"font-weight:500;letter-spacing:.15em;text-transform:uppercase;"
                f"color:{_kexp_sub};margin:12px 0 6px;'>Expiring Contracts</div>",
                unsafe_allow_html=True,
            )
            exp_disp = expiring[["name", "pos", "age", "cap_hit",
                                  "predicted_value", "kings_signal"]].copy()
            exp_disp["age"]             = exp_disp["age"].apply(lambda v: f"{v:.0f}" if pd.notna(v) else "?")
            exp_disp["cap_hit"]         = exp_disp["cap_hit"].apply(fmt_m)
            exp_disp["predicted_value"] = exp_disp["predicted_value"].apply(fmt_m)
            exp_disp = exp_disp.rename(columns={
                "name": "Player", "pos": "Pos", "age": "Age",
                "cap_hit": "Current Cap Hit", "predicted_value": "Predicted Value",
                "kings_signal": "Re-sign Signal",
            })
            st.dataframe(exp_disp, use_container_width=True, hide_index=True)

    # ── Roster grouped by F / D ──────────────────────────────────────────────
    st.markdown(
        f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:1.8rem;"
        f"font-weight:400;color:{_T['page_text']};margin:24px 0 16px;'>Player Breakdown</div>",
        unsafe_allow_html=True,
    )

    tbl = kings_all.sort_values("value_delta", ascending=False, na_position="last").copy()

    def _render_kings_group(group_df, group_label):
        """Render one positional group (Forwards / Defensemen) as player cards."""
        if group_df.empty:
            return
        st.markdown(
            f"<div class='group-label'>{group_label}</div>",
            unsafe_allow_html=True,
        )
        for _, row in group_df.iterrows():
            _render_kings_player_card(row)

    def _render_kings_player_card(row):
        """Render one Kings player card."""
        pid       = row.get("player_id")
        name      = row.get("name", "?")
        pos       = row.get("pos", "?")
        age       = row.get("age")
        age_str   = f"{age:.0f}" if pd.notna(age) else "?"
        ch        = row.get("cap_hit")
        pv        = row.get("predicted_value")
        delta     = row.get("value_delta")
        delta_pct = row.get("value_delta_pct")
        exp_yr    = row.get("expiry_year")
        exp_st    = row.get("expiry_status") or "—"
        yrs       = row.get("years_left")
        signal    = row.get("kings_signal", "—")
        is_est    = bool(row.get("is_estimated", False))
        has_data  = bool(row.get("has_contract_data", False))
        cluster   = row.get("cluster_label") or "—"
        perf_raw  = row.get("performance_score")
        perf_str  = f"{perf_raw:+.1f}" if pd.notna(perf_raw) else "—"
        perf_clr  = _T.get("positive", "#2A7A4B") if (pd.notna(perf_raw) and perf_raw >= 0) else _T.get("negative", "#C0392B")

        ch_str = (fmt_m(ch) + ("*" if is_est else "")) if has_data and pd.notna(ch) else "UFA"
        pv_str = fmt_m(pv) if pd.notna(pv) else "—"
        sig_color = KINGS_SIGNAL_PALETTE.get(signal, "#2C3A40")

        delta_str = "—"
        pct_str   = ""
        if pd.notna(delta):
            sign = "+" if delta >= 0 else ""
            clr  = _T.get("positive", "#2A7A4B") if delta >= 0 else _T.get("negative", "#C0392B")
            delta_str = f"<span style='color:{clr};font-weight:700;'>{sign}{fmt_m(delta)}</span>"
            if pd.notna(delta_pct):
                pct_str = f"<span style='color:{clr};font-size:.8rem;'>({sign}{delta_pct:.1f}%)</span>"

        exp_str = f"{int(exp_yr)}" if pd.notna(exp_yr) else "—"
        yrs_str = f"{int(yrs)}" if pd.notna(yrs) else "—"

        hs_html = ""
        if pid and pd.notna(pid):
            hs_url = headshot_url(pid, "LAK")
            hs_html = (
                f"<img src='{hs_url}' width='48' height='48' "
                f"style='object-fit:cover;flex-shrink:0;' "
                f"onerror=\"this.style.display='none'\">"
            )

        _kbdg_bg  = _T["card_border"]
        _kbdg_txt = _T["card_subtext"]
        est_badge = (
            f"<span style='background:{_kbdg_bg};color:{_kbdg_txt};padding:1px 5px;"
            f"font-family:\"Inter\",sans-serif;font-size:.65rem;letter-spacing:.06em;"
            f"text-transform:uppercase;margin-left:4px;' "
            f"title='Salary estimated — contract data pending verification'>est*</span>"
        ) if is_est else ""

        # Extension note for Kings card
        has_ext = bool(row.get("has_extension", False))
        ext_note = ""
        if has_ext:
            ext_ch_val  = row.get("extension_cap_hit")
            ext_start_v = row.get("extension_start_year")
            ext_len_v   = row.get("extension_length")
            ext_ch_s    = f"${ext_ch_val/1e6:.2f}M" if ext_ch_val else "?"
            ext_yr_s    = f"{int(ext_start_v)-1}-{str(int(ext_start_v))[-2:]}" if ext_start_v else "?"
            ext_len_s   = f"{int(ext_len_v)}-yr" if ext_len_v else ""
            _kext_sub = _T["card_subtext"]
            ext_note    = (
                f"<div style='margin-top:5px;font-size:.72rem;font-family:\"Inter\",sans-serif;"
                f"letter-spacing:.06em;text-transform:uppercase;color:{_kext_sub};'>"
                f"Extension Signed — {ext_len_s} {ext_ch_s}/yr starting {ext_yr_s}</div>"
            )

        _kcard_top = _T.get("positive", "#2A7A4B") if pd.notna(delta) and delta >= 0 else _T.get("negative", "#C0392B") if pd.notna(delta) else _T["card_border"]
        st.markdown(
            f"<div class='kings-card' style='display:flex;align-items:center;gap:16px;"
            f"border-top:3px solid {KINGS_GOLD};'>"
            f"  {hs_html}"
            f"  <div style='flex:1;min-width:0;'>"
            f"    <div style='display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;'>"
            f"      <span style='font-size:1.05rem;font-weight:400;color:{_T['card_text']};"
            f"font-family:\"Space Grotesk\",sans-serif;'>{name}</span>"
            f"      {est_badge}"
            f"      <span style='color:{_T['card_subtext']};font-size:.72rem;font-family:\"Inter\",sans-serif;"
            f"letter-spacing:.06em;text-transform:uppercase;white-space:nowrap;'>{pos} · {age_str}</span>"
            f"    </div>"
            f"    <div style='display:flex;gap:22px;margin-top:8px;flex-wrap:wrap;'>"
            f"      <div><div class='stat-label'>Cap Hit</div>"
            f"           <div class='stat-value'>{ch_str}</div></div>"
            f"      <div><div class='stat-label'>Pred. Value</div>"
            f"           <div class='stat-value'>{pv_str}</div></div>"
            f"      <div><div class='stat-label'>Value Delta</div>"
            f"           <div style='font-size:.9rem;font-weight:700;font-family:\"JetBrains Mono\",monospace;'>"
            f"             {delta_str} {pct_str}</div></div>"
            f"      <div><div class='stat-label'>Expiry</div>"
            f"           <div class='stat-value'>{exp_str} ({exp_st})</div></div>"
            f"      <div><div class='stat-label'>Yrs Left</div>"
            f"           <div class='stat-value'>{yrs_str}</div></div>"
            f"      <div><div class='stat-label'>Role</div>"
            f"           <div class='stat-value' style='font-size:.72rem;font-family:\"Inter\",sans-serif;"
            f"letter-spacing:.04em;'>{cluster}</div></div>"
            f"      <div><div class='stat-label'>Perf Score</div>"
            f"           <div style='font-size:.9rem;font-weight:700;font-family:\"JetBrains Mono\",monospace;"
            f"color:{perf_clr};'>{perf_str}</div></div>"
            f"    </div>"
            f"    {ext_note}"
            f"  </div>"
            f"  <div style='text-align:center;flex-shrink:0;'>"
            f"    <div class='stat-label'>Signal</div>"
            f"    <span class='signal-badge' style='background:{sig_color};color:#fff;"
            f"display:inline-block;margin-top:6px;'>{signal}</span>"
            f"  </div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Render forwards then defensemen ─────────────────────────────────────
    fwds = tbl[tbl["pos"].isin(["C", "L", "R"])]
    dmen = tbl[tbl["pos"] == "D"]
    _render_kings_group(fwds, "Forwards")
    _render_kings_group(dmen, "Defensemen")

    # ── Signal legend ────────────────────────────────────────────────────────
    st.markdown("---")
    badges = "".join(
        f"<span class='signal-badge' style='background:{clr};color:#fff;"
        f"white-space:nowrap;margin:3px 4px 3px 0;display:inline-block;'>{lbl}</span>"
        for lbl, clr in KINGS_SIGNAL_PALETTE.items()
    )
    st.markdown(
        f"<div style='color:{KINGS_SILVER};font-size:.8rem;margin-bottom:6px;'>"
        f"Re-sign Signal Legend</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:4px;'>{badges}</div>",
        unsafe_allow_html=True,
    )

    if kings_all["is_estimated"].any():
        st.caption("*Salary estimated from position/TOI medians — contract data pending verification")


# ── Tab 4: Player Search ───────────────────────────────────────────────────────
def _player_card(player: pd.Series, df: pd.DataFrame, shap_vals: pd.DataFrame,
                 comp_pool: pd.DataFrame = None, col_prefix: str = ""):
    name   = player["name"]
    team   = player.get("team",   "N/A")
    pos    = player.get("pos",    "N/A")
    age    = player.get("age",    "N/A")
    pid    = player.get("player_id")
    if isinstance(age, float) and not pd.isna(age):
        age = int(age)

    _raw_ch      = player.get("cap_hit")
    is_estimated = bool(player.get("is_estimated", False))
    has_contract = bool(player.get("has_contract_data", True)) and pd.notna(_raw_ch)
    ch    = float(_raw_ch) if has_contract else None
    pv    = float(player["predicted_value"]) if pd.notna(player.get("predicted_value")) else None
    _dlt  = player.get("value_delta")
    delta = float(_dlt) if pd.notna(_dlt) else None
    delta_pct = (delta / ch * 100) if (delta is not None and ch) else 0

    # Header with headshot
    has_prior = bool(player.get("has_prior_market_data", True))
    exp_status = str(player.get("expiry_status") or "")

    _bdg_bg  = _T["card_border"]
    _bdg_txt = _T["card_subtext"]
    prior_badge = (
        "" if has_prior else
        f"<span style='background:{_bdg_bg};color:{_bdg_txt};padding:1px 6px;"
        f"font-family:\"Inter\",sans-serif;font-size:.68rem;letter-spacing:.08em;"
        f"margin-left:8px;text-transform:uppercase;'>No Prior Stats</span>"
    )
    if not has_contract and exp_status.upper() == "UFA":
        contract_badge = (
            f"<span style='background:{_bdg_bg};color:{_bdg_txt};padding:1px 6px;"
            f"font-family:\"Inter\",sans-serif;font-size:.68rem;letter-spacing:.08em;"
            f"margin-left:8px;text-transform:uppercase;'>UFA / Unsigned</span>"
        )
    elif is_estimated:
        contract_badge = (
            f"<span style='background:{_bdg_bg};color:{_bdg_txt};padding:1px 6px;"
            f"font-family:\"Inter\",sans-serif;font-size:.68rem;letter-spacing:.08em;"
            f"margin-left:8px;text-transform:uppercase;' "
            f"title='Salary estimated — contract data pending verification'>Salary Est.*</span>"
        )
    else:
        contract_badge = ""

    _delta_val = delta or 0
    _top_strip = _T.get("positive", "#2A7A4B") if _delta_val >= 0 else _T.get("negative", "#C0392B")
    if not has_contract:
        _top_strip = _T["card_border"]

    hs_html = ""
    if pid and pd.notna(pid):
        hs_url = headshot_url(pid, team)
        hs_html = (
            f"<img src='{hs_url}' width='80' height='80' "
            f"style='object-fit:cover;flex-shrink:0;' "
            f"onerror=\"this.style.display='none'\">"
        )

    _pctxt = _T["card_text"]; _pcsub = _T["card_subtext"]
    st.markdown(
        f"<div class='player-card' style='border-top:3px solid {_top_strip};'>"
        f"  <div style='display:flex;gap:18px;align-items:center;'>"
        f"    {hs_html}"
        f"    <div>"
        f"      <div style='font-size:1.4rem;font-weight:400;color:{_pctxt};"
        f"font-family:\"Space Grotesk\",sans-serif;line-height:1.1;'>"
        f"        {name}{prior_badge}{contract_badge}"
        f"      </div>"
        f"      <div style='color:{_pcsub};margin-top:5px;font-size:.75rem;"
        f"font-family:\"Inter\",sans-serif;letter-spacing:.08em;text-transform:uppercase;'>"
        f"        {team} &nbsp;·&nbsp; {pos} &nbsp;·&nbsp; Age {age}"
        f"      </div>"
        f"    </div>"
        f"  </div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Value metrics
    m1, m2, m3 = st.columns(3)
    cap_label = "Cap Hit*" if is_estimated else "Cap Hit"
    m1.metric(cap_label, fmt_m(ch) if ch else "N/A")
    m2.metric("Predicted Value", fmt_m(pv) if pv else "N/A")
    if delta is not None:
        sign  = "+" if delta >= 0 else ""
        label = "UNDERPAID" if delta >= 0 else "OVERPAID"
        m3.metric("Value Delta",
                  f"{sign}{fmt_m(abs(delta))} ({sign}{delta_pct:.1f}%)",
                  delta=label,
                  delta_color="normal" if delta >= 0 else "inverse")
    else:
        m3.metric("Value Delta", "N/A — no contract data")

    # Contract details
    with st.expander("Contract Details", expanded=True):
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.markdown(
            f"<div class='stat-label'>Contract Length</div>"
            f"<div class='stat-value'>{player.get('length_of_contract','—')} yrs</div>",
            unsafe_allow_html=True,
        )
        cc2.markdown(
            f"<div class='stat-label'>Years Remaining</div>"
            f"<div class='stat-value'>{player.get('years_left','—')}</div>",
            unsafe_allow_html=True,
        )
        cc3.markdown(
            f"<div class='stat-label'>Expiry Status</div>"
            f"<div class='stat-value'>{player.get('expiry_status','—')}</div>",
            unsafe_allow_html=True,
        )
        cc4.markdown(
            f"<div class='stat-label'>Expiry Year</div>"
            f"<div class='stat-value'>{player.get('expiry_year','—')}</div>",
            unsafe_allow_html=True,
        )

    # Extension badge (shown when a future contract is already signed)
    has_ext = bool(player.get("has_extension", False))
    if has_ext:
        ext_ch    = player.get("extension_cap_hit")
        ext_start = player.get("extension_start_year")
        ext_exp   = player.get("extension_expiry_year")
        ext_len   = player.get("extension_length")
        ext_stat  = player.get("extension_expiry_status") or ""
        ext_ch_str   = f"${ext_ch/1e6:.2f}M" if ext_ch else "?"
        ext_yrs_str  = f"{int(ext_len)}-yr" if ext_len else ""
        ext_start_str = f"{int(ext_start)-1}-{str(int(ext_start))[-2:]}" if ext_start else "?"
        ext_exp_str  = f"{int(ext_exp)-1}-{str(int(ext_exp))[-2:]}" if ext_exp else "?"
        _extbg = _T["card_bg"]; _extbd = _T["card_border"]; _extsub = _T["card_subtext"]; _exttxt = _T["card_text"]
        st.markdown(
            f"<div style='background:{_extbg};border:1px solid {_extbd};"
            f"padding:10px 14px;margin:8px 0;'>"
            f"<span style='color:{_exttxt};font-family:\"Inter\",sans-serif;"
            f"font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;'>Extension Signed</span>"
            f"<span style='color:{_extsub};font-family:\"JetBrains Mono\",monospace;"
            f"font-size:.82rem;margin-left:10px;'>"
            f"{ext_yrs_str} · {ext_ch_str}/yr · {ext_start_str} → {ext_exp_str}"
            f"{' · ' + ext_stat if ext_stat else ''}"
            f"</span></div>",
            unsafe_allow_html=True,
        )

    # Re-sign signal + cluster role + performance score
    signal    = player.get("resign_signal", "—")
    sig_color = RESIGN_PALETTE.get(signal, "#555")
    cluster   = player.get("cluster_label") or "—"
    perf_raw  = player.get("performance_score")
    perf_str  = f"{perf_raw:+.1f}" if pd.notna(perf_raw) else "—"
    perf_clr  = _T.get("positive", "#2A7A4B") if (pd.notna(perf_raw) and perf_raw >= 0) else _T.get("negative", "#C0392B")
    _csub = _T["card_subtext"]; _ctxt = _T["card_text"]
    st.markdown(
        f"<div style='margin:12px 0;display:flex;align-items:center;gap:20px;flex-wrap:wrap;'>"
        f"<div style='display:flex;align-items:center;gap:10px;'>"
        f"<span style='color:{_csub};font-size:.65rem;letter-spacing:.15em;"
        f"font-family:\"Inter\",sans-serif;text-transform:uppercase;'>Re-sign Signal</span>"
        f"<span class='signal-badge' style='background:{sig_color};color:#fff;'>{signal}</span>"
        f"</div>"
        f"<div style='display:flex;align-items:center;gap:8px;'>"
        f"<span style='color:{_csub};font-size:.65rem;letter-spacing:.15em;"
        f"font-family:\"Inter\",sans-serif;text-transform:uppercase;'>Role</span>"
        f"<span style='color:{_ctxt};font-size:.82rem;font-family:\"JetBrains Mono\",monospace;"
        f"font-weight:600;'>{cluster}</span>"
        f"</div>"
        f"<div style='display:flex;align-items:center;gap:8px;'>"
        f"<span style='color:{_csub};font-size:.65rem;letter-spacing:.15em;"
        f"font-family:\"Inter\",sans-serif;text-transform:uppercase;'>Perf Score</span>"
        f"<span style='color:{perf_clr};font-size:.9rem;font-weight:700;"
        f"font-family:\"JetBrains Mono\",monospace;'>{perf_str}</span>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # League rank
    rank     = int((df["predicted_value"] >= (pv or 0)).sum())
    total    = len(df)
    rank_pct = pct_rank(df["predicted_value"], pv or 0)
    _rk_sub = _T["card_subtext"]; _rk_txt = _T["page_text"]
    st.markdown(
        f"<div style='color:{_rk_sub};font-size:.72rem;font-family:\"Inter\",sans-serif;"
        f"letter-spacing:.1em;text-transform:uppercase;'>League Rank by Predicted Value: "
        f"<span style='color:{_rk_txt};font-family:\"JetBrains Mono\",monospace;'>#{rank} of {total}</span>"
        f"&nbsp;·&nbsp; Top "
        f"<span style='color:{_rk_txt};font-family:\"JetBrains Mono\",monospace;'>{100-rank_pct}%</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Percentile bars with cluster toggle
    _pct_mode = st.radio("Compare against", ["All Players", "Same Cluster"],
                         horizontal=True, key=f"pct_mode_{col_prefix}")
    if _pct_mode == "Same Cluster":
        pct_pool = df[df["cluster_label"] == cluster]
        _pct_label = f"Stat Percentiles (within {cluster})"
    else:
        pct_pool = df
        _pct_label = "Stat Percentiles (league-wide)"
    st.markdown(f"**{_pct_label}**")

    pct_data = []
    for col, label in PERCENTILE_STATS:
        if col in pct_pool.columns and col in player.index and pd.notna(player.get(col)):
            val = player[col]
            pct = pct_rank(pct_pool[col], val)
            pct_data.append({
                "Stat": label,
                "Percentile": pct,
                "Raw": f"{val:.2f}" if isinstance(val, float) else str(val),
            })

    if pct_data:
        pct_df = pd.DataFrame(pct_data)
        fig_pct = px.bar(
            pct_df, x="Percentile", y="Stat", orientation="h",
            color="Percentile", color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            text=pct_df.apply(lambda r: f"{r['Percentile']}th ({r['Raw']})", axis=1),
            height=max(200, len(pct_data) * 44),
        )
        fig_pct.update_traces(textposition="inside", textfont_size=14)
        fig_pct.update_layout(
            paper_bgcolor=_T["plot_paper"], plot_bgcolor=_T["plot_bg"], font_color=_T["plot_font"],
            showlegend=False, coloraxis_showscale=False,
            xaxis=dict(range=[0, 100], title="Percentile", gridcolor=_T["grid"]),
            yaxis=dict(title=""),
            margin=dict(l=0, r=0, t=10, b=10),
        )
        st.plotly_chart(fig_pct, use_container_width=True)

    # ── A. Cluster Placement ──────────────────────────────────────────────────
    st.markdown(
        f"<div style='margin:32px 0 4px 0;font-family:\"Space Grotesk\",sans-serif;"
        f"font-size:2rem;font-weight:400;color:{_T['page_text']};'>"
        f"How {name}'s value is calculated</div>"
        f"<div style='color:{_T['card_subtext']};font-size:.72rem;margin-bottom:20px;"
        f"font-family:\"JetBrains Mono\",monospace;letter-spacing:.08em;text-transform:uppercase;'>"
        f"Cluster placement → within-cluster scoring → 5 closest comps → weighted value</div>",
        unsafe_allow_html=True,
    )

    _cbg = _T["card_bg"]; _cbd = _T["card_border"]; _csub2 = _T["card_subtext"]; _ctxt2 = _T["card_text"]
    _cl_color = _T.get("card_text", "#F2EEE5")

    st.markdown(
        f"<div style='font-family:\"Space Grotesk\",sans-serif;font-weight:400;font-size:1.5rem;"
        f"color:{_cl_color};margin-bottom:10px;"
        f"border-left:3px solid {_cl_color};padding-left:10px;'>"
        f"Step 1 — Cluster: {cluster}</div>",
        unsafe_allow_html=True,
    )

    # Cluster feature comparison bars
    cluster_features = [
        ("toi_per_g", "TOI / Game"),
        ("pp_pts", "PP Points"),
        ("plus_minus", "Plus/Minus"),
    ]
    if player.get("pos") == "C":
        cluster_features.append(("faceoff_pct", "Faceoff %"))

    cluster_mates = df[df["cluster_label"] == cluster]
    cf_data = []
    for feat, label in cluster_features:
        p_val = float(player.get(feat) or 0)
        c_avg = float(cluster_mates[feat].mean()) if feat in cluster_mates.columns else 0
        cf_data.append({"Feature": label, "Value": p_val, "Type": name})
        cf_data.append({"Feature": label, "Value": c_avg, "Type": f"{cluster} Avg"})

    if cf_data:
        cf_df = pd.DataFrame(cf_data)
        fig_cf = px.bar(
            cf_df, x="Value", y="Feature", color="Type", barmode="group",
            orientation="h", height=max(180, len(cluster_features) * 55),
            color_discrete_map={name: _cl_color, f"{cluster} Avg": _T.get("grid", "#333")},
        )
        fig_cf.update_layout(
            paper_bgcolor=_T["plot_paper"], plot_bgcolor=_T["plot_bg"],
            font=dict(family="'Inter', sans-serif", color=_T["plot_font"]),
            showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis=dict(gridcolor=_T["grid"], title=""),
            yaxis=dict(title=""),
            margin=dict(l=0, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_cf, use_container_width=True)

    # ── B. Performance Within Cluster ─────────────────────────────────────────
    st.markdown(
        f"<div style='font-family:\"Space Grotesk\",sans-serif;font-weight:400;font-size:1.5rem;"
        f"color:{_cl_color};margin-bottom:10px;margin-top:20px;"
        f"border-left:3px solid {_cl_color};padding-left:10px;'>"
        f"Step 2 — Performance Score: {perf_str}</div>",
        unsafe_allow_html=True,
    )

    if not cluster_mates.empty and "performance_score" in cluster_mates.columns:
        cm_scores = cluster_mates["performance_score"].dropna()
        p_score = float(perf_raw) if pd.notna(perf_raw) else 0
        cm_rank = int((cm_scores >= p_score).sum())
        cm_total = len(cm_scores)

        # Strip plot: all cluster-mates as dots, this player highlighted
        fig_strip = go.Figure()
        fig_strip.add_trace(go.Scatter(
            x=cm_scores, y=[""] * len(cm_scores),
            mode="markers",
            marker=dict(size=8, color=_T.get("grid", "#555"), opacity=0.4),
            name=f"{cluster} players",
            hovertext=cluster_mates.loc[cm_scores.index, "name"],
            hovertemplate="%{hovertext}: %{x:.1f}<extra></extra>",
        ))
        fig_strip.add_trace(go.Scatter(
            x=[p_score], y=[""],
            mode="markers",
            marker=dict(size=16, color=_cl_color, symbol="diamond",
                        line=dict(width=2, color="#fff")),
            name=name,
            hovertemplate=f"{name}: {p_score:+.1f}<extra></extra>",
        ))
        fig_strip.update_layout(
            paper_bgcolor=_T["plot_paper"], plot_bgcolor=_T["plot_bg"],
            font=dict(family="'Inter', sans-serif", color=_T["plot_font"]),
            showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
            xaxis=dict(title="Performance Score", range=[-110, 110],
                       gridcolor=_T["grid"], zeroline=True,
                       zerolinecolor=_T["zero"], zerolinewidth=2),
            yaxis=dict(visible=False),
            height=140, margin=dict(l=0, r=10, t=10, b=30),
        )
        st.plotly_chart(fig_strip, use_container_width=True)
        st.markdown(
            f"<div style='color:{_csub2};font-size:.75rem;font-family:\"JetBrains Mono\",monospace;"
            f"text-align:center;margin-top:-8px;'>"
            f"Ranked <strong style='color:{_ctxt2};'>#{cm_rank}</strong> of {cm_total} "
            f"in the {cluster} cluster</div>",
            unsafe_allow_html=True,
        )

    # ── C. 5 Closest Comps ────────────────────────────────────────────────────
    st.markdown(
        f"<div style='font-family:\"Space Grotesk\",sans-serif;font-weight:400;font-size:1.5rem;"
        f"color:{_cl_color};margin-bottom:10px;margin-top:24px;"
        f"border-left:3px solid {_cl_color};padding-left:10px;'>"
        f"Step 3 — 5 Closest Comps</div>",
        unsafe_allow_html=True,
    )

    comps = _get_player_comps(player, comp_pool) if comp_pool is not None else pd.DataFrame()
    if not comps.empty:
        # Comp cards
        n_comps = min(5, len(comps))
        comp_cols = st.columns(n_comps)
        for ci, (_, comp) in enumerate(comps.head(n_comps).iterrows()):
            c_pid  = comp.get("player_id")
            c_name = comp.get("name", "?")
            c_team = comp.get("team", "?")
            c_pos  = comp.get("pos", "?")
            c_age  = comp.get("age")
            c_ch   = comp.get("cap_hit")
            c_wt   = comp.get("_weight", 0)
            c_same = bool(comp.get("_same", 0))
            c_cl   = comp.get("cluster_label", "")

            c_hs = ""
            if c_pid and pd.notna(c_pid):
                c_hs_url = headshot_url(c_pid, c_team)
                c_hs = (
                    f"<img src='{c_hs_url}' width='48' height='48' "
                    f"style='object-fit:cover;display:block;margin:0 auto;' "
                    f"onerror=\"this.style.display='none'\">"
                )

            same_badge = (
                f"<div style='font-size:.55rem;color:{_cl_color};font-family:\"Inter\",sans-serif;"
                f"letter-spacing:.1em;text-transform:uppercase;margin-top:4px;'>SAME CLUSTER</div>"
                if c_same else
                f"<div style='font-size:.55rem;color:{_csub2};font-family:\"Inter\",sans-serif;"
                f"letter-spacing:.1em;text-transform:uppercase;margin-top:4px;'>{c_cl}</div>"
            )
            ch_str = f"${c_ch/1e6:.2f}M" if c_ch and pd.notna(c_ch) else "—"
            wt_pct = f"{c_wt:.2f}"
            age_s = f"{c_age:.0f}" if pd.notna(c_age) else "?"

            comp_cols[ci].markdown(
                f"<div style='background:{_cbg};padding:12px 8px;"
                f"border:1px solid {_cbd};text-align:center;"
                f"border-left:3px solid {_cbd};'>"
                f"  {c_hs}"
                f"  <div style='font-weight:400;color:{_ctxt2};margin-top:6px;"
                f"font-size:.85rem;font-family:\"Space Grotesk\",sans-serif;"
                f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{c_name}</div>"
                f"  <div style='color:{_csub2};font-size:.65rem;margin:2px 0;"
                f"font-family:\"Inter\",sans-serif;letter-spacing:.06em;'>"
                f"    {c_team} · {c_pos} · {age_s}"
                f"  </div>"
                f"  {same_badge}"
                f"  <div style='margin-top:6px;font-size:.82rem;"
                f"font-family:\"JetBrains Mono\",monospace;color:{_ctxt2};'>{ch_str}</div>"
                f"  <div style='color:{_csub2};font-size:.6rem;"
                f"font-family:\"JetBrains Mono\",monospace;margin-top:2px;'>wt: {wt_pct}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Comp AAV bar chart with weighted average line
        comp_chart = comps.head(n_comps).copy()
        comp_chart["cap_hit_m"] = pd.to_numeric(comp_chart["cap_hit"], errors="coerce") / 1e6
        comp_chart["label"] = comp_chart["name"].str.split().str[-1]  # last name
        weighted_avg = pv / 1e6 if pv else 0

        fig_comps = px.bar(
            comp_chart, x="cap_hit_m", y="label", orientation="h",
            color="cluster_label",             labels={"cap_hit_m": "Cap Hit ($M)", "label": ""},
            height=max(180, n_comps * 38),
        )
        fig_comps.add_vline(
            x=weighted_avg, line_dash="dash", line_color="#fff", line_width=2,
            annotation_text=f"Predicted: ${weighted_avg:.2f}M",
            annotation_position="top right",
            annotation_font=dict(color=_T["plot_font"], size=12),
        )
        fig_comps.update_layout(
            paper_bgcolor=_T["plot_paper"], plot_bgcolor=_T["plot_bg"],
            font=dict(family="'Inter', sans-serif", color=_T["plot_font"]),
            showlegend=False,
            xaxis=dict(title="Cap Hit ($M)", gridcolor=_T["grid"], tickprefix="$"),
            yaxis=dict(title=""),
            margin=dict(l=0, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_comps, use_container_width=True)
        st.caption("Dashed line = weighted average AAV of comps = predicted market value.")
    else:
        st.caption("No comparable players found (player may lack performance score data).")

    # Estimated Market Value result box
    pv_val   = pv or 0
    pv_color = _T.get("positive", "#2A7A4B") if (delta or 0) >= 0 else _T.get("negative", "#C0392B")
    cap_str  = f"${ch/1e6:.2f}M" if ch else "—"
    if delta is not None:
        delta_str = f"+${delta/1e6:.2f}M" if delta >= 0 else f"-${abs(delta)/1e6:.2f}M"
    else:
        delta_str = "—"
    _mv_bg  = _T["card_bg"]
    _mv_bd  = _T["card_border"]
    _mv_sub = _T["card_subtext"]
    _mv_txt = _T["card_text"]
    st.markdown(
        f"<div style='background:{_mv_bg};border:1px solid {_mv_bd};border-radius:8px;"
        f"padding:20px 24px;margin-top:20px;width:100%;'>"
        f"<div style='font-size:.66rem;color:{_mv_sub};font-family:\"JetBrains Mono\",monospace;"
        f"letter-spacing:.14em;text-transform:uppercase;margin-bottom:10px;'>"
        f"Estimated Market Value</div>"
        f"<div style='font-size:2.2rem;font-weight:500;color:{_mv_txt};"
        f"font-variation-settings:\"opsz\" 144;letter-spacing:-0.02em;"
        f"font-family:\"Space Grotesk\",sans-serif;margin-bottom:10px;'>"
        f"${pv_val/1e6:.2f}M</div>"
        f"<div style='font-size:.8rem;color:{_mv_sub};font-family:\"JetBrains Mono\",monospace;'>"
        f"Cap Hit: <span style='color:{_mv_txt};'>{cap_str}</span>"
        f"&nbsp;&nbsp;·&nbsp;&nbsp;"
        f"Delta: <span style='color:{pv_color};font-weight:600;'>{delta_str}</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # ── Demoted: XGBoost SHAP breakdown ───────────────────────────────────────
    if (not shap_vals.empty and "name" in shap_vals.columns
            and name in shap_vals["name"].values):
        with st.expander("XGBoost Benchmark — SHAP Analysis", expanded=False):
            st.caption("These SHAP values are from the XGBoost validation model, not the primary comps model. Shown for reference.")
            row_shap     = shap_vals[shap_vals["name"] == name].iloc[0].drop("name")
            vals_dollars = row_shap.astype(float) * CAP_CEILING
            pos_factors = vals_dollars[vals_dollars > 0].nlargest(5)
            neg_factors = vals_dollars[vals_dollars < 0].nsmallest(5)
            all_factors = pd.concat([neg_factors, pos_factors]).sort_values()
            if not all_factors.empty:
                sf_df = pd.DataFrame({
                    "Feature": [_label(f) for f in all_factors.index],
                    "SHAP": all_factors.values,
                })
                fig_shap = px.bar(
                    sf_df, x="SHAP", y="Feature", orientation="h",
                    color="SHAP", color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0, height=max(250, len(sf_df) * 32),
                )
                fig_shap.update_layout(
                    paper_bgcolor=_T["plot_paper"], plot_bgcolor=_T["plot_bg"],
                    font=dict(family="'Inter', sans-serif", color=_T["plot_font"]),
                    showlegend=False, coloraxis_showscale=False,
                    xaxis=dict(title="Dollar Impact", gridcolor=_T["grid"],
                               zeroline=True, zerolinecolor=_T["zero"]),
                    yaxis=dict(title=""), margin=dict(l=0, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_shap, use_container_width=True)


def tab_player_search(df: pd.DataFrame, shap_vals: pd.DataFrame, full_df: pd.DataFrame):
    st.markdown(f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:2.2rem;color:{_T['page_text']};letter-spacing:0;margin:0 0 20px;font-weight:400;'>Player Search</div>", unsafe_allow_html=True)

    # Build comp pool from full (unfiltered) data
    comp_pool = _build_comp_pool(full_df)

    all_names = sorted(full_df["name"].dropna().tolist())
    search1 = st.text_input(
        "Search by name (partial match)",
        placeholder="e.g. byf, kopitar, mcdavid, kempe…",
        key="search1",
    )
    matches1 = [n for n in all_names if search1.lower() in n.lower()] if search1 else all_names
    if not matches1:
        st.warning(f"No players matching '{search1}'")
        return

    selected1 = st.selectbox("Select player", matches1, key="sel1")
    player1   = full_df[full_df["name"] == selected1].iloc[0]

    compare = st.checkbox("Compare with another player")
    if compare:
        search2  = st.text_input("Second player", placeholder="e.g. mcdavid, draisaitl…",
                                  key="search2")
        matches2 = [n for n in all_names if search2.lower() in n.lower()] if search2 else all_names
        sel2     = st.selectbox("Select", matches2, key="sel2")
        player2  = full_df[full_df["name"] == sel2].iloc[0]

        col_a, col_b = st.columns(2)
        with col_a:
            _player_card(player1, full_df, shap_vals, comp_pool, "p1")
        with col_b:
            _player_card(player2, full_df, shap_vals, comp_pool, "p2")
    else:
        _player_card(player1, full_df, shap_vals, comp_pool)


# ── Tab 5: Model Insights ──────────────────────────────────────────────────────
def tab_insights(df: pd.DataFrame):
    st.markdown(
        f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:2.2rem;"
        f"color:{_T['page_text']};letter-spacing:0;margin:0 0 8px;font-weight:400;'>"
        f"Model Deep-Dive</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='color:{_T['card_subtext']};font-size:.75rem;font-family:\"Inter\",sans-serif;"
        f"margin-bottom:20px;line-height:1.6;max-width:700px;'>"
        f"Detailed analysis of the clustering, scoring, and comps engine internals. "
        f"See the League Overview tab for the full pipeline methodology.</div>",
        unsafe_allow_html=True,
    )

    # ── Section 1: Cluster Distribution ───────────────────────────────────────
    st.markdown("---")
    st.markdown(f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:1.7rem;color:{_T['page_text']};letter-spacing:0;margin:0 0 12px;font-weight:400;'>Cluster Distribution</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:{_T['card_subtext']};font-size:.72rem;font-family:\"Inter\",sans-serif;"
        f"margin-bottom:16px;'>K-means groups every skater into one of 7 role clusters. "
        f"Bar heights show how many players fall into each role.</div>",
        unsafe_allow_html=True,
    )

    if "cluster_label" in df.columns:
        cluster_counts = df["cluster_label"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Players"]
        # Sort by CLUSTER_ORDER
        _order = {c: i for i, c in enumerate(CLUSTER_ORDER)}
        cluster_counts["_ord"] = cluster_counts["Cluster"].map(_order).fillna(99)
        cluster_counts = cluster_counts.sort_values("_ord", ascending=False).drop(columns=["_ord"])

        fig_cl = px.bar(
            cluster_counts, x="Players", y="Cluster", orientation="h",
            color="Cluster",             height=max(300, len(cluster_counts) * 48),
        )
        fig_cl.update_layout(
            paper_bgcolor=_T["plot_paper"], plot_bgcolor=_T["plot_bg"],
            font=dict(family="'Inter', sans-serif", color=_T["plot_font"]),
            showlegend=False,
            xaxis=dict(title="Number of Players", gridcolor=_T["grid"]),
            yaxis=dict(title=""),
            margin=dict(l=10, r=20, t=10, b=10),
        )
        fig_cl.update_traces(texttemplate="%{x}", textposition="inside", textfont_size=13)
        st.plotly_chart(fig_cl, use_container_width=True)

    # ── Section 3: Performance Score Distribution ─────────────────────────────
    st.markdown("---")
    st.markdown(f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:1.7rem;color:{_T['page_text']};letter-spacing:0;margin:0 0 12px;font-weight:400;'>Performance Score Distribution</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:{_T['card_subtext']};font-size:.72rem;font-family:\"Inter\",sans-serif;"
        f"margin-bottom:16px;'>Each player is scored −100 to +100 within their cluster. "
        f"A score of 0 means cluster-average; +100 is the top performer in that role.</div>",
        unsafe_allow_html=True,
    )

    if "performance_score" in df.columns and "cluster_label" in df.columns:
        _ps_df = df[["performance_score", "cluster_label"]].dropna()
        if not _ps_df.empty:
            fig_ps = px.histogram(
                _ps_df, x="performance_score", color="cluster_label",
                nbins=40,
                barmode="stack",
                labels={"performance_score": "Performance Score", "cluster_label": "Cluster"},
                height=380,
            )
            fig_ps.update_layout(
                paper_bgcolor=_T["plot_paper"], plot_bgcolor=_T["plot_bg"],
                font=dict(family="'Inter', sans-serif", color=_T["plot_font"]),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
                xaxis=dict(title="Performance Score", gridcolor=_T["grid"],
                           zeroline=True, zerolinecolor=_T["zero"], zerolinewidth=2),
                yaxis=dict(title="Number of Players", gridcolor=_T["grid"]),
                margin=dict(l=10, r=20, t=10, b=10),
            )
            st.plotly_chart(fig_ps, use_container_width=True)

            scores = _ps_df["performance_score"]
            ps_c1, ps_c2, ps_c3, ps_c4 = st.columns(4)
            ps_c1.metric("Mean", f"{scores.mean():+.1f}")
            ps_c2.metric("Std Dev", f"{scores.std():.1f}")
            ps_c3.metric("Min", f"{scores.min():+.1f}")
            ps_c4.metric("Max", f"{scores.max():+.1f}")

    # ── Section 4: Comps Engine — Distance Weights ────────────────────────────
    st.markdown("---")
    st.markdown(f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:1.7rem;color:{_T['page_text']};letter-spacing:0;margin:0 0 12px;font-weight:400;'>Comps Engine — Distance Weights</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:{_T['card_subtext']};font-size:.72rem;font-family:\"Inter\",sans-serif;"
        f"margin-bottom:16px;'>The comps engine finds the 5 most similar players using a weighted distance "
        f"across three dimensions. UFA contracts receive 1.5× weight as freely-negotiated market signals.</div>",
        unsafe_allow_html=True,
    )

    _wt_bg = _T["card_bg"]; _wt_bd = _T["card_border"]
    _wt_txt = _T["card_text"]; _wt_sub = _T["card_subtext"]
    _wt_data = [
        ("P/60 Proximity", "45%", 0.45, "#2E6B62"),
        ("Performance Score", "30%", 0.30, "#4A5E80"),
        ("Age Proximity", "25%", 0.25, "#6B5240"),
    ]
    wt_cols = st.columns(3)
    for i, (wt_name, wt_pct, wt_frac, wt_clr) in enumerate(_wt_data):
        wt_cols[i].markdown(
            f"<div style='background:{_wt_bg};border:1px solid {_wt_bd};padding:16px 14px;'>"
            f"<div style='font-family:\"Inter\",sans-serif;font-size:.65rem;letter-spacing:.12em;"
            f"text-transform:uppercase;color:{_wt_sub};'>{wt_name}</div>"
            f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:1.8rem;color:{_wt_txt};"
            f"margin:8px 0 10px;font-weight:400;'>{wt_pct}</div>"
            f"<div style='height:4px;background:{_wt_bd};'>"
            f"<div style='height:4px;background:{wt_clr};width:{int(wt_frac*100)}%;'></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f"<div style='color:{_wt_sub};font-size:.72rem;font-family:\"Inter\",sans-serif;"
        f"margin-top:10px;'>UFA contracts receive 1.5× weight as freely-negotiated market signals.</div>",
        unsafe_allow_html=True,
    )

    # ── Section 5: Cross-Cluster Value Map ───────────────────────────────────
    st.markdown("---")
    st.markdown(f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:1.7rem;color:{_T['page_text']};letter-spacing:0;margin:0 0 12px;font-weight:400;'>Cross-Cluster Value Map</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:{_T['card_subtext']};font-size:.72rem;font-family:\"Inter\",sans-serif;"
        f"margin-bottom:16px;'>Every player plotted by performance score vs cap hit, colored by cluster. "
        f"Shows how the market prices different roles and performance levels.</div>",
        unsafe_allow_html=True,
    )

    _map_df = df[df["cap_hit"].notna() & df["performance_score"].notna()].copy()
    if not _map_df.empty:
        fig_map = px.scatter(
            _map_df, x="performance_score", y="cap_hit",
            color="cluster_label",             hover_data=["name", "team", "pos", "age"],
            labels={"performance_score": "Performance Score", "cap_hit": "Cap Hit ($)",
                    "cluster_label": "Cluster"},
            height=500,
        )
        fig_map.update_layout(
            paper_bgcolor=_T["plot_paper"], plot_bgcolor=_T["plot_bg"],
            font=dict(family="'Inter', sans-serif", color=_T["plot_font"]),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
            xaxis=dict(gridcolor=_T["grid"], zeroline=True, zerolinecolor=_T["zero"]),
            yaxis=dict(tickformat="$,.0f", gridcolor=_T["grid"]),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig_map.update_traces(marker=dict(size=6, opacity=0.75, line=dict(width=0)))
        st.plotly_chart(fig_map, use_container_width=True)

    # ── Section 6: Cluster Comparison Table ───────────────────────────────────
    st.markdown("---")
    st.markdown(f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:1.7rem;color:{_T['page_text']};letter-spacing:0;margin:0 0 12px;font-weight:400;'>Cluster Comparison</div>", unsafe_allow_html=True)

    if "cluster_label" in df.columns:
        _cl_rows = []
        for cl in CLUSTER_ORDER:
            cl_sub = df[df["cluster_label"] == cl]
            if cl_sub.empty:
                continue
            _cl_rows.append({
                "Cluster": cl,
                "Players": len(cl_sub),
                "Avg Age": cl_sub["age"].mean(),
                "Avg Cap Hit": cl_sub["cap_hit"].dropna().mean(),
                "Avg Predicted": cl_sub["predicted_value"].dropna().mean(),
                "Avg Delta": cl_sub["value_delta"].dropna().mean(),
                "Avg Perf Score": cl_sub["performance_score"].dropna().mean(),
            })
        if _cl_rows:
            _cl_table = pd.DataFrame(_cl_rows)
            # Build HTML table
            _hdr_style = (f"font-family:\"Inter\",sans-serif;font-size:.65rem;letter-spacing:.1em;"
                          f"text-transform:uppercase;color:{_T['card_text']};padding:8px 10px;"
                          f"border-bottom:2px solid {_T['card_border']};text-align:right;")
            _cell_style = (f"font-family:\"JetBrains Mono\",monospace;font-size:.78rem;color:{_T['card_text']};"
                           f"padding:8px 10px;border-bottom:1px solid {_T['card_border']};text-align:right;")
            _name_style = (f"font-family:\"Inter\",sans-serif;font-size:.78rem;font-weight:600;"
                           f"padding:8px 10px;border-bottom:1px solid {_T['card_border']};text-align:left;")

            _rows_html = ""
            for _, r in _cl_table.iterrows():
                _cl_c = _T.get("card_text", "#F2EEE5")
                _d_c = _T.get("positive") if r["Avg Delta"] >= 0 else _T.get("negative")
                _rows_html += (
                    f"<tr>"
                    f"<td style='{_name_style}color:{_cl_c};'>{r['Cluster']}</td>"
                    f"<td style='{_cell_style}'>{r['Players']}</td>"
                    f"<td style='{_cell_style}'>{r['Avg Age']:.1f}</td>"
                    f"<td style='{_cell_style}'>{fmt_m(r['Avg Cap Hit'])}</td>"
                    f"<td style='{_cell_style}'>{fmt_m(r['Avg Predicted'])}</td>"
                    f"<td style='{_cell_style}color:{_d_c};'>{fmt_delta(r['Avg Delta'])}</td>"
                    f"<td style='{_cell_style}'>{r['Avg Perf Score']:+.1f}</td>"
                    f"</tr>"
                )
            st.markdown(
                f"<div style='overflow-x:auto;'>"
                f"<table style='width:100%;border-collapse:collapse;background:{_T['card_bg']};'>"
                f"<thead><tr>"
                f"<th style='{_hdr_style}text-align:left;'>Cluster</th>"
                f"<th style='{_hdr_style}'>N</th>"
                f"<th style='{_hdr_style}'>Avg Age</th>"
                f"<th style='{_hdr_style}'>Avg Cap</th>"
                f"<th style='{_hdr_style}'>Avg Pred</th>"
                f"<th style='{_hdr_style}'>Avg Delta</th>"
                f"<th style='{_hdr_style}'>Avg Score</th>"
                f"</tr></thead>"
                f"<tbody>{_rows_html}</tbody>"
                f"</table></div>",
                unsafe_allow_html=True,
            )

    # ── Section 7: Cluster Deep-Dive ──────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:1.7rem;color:{_T['page_text']};letter-spacing:0;margin:0 0 12px;font-weight:400;'>Cluster Deep-Dive</div>", unsafe_allow_html=True)

    _dive_opts = [c for c in CLUSTER_ORDER if c in df["cluster_label"].unique()]
    if _dive_opts:
        _dive_sel = st.selectbox("Select a cluster to explore", _dive_opts, key="insights_cluster_dive")
        _dive_df = df[df["cluster_label"] == _dive_sel].copy()
        _dive_clr = _T.get("card_text", "#F2EEE5")

        # Summary metrics
        dc1, dc2, dc3, dc4, dc5 = st.columns(5)
        dc1.metric("Players", len(_dive_df))
        _pos_bk = _dive_df["pos"].value_counts().to_dict()
        _pos_str = ", ".join(f"{p}: {n}" for p, n in sorted(_pos_bk.items()))
        dc2.metric("Positions", _pos_str)
        dc3.metric("Avg Age", f"{_dive_df['age'].mean():.1f}")
        dc4.metric("Avg Cap Hit", fmt_m(_dive_df["cap_hit"].dropna().mean()))
        _d_avg = _dive_df["value_delta"].dropna().mean()
        dc5.metric("Avg Delta", fmt_delta(_d_avg),
                    delta_color="normal" if _d_avg >= 0 else "inverse")

        # Scatter: performance_score vs cap_hit
        _dive_c = _dive_df[_dive_df["cap_hit"].notna() & _dive_df["performance_score"].notna()]
        if not _dive_c.empty:
            _dive_c = add_delta_pct(_dive_c)
            fig_dive = px.scatter(
                _dive_c, x="performance_score", y="cap_hit",
                color="value_delta", color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                hover_data=["name", "team", "pos", "age", "predicted_value"],
                labels={"performance_score": "Performance Score", "cap_hit": "Cap Hit ($)"},
                height=420,
            )
            fig_dive.update_layout(
                paper_bgcolor=_T["plot_paper"], plot_bgcolor=_T["plot_bg"],
                font=dict(family="'Inter', sans-serif", color=_T["plot_font"]),
                showlegend=False, coloraxis_showscale=True,
                coloraxis_colorbar=dict(title="Delta ($)", tickformat="$,.0f"),
                xaxis=dict(gridcolor=_T["grid"], zeroline=True, zerolinecolor=_T["zero"]),
                yaxis=dict(tickformat="$,.0f", gridcolor=_T["grid"]),
                margin=dict(l=10, r=10, t=10, b=10),
                title=dict(text=f"{_dive_sel} — Performance vs. Cap Hit",
                           font=dict(family="'Inter', sans-serif", color=_dive_clr)),
            )
            fig_dive.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=0.5, color="#000")))
            st.plotly_chart(fig_dive, use_container_width=True)

        # Top underpaid/overpaid within cluster
        _dive_ranked = _dive_df[_dive_df["value_delta"].notna()]
        if len(_dive_ranked) >= 3:
            _di1, _di2 = st.columns(2)
            with _di1:
                st.markdown(
                    f"<div style='font-family:\"Inter\",sans-serif;font-size:.7rem;font-weight:500;"
                    f"letter-spacing:.12em;text-transform:uppercase;color:{_T['positive']};margin-bottom:8px;"
                    f"border-left:2px solid {_T['positive']};padding-left:8px;'>"
                    f"Top 3 Underpaid {_dive_sel}</div>",
                    unsafe_allow_html=True,
                )
                _mini_player_cards(_dive_ranked.nlargest(3, "value_delta"))
            with _di2:
                st.markdown(
                    f"<div style='font-family:\"Inter\",sans-serif;font-size:.7rem;font-weight:500;"
                    f"letter-spacing:.12em;text-transform:uppercase;color:{_T['negative']};margin-bottom:8px;"
                    f"border-left:2px solid {_T['negative']};padding-left:8px;'>"
                    f"Top 3 Overpaid {_dive_sel}</div>",
                    unsafe_allow_html=True,
                )
                _mini_player_cards(_dive_ranked.nsmallest(3, "value_delta"))

    # ── Section 8: SHAP Feature Importance (XGBoost benchmark) ────────────────
    st.markdown("---")
    st.markdown(f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:1.7rem;color:{_T['page_text']};letter-spacing:0;margin:0 0 12px;font-weight:400;'>XGBoost Benchmark — SHAP Feature Importance</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:{_T['card_subtext']};font-size:.72rem;font-family:\"Inter\",sans-serif;"
        f"margin-bottom:16px;'>SHAP values from the XGBoost validation model show which features "
        f"have the largest impact on predicted salary. This validates the comps model's inputs.</div>",
        unsafe_allow_html=True,
    )

    shap_summary = load_shap_summary()
    shap_vals    = load_shap_values()

    if shap_summary.empty:
        st.info("Run `py -3 pipeline.py` to generate SHAP values.")
        return

    top_n = int(st.number_input("Features to show", min_value=5, max_value=min(30, len(shap_summary)), value=15, step=1))
    top   = shap_summary.head(top_n).copy()
    top["feature"] = top["feature"].apply(_label)

    max_val = top["mean_abs_shap"].max()
    tick_vals = [i * 500_000 for i in range(0, int(max_val / 500_000) + 2)]
    tick_text = [f"${v/1e6:.1f}M" for v in tick_vals]

    fig = px.bar(
        top.sort_values("mean_abs_shap"),
        x="mean_abs_shap", y="feature", orientation="h",
        color="mean_abs_shap", color_continuous_scale="Tealgrn",
        labels={"mean_abs_shap": "Mean |SHAP| ($)", "feature": ""},
        height=max(500, top_n * 34),
    )
    fig.update_layout(
        paper_bgcolor=_T["plot_paper"], plot_bgcolor=_T["plot_bg"],
        font=dict(family="'Inter', sans-serif", color=_T["plot_font"]), showlegend=False, coloraxis_showscale=False,
        xaxis=dict(
            tickvals=tick_vals, ticktext=tick_text,
            title="Avg. Dollar Impact on Prediction", gridcolor=_T["grid"],
        ),
        margin=dict(l=10, r=20, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Mean absolute SHAP value = average dollar impact of each feature on the XGBoost benchmark predictions.")

    if not shap_vals.empty and "name" in shap_vals.columns:
        st.markdown("---")
        st.markdown(f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:1.7rem;color:{_T['page_text']};letter-spacing:0;margin:0 0 12px;font-weight:400;'>Player-Level SHAP Breakdown</div>", unsafe_allow_html=True)
        chosen = st.selectbox("Select a player",
                               sorted(df["name"].dropna().tolist()),
                               key="insights_player")
        if chosen and chosen in shap_vals["name"].values:
            pdata = df[df["name"] == chosen]
            if not pdata.empty:
                p = pdata.iloc[0]
                ic1, ic2, ic3 = st.columns(3)
                ic1.metric("Cap Hit",         fmt_m(p.get("cap_hit")))
                ic2.metric("Predicted Value", fmt_m(p.get("predicted_value")))
                dv = p.get("value_delta")
                ic3.metric("Value Delta", fmt_delta(dv),
                            delta_color="normal" if (dv or 0) >= 0 else "inverse")

            row   = shap_vals[shap_vals["name"] == chosen].iloc[0].drop("name").astype(float)
            top12 = row.abs().nlargest(12).index
            shap_dollars = row[top12] * CAP_CEILING
            pdf   = pd.DataFrame({
                "Feature": [_label(f) for f in top12],
                "SHAP":    shap_dollars.values,
            }).sort_values("SHAP")

            max_abs = pdf["SHAP"].abs().max()
            tick_vals2 = sorted(set(
                [i * 500_000 for i in range(-int(max_abs / 500_000) - 2, int(max_abs / 500_000) + 2)]
            ))
            tick_text2 = [
                (f"+${v/1e6:.1f}M" if v > 0 else (f"-${abs(v)/1e6:.1f}M" if v < 0 else "$0"))
                for v in tick_vals2
            ]

            fig2 = px.bar(
                pdf, x="SHAP", y="Feature", orientation="h",
                color="SHAP", color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                labels={"SHAP": "SHAP Value ($)", "Feature": ""},
                height=max(500, len(top12) * 42 + 80),
            )
            fig2.update_layout(
                paper_bgcolor=_T["plot_paper"], plot_bgcolor=_T["plot_bg"],
                font=dict(family="'Inter', sans-serif", color=_T["plot_font"]), showlegend=False, coloraxis_showscale=False,
                xaxis=dict(
                    tickvals=tick_vals2, ticktext=tick_text2,
                    title="Dollar Impact on Prediction", gridcolor=_T["grid"],
                    zeroline=True, zerolinecolor=_T["zero"], zerolinewidth=2,
                ),
                title=dict(text=f"SHAP Breakdown — {chosen}",
                           font=dict(family="'Inter', sans-serif", color=_T["plot_font"])),
                margin=dict(l=10, r=20, t=40, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
def render_footer(df: pd.DataFrame):
    lu = load_last_updated()
    last_ts = ""
    if lu:
        entry = lu.get("nightly_stats") or lu.get("weekly_contracts") or {}
        ts = entry.get("timestamp", "")
        if ts:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(ts).astimezone()
                last_ts = dt.strftime("%b %d %Y, %I:%M %p")
            except Exception:
                last_ts = ts[:10]

    n_real = int(df["has_contract_data"].fillna(False).sum())
    n_est  = int(df["is_estimated"].sum()) if "is_estimated" in df.columns else 0

    _last_updated = f" · {last_ts}" if last_ts else ""
    st.markdown(
        f"""
        <div class="rink-footer">
          <div>RINK<span class="dot">.</span> &nbsp;NHL Value Model</div>
          <div>NHL API · PuckPedia · {n_real} contracts · {n_est} estimated{_last_updated}</div>
          <div>{_season_str(load_season_context())} · Cap ${CAP_CEILING/1e6:.0f}M</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Initialise theme
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = True
    _dark = st.session_state.get("dark_mode", True)
    _set_theme(_dark)
    _inject_css(_dark)

    # Background data refresh disabled for now — load from cached predictions only
    # start_background_refresh(PROCESSED_DIR)

    # If background job just finished, trigger a rerun so UI picks up fresh mtime
    # (disabled while background refresh is paused)
    # if _refresh_status["done"]:
    #     _refresh_status["done"] = False
    #     st.rerun()

    _ctx = load_season_context()
    _n_teams = int(_ctx.get("n_teams", 32)) if isinstance(_ctx, dict) else 32
    _cap_m = CAP_CEILING / 1e6
    st.markdown(
        f"""
        <div class="rink-brand">
          <div class="logo">RINK<span class="dot">.</span></div>
          <div class="tag">NHL Player Value Model</div>
          <div class="season">{_season_str(_ctx)} &nbsp;·&nbsp; {_n_teams} teams &nbsp;·&nbsp; cap ${_cap_m:.1f}M</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Banner sits below the title so it's never clipped by Streamlit's top toolbar.
    # Disappears automatically when the background thread finishes and st.rerun() fires.
    _started = _refresh_status.get("started_at")
    _timed_out = _started and (time.time() - _started) > 600  # 10 min timeout
    if _timed_out:
        _refresh_status["running"] = False
        _refresh_status["error"] = "Data fetch timed out after 10 minutes."
    if _refresh_status["running"]:
        st.info(
            "Fetching latest NHL stats — data will refresh automatically when ready.",
            icon=":material/refresh:",
        )
    elif _refresh_status["error"] and _timed_out:
        st.warning("Data fetch timed out. Showing last cached data.", icon=":material/warning:")

    df        = load_predictions()
    shap_vals = load_shap_values()
    filtered  = sidebar_filters(df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Leaders",
        "Teams",
        "Search",
        "Insights",
    ])

    # Persist active tab across reruns (theme toggle, UFA toggle, etc.) via
    # sessionStorage in the parent window — survives st.rerun() without full navigation.
    st.markdown("""
<script>
(function() {
  var _KEY = '_st_active_tab';
  function restoreTab() {
    var idx = parseInt(sessionStorage.getItem(_KEY) || '0');
    if (idx <= 0) return;
    var tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
    if (tabs.length > idx) {
      tabs[idx].click();
    } else {
      setTimeout(restoreTab, 80);
    }
  }
  function attachListeners() {
    var tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
    tabs.forEach(function(tab, i) {
      if (!tab.dataset._stTabBound) {
        tab.dataset._stTabBound = '1';
        tab.addEventListener('click', function() {
          sessionStorage.setItem(_KEY, String(i));
        });
      }
    });
  }
  setTimeout(function() { restoreTab(); attachListeners(); }, 250);
  var _obs = new MutationObserver(function() { attachListeners(); });
  _obs.observe(window.parent.document.body, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_html=True)

    with tab1:
        tab_overview(filtered, df)
    with tab2:
        tab_leaderboards(filtered)
    with tab3:
        _all_team_codes = sorted([t for t in df["team"].dropna().unique() if t in TEAM_NAMES])
        _team_labels    = [f"{TEAM_NAMES.get(t, t)} ({t})" for t in _all_team_codes]
        # Initialise session state once; omit index= to avoid fighting Streamlit's
        # own key↔value tracking — session state persists across st.rerun() calls.
        _default_label = f"{TEAM_NAMES.get('LAK','Los Angeles Kings')} (LAK)"
        if "team_tab_sel" not in st.session_state:
            st.session_state["team_tab_sel"] = _default_label if _default_label in _team_labels else _team_labels[0]
        _sel_label = st.selectbox("Select Team", _team_labels, key="team_tab_sel")
        _sel_code  = _all_team_codes[_team_labels.index(_sel_label)] if _sel_label in _team_labels else "LAK"
        tab_team(df, _sel_code)
    with tab4:
        tab_player_search(filtered, shap_vals, df)
    with tab5:
        tab_insights(df)

    render_footer(df)


if __name__ == "__main__":
    main()
