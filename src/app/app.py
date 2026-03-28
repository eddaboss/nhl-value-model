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
        from src.features.build import build_features, get_feature_matrix, resign_label
        from src.models.train import load_model

        df_raw, ctx = load_and_merge()
        df = build_features(df_raw)
        X, _ = get_feature_matrix(df)
        model = load_model("xgb")

        df["predicted_value"] = model.predict(X) * CAP_CEILING
        df["value_delta"] = df.apply(
            lambda r: r["predicted_value"] - r["cap_hit"]
            if r.get("has_contract_data") else None, axis=1
        )
        df["resign_signal"] = df.apply(resign_label, axis=1)

        keep = [
            "name", "team", "pos", "age",
            "cap_hit", "predicted_value", "value_delta",
            "expiry_status", "expiry_year", "years_left", "length_of_contract",
            "gp", "g", "a", "p", "ppg",
            "toi_per_g", "plus_minus", "pim",
            "g60", "p60", "pp_pts", "shots", "shooting_pct",
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


def start_background_refresh(processed_dir: Path) -> None:
    """Launch background pipeline thread once per process lifetime."""
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
    page_icon="🏒",
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
            "page_text":   "#E8E4DC",
            "plot_paper":  "#040404",
            "plot_bg":     "#0d0d1a",
            "plot_font":   "#A0A0A0",
            "grid":        "#1e1e35",
            "grid_alt":    "#1e1e35",
            "zero":        "#2a2a4a",
            "legend_bg":   "#1a1a2e",
        })
    else:
        _T.update({
            "page_text":   "#1C1C1C",
            "plot_paper":  "rgba(0,0,0,0)",
            "plot_bg":     "#EDE9E0",
            "plot_font":   "#333333",
            "grid":        "#D8D3C8",
            "grid_alt":    "#D8D3C8",
            "zero":        "#AAAAAA",
            "legend_bg":   "#1a1a2e",
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
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,700;1,400&family=Manrope:wght@300;400;500;600;700&display=swap" rel="stylesheet">
"""

_DARK_CSS  = """<style>
  .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }
  [data-testid="stMetric"] {
      background: transparent !important; border: none !important;
      border-radius: 0 !important; padding: 14px 2px 10px !important; box-shadow: none !important;
  }
  [data-testid="stMetricLabel"] {
      font-family: 'IBM Plex Mono', monospace !important;
      font-size: 0.88rem !important; letter-spacing: 0.28em !important; text-transform: uppercase !important;
  }
  [data-testid="stMetricValue"] {
      font-family: 'Bebas Neue', cursive !important;
      font-size: 2.6rem !important; line-height: 1.0 !important; letter-spacing: 0.04em !important; font-weight: 400 !important;
  }
  [data-testid="stMetricDelta"] {
      font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78rem !important; letter-spacing: 0.08em !important;
  }
  [data-baseweb="tab-list"] { gap: 0 !important; background: transparent !important; padding-bottom: 0 !important; margin-bottom: 20px !important; }
  [data-baseweb="tab"] {
      font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78rem !important;
      letter-spacing: 0.22em !important; text-transform: uppercase !important;
      padding: 12px 24px !important; border-radius: 0 !important;
      background: transparent !important; border-bottom: 2px solid transparent !important; margin-bottom: -1px !important;
  }
  .player-card { border-radius: 0; padding: 20px 24px; border-left-width: 3px; margin-bottom: 14px; }
  .kings-card  { border-radius: 0; padding: 18px 22px; border-left-width: 3px; margin-bottom: 6px; transition: background 0.1s; }
  .stat-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.2em; }
  .stat-value { font-family: 'IBM Plex Mono', monospace; font-size: 0.95rem; font-weight: 500; }
  .delta-pos  { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; font-weight: 500; }
  .delta-neg  { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; font-weight: 500; }
  .pct-pos    { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
  .pct-neg    { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
  .section-header { font-family: 'Bebas Neue', cursive; font-size: 1.8rem; letter-spacing: 0.04em; margin-bottom: 8px; font-weight: 400; line-height: 1; }
  .group-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.25em; text-transform: uppercase; margin: 20px 0 10px; padding-left: 10px; border-left-width: 2px; border-left-style: solid; }
  .signal-badge { display: inline-block; padding: 3px 8px; border-radius: 0; font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; font-weight: 500; letter-spacing: 0.14em; text-transform: uppercase; }
  h1, h2, h3, [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {
      font-family: 'Bebas Neue', cursive !important; letter-spacing: 0.04em !important; font-weight: 400 !important; line-height: 1.05 !important;
  }
  [data-testid="stExpander"] { border-radius: 0 !important; }
  details summary p { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.85rem !important; letter-spacing: 0.1em !important; }
  [data-testid="stCaptionContainer"] p { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.84rem !important; letter-spacing: 0.08em !important; }
  input, [data-baseweb="input"] input { border-radius: 0 !important; font-family: 'Manrope', sans-serif !important; }
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-thumb { border-radius: 0; }
  hr { margin: 16px 0 !important; }
  html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
  p, label, [data-testid="stMarkdownContainer"], [class*="css"] { font-family: 'Manrope', sans-serif !important; }
  footer { visibility: hidden; }

  :root { --font: 'Manrope', sans-serif; --font-mono: 'IBM Plex Mono', monospace; }
  .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] { background-color: #040404 !important; }
  [data-testid="stMetric"] { border-top: 1px solid #C8A84B !important; }
  [data-testid="stMetricLabel"] { color: #A0A0A0 !important; }
  [data-testid="stMetricValue"] { color: #E8E4DC !important; }
  [data-testid="stMetricDelta"] { color: #888 !important; }
  [data-testid="stSidebar"], section[data-testid="stSidebar"] { background-color: #0d0d1a !important; border-right: 1px solid #1e1e35 !important; }
  [data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] { color: #A0A0A0 !important; }
  [data-baseweb="tab-list"] { border-bottom: 1px solid #1e1e35 !important; }
  [data-baseweb="tab"] { color: #A0A0A0 !important; border-right: 1px solid #1e1e35 !important; }
  [aria-selected="true"][data-baseweb="tab"] { color: #C8A84B !important; border-bottom: 2px solid #C8A84B !important; }
  .player-card { background: #1a1a2e; border: 1px solid #252545; border-left-color: #C8A84B; }
  .kings-card  { background: #1a1a2e; border: 1px solid #252545; border-left-color: #C8A84B; }
  .kings-card:hover { background: #1e2235; }
  .stat-label { color: #A0A0A0; }
  .stat-value { color: #E8E4DC; }
  .delta-pos  { color: #1FBFA0; }
  .delta-neg  { color: #E84040; }
  .pct-pos    { color: #3ED4B6; }
  .pct-neg    { color: #EF7070; }
  .kings-gold { color: #C8A84B; font-weight: 700; }
  .section-header { color: #E8E4DC; }
  .group-label { color: #A0A0A0; border-left-color: #C8A84B; }
  .signal-badge { color: #fff !important; }
  [data-testid="stExpander"] { border: 1px solid #252545 !important; background-color: #1a1a2e !important; }
  [data-testid="stCaptionContainer"] p { color: #A0A0A0 !important; }
  [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 { color: #E8E4DC !important; }
  input, [data-baseweb="input"] input { background: #0C0C0C !important; border-color: #1C1C1C !important; color: #E8E4DC !important; }
  ::-webkit-scrollbar-track { background: #040404; }
  ::-webkit-scrollbar-thumb { background: #1C1C1C; }
  hr { border-color: #1e1e35 !important; }
  /* ── Widget overrides (config.toml base=light → must flip all to dark) ── */
  /* Buttons */
  [data-testid="stButton"] button {
      background: #141414 !important; color: #E8E4DC !important;
      border: 1px solid #2C2C2C !important; border-radius: 0 !important;
      font-family: 'Manrope', sans-serif !important;
  }
  [data-testid="stButton"] button:hover { background: #1E1E1E !important; border-color: #C8A84B !important; }
  /* Selectbox */
  [data-baseweb="select"] > div { background: #0C0C0C !important; border-color: #2C2C2C !important; }
  [data-baseweb="select"] span, [data-baseweb="select"] div { color: #E8E4DC !important; }
  [data-baseweb="select"] svg { fill: #A0A0A0 !important; }
  [data-baseweb="popover"] { background: #0C0C0C !important; }
  [data-baseweb="menu"] { background: #0C0C0C !important; border: 1px solid #2C2C2C !important; }
  [data-baseweb="menu-item"], [role="option"] { color: #E8E4DC !important; background: #0C0C0C !important; }
  [data-baseweb="menu-item"]:hover, [role="option"]:hover { background: #141414 !important; }
  /* Slider track */
  [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child { background: #2C2C2C !important; }
  /* Tick bar labels */
  [data-testid="stSlider"] [data-testid="stTickBarMin"],
  [data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #A0A0A0 !important; }
  /* Dataframe — invert to match dark theme */
  [data-testid="stDataFrameContainer"],
  [data-testid="stDataFrame"],
  .stDataFrame { filter: invert(0.88) hue-rotate(180deg) !important; background: #e8e4dc !important; }
  /* Markdown container general text */
  [data-testid="stMainBlockContainer"] p,
  [data-testid="stMainBlockContainer"] li { color: #E8E4DC !important; }
  /* Caption */
  .stCaptionContainer p { color: #A0A0A0 !important; }
  /* Widget labels */
  [data-testid="stWidgetLabel"] p,
  [data-testid="stWidgetLabel"] label { color: #A0A0A0 !important; }
</style>"""
_LIGHT_CSS = """<style>
  .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }
  [data-testid="stMetric"] {
      background: transparent !important; border: none !important;
      border-radius: 0 !important; padding: 14px 2px 10px !important; box-shadow: none !important;
  }
  [data-testid="stMetricLabel"] {
      font-family: 'IBM Plex Mono', monospace !important;
      font-size: 0.88rem !important; letter-spacing: 0.28em !important; text-transform: uppercase !important;
  }
  [data-testid="stMetricValue"] {
      font-family: 'Bebas Neue', cursive !important;
      font-size: 2.6rem !important; line-height: 1.0 !important; letter-spacing: 0.04em !important; font-weight: 400 !important;
  }
  [data-testid="stMetricDelta"] {
      font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78rem !important; letter-spacing: 0.08em !important;
  }
  [data-baseweb="tab-list"] { gap: 0 !important; background: transparent !important; padding-bottom: 0 !important; margin-bottom: 20px !important; }
  [data-baseweb="tab"] {
      font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78rem !important;
      letter-spacing: 0.22em !important; text-transform: uppercase !important;
      padding: 12px 24px !important; border-radius: 0 !important;
      background: transparent !important; border-bottom: 2px solid transparent !important; margin-bottom: -1px !important;
  }
  .player-card { border-radius: 0; padding: 20px 24px; border-left-width: 3px; margin-bottom: 14px; }
  .kings-card  { border-radius: 0; padding: 18px 22px; border-left-width: 3px; margin-bottom: 6px; transition: background 0.1s; }
  .stat-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.2em; }
  .stat-value { font-family: 'IBM Plex Mono', monospace; font-size: 0.95rem; font-weight: 500; }
  .delta-pos  { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; font-weight: 500; }
  .delta-neg  { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; font-weight: 500; }
  .pct-pos    { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
  .pct-neg    { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
  .section-header { font-family: 'Bebas Neue', cursive; font-size: 1.8rem; letter-spacing: 0.04em; margin-bottom: 8px; font-weight: 400; line-height: 1; }
  .group-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.25em; text-transform: uppercase; margin: 20px 0 10px; padding-left: 10px; border-left-width: 2px; border-left-style: solid; }
  .signal-badge { display: inline-block; padding: 3px 8px; border-radius: 0; font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; font-weight: 500; letter-spacing: 0.14em; text-transform: uppercase; }
  h1, h2, h3, [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {
      font-family: 'Bebas Neue', cursive !important; letter-spacing: 0.04em !important; font-weight: 400 !important; line-height: 1.05 !important;
  }
  [data-testid="stExpander"] { border-radius: 0 !important; }
  details summary p { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.85rem !important; letter-spacing: 0.1em !important; }
  [data-testid="stCaptionContainer"] p { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.84rem !important; letter-spacing: 0.08em !important; }
  input, [data-baseweb="input"] input { border-radius: 0 !important; font-family: 'Manrope', sans-serif !important; }
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-thumb { border-radius: 0; }
  hr { margin: 16px 0 !important; }
  html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
  p, label, [data-testid="stMarkdownContainer"], [class*="css"] { font-family: 'Manrope', sans-serif !important; }
  footer { visibility: hidden; }

  /* ════ LIGHT MODE PALETTE ════ */
  :root { --font: 'Manrope', sans-serif; --font-mono: 'IBM Plex Mono', monospace; }

  /* Page surfaces — cream */
  .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] { background-color: #F4F1EC !important; }

  /* Metrics */
  [data-testid="stMetric"] { border-top: 1px solid #A8861A !important; }
  [data-testid="stMetricLabel"] { color: #555 !important; }
  [data-testid="stMetricValue"] { color: #1a1a1a !important; }
  [data-testid="stMetricDelta"] { color: #555 !important; }

  /* Sidebar — cream bg, DARK readable text */
  [data-testid="stSidebar"], section[data-testid="stSidebar"] { background-color: #EDE9DF !important; border-right: 1px solid #D8D3C8 !important; }
  [data-testid="stSidebar"] * { color: #1a1a1a !important; }
  [data-testid="stSidebar"] .stCaptionContainer p { color: #555 !important; }

  /* Tabs */
  [data-baseweb="tab-list"] { border-bottom: 1px solid #D8D3C8 !important; }
  [data-baseweb="tab"] { color: #666 !important; border-right: 1px solid #D8D3C8 !important; }
  [aria-selected="true"][data-baseweb="tab"] { color: #A8861A !important; border-bottom: 2px solid #A8861A !important; }

  /* ── CARDS: dark navy on cream — intentional editorial contrast ── */
  .player-card { background: #1a1a2e; border: 1px solid #252545; border-left-color: #A8861A; }
  .kings-card  { background: #1a1a2e; border: 1px solid #252545; border-left-color: #A8861A; }
  .kings-card:hover { background: #1e2235; }

  /* Text ON dark navy cards stays white */
  .stat-label { color: #9090b0; }
  .stat-value { color: #E8E4DC; }
  .delta-pos  { color: #1FBFA0; }
  .delta-neg  { color: #E84040; }
  .pct-pos    { color: #3ED4B6; }
  .pct-neg    { color: #EF7070; }
  .kings-gold { color: #C8A84B; font-weight: 700; }

  /* Section headers sit on cream page — use dark text */
  .section-header { color: #1a1a1a; }
  .group-label { color: #9090b0; border-left-color: #A8861A; }
  .signal-badge { color: #fff !important; }

  /* Expanders — light */
  [data-testid="stExpander"] { border: 1px solid #D8D3C8 !important; background-color: #F8F5EE !important; }
  details summary p { color: #1a1a1a !important; }

  /* Captions and general page text */
  [data-testid="stCaptionContainer"] p { color: #666 !important; }
  [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 { color: #1a1a1a !important; }
  [data-testid="stMarkdownContainer"] p { color: #1a1a1a !important; }
  [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color: #333 !important; }

  /* Inputs */
  input, [data-baseweb="input"] input { background: #FFFFFF !important; border-color: #D8D3C8 !important; color: #1a1a1a !important; }
  ::-webkit-scrollbar-track { background: #F4F1EC; }
  ::-webkit-scrollbar-thumb { background: #D8D3C8; }
  hr { border-color: #E0DBD0 !important; }

  /* ── CLASS OVERRIDES for dark-card classes on light page ── */
  /* Ensure stMarkdownContainer class styles match dark-card intent */
  [data-testid="stMarkdownContainer"] .stat-value { color: #E8E4DC !important; }
  [data-testid="stMarkdownContainer"] .stat-label  { color: #9090b0 !important; }
  [data-testid="stMarkdownContainer"] .delta-pos   { color: #1FBFA0 !important; }
  [data-testid="stMarkdownContainer"] .delta-neg   { color: #E84040 !important; }
  [data-testid="stMarkdownContainer"] .kings-gold  { color: #C8A84B !important; }
  [data-testid="stMarkdownContainer"] .group-label { color: #9090b0 !important; }

  /* ── WIDGET OVERRIDES (Streamlit base=light, just need our custom colours) ── */
  [data-testid="stButton"] button {
      background: #EDE9DF !important; color: #1a1a1a !important;
      border: 1px solid #C8C3B8 !important; border-radius: 0 !important;
      font-family: 'Manrope', sans-serif !important;
  }
  [data-testid="stButton"] button:hover { background: #E0DBD0 !important; border-color: #A8861A !important; }
  [data-baseweb="select"] > div { background: #FFFFFF !important; border-color: #C8C3B8 !important; }
  [data-baseweb="select"] span { color: #1a1a1a !important; }
  [data-baseweb="popover"] { background: #FFFFFF !important; }
  [data-baseweb="menu"] { background: #FFFFFF !important; border: 1px solid #D8D3C8 !important; }
  [data-baseweb="menu-item"], [role="option"] { color: #1a1a1a !important; background: #FFFFFF !important; }
  [data-baseweb="menu-item"]:hover, [role="option"]:hover { background: #F0EBE0 !important; }
  /* Tick bar labels */
  [data-testid="stSlider"] [data-testid="stTickBarMin"],
  [data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #888 !important; }
  .stCaptionContainer p { color: #666 !important; }
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
    """Horizontal row of mini headshot cards — used in top-5 lists."""
    cols = st.columns(len(players_df))
    for i, (_, row) in enumerate(players_df.iterrows()):
        pid   = row.get("player_id")
        name  = row.get("name", "?")
        team  = row.get("team", "?")
        pos   = row.get("pos", "?")
        age   = row.get("age")
        delta = row.get(delta_col)
        pv    = row.get("predicted_value")

        clr     = "#1FBFA0" if (delta or 0) >= 0 else "#E84040"
        val_str = fmt_delta(delta) if not show_pred else fmt_m(pv)
        age_str = f"Age {age:.0f}" if pd.notna(age) else ""

        hs_html = ""
        if pid and pd.notna(pid):
            hs_url  = headshot_url(pid, team)
            hs_html = (
                f"<img src='{hs_url}' width='60' height='60' "
                f"style='border-radius:50%;object-fit:cover;"
                f"border:2px solid {clr};display:block;margin:0 auto 6px;' "
                f"onerror=\"this.style.display='none'\">"
            )

        logo_html = (
            f"<img src='{team_logo_url(team)}' width='18' height='18' "
            f"style='vertical-align:middle;margin-right:3px;' "
            f"onerror=\"this.style.display='none'\">"
        )

        cols[i].markdown(
            f"<div style='background:#1a1a2e;border-radius:2px;padding:12px 8px;"
            f"text-align:center;border:1px solid #252545;border-bottom:2px solid {clr};'>"
            f"  {hs_html}"
            f"  <div style='font-weight:700;color:#E8E4DC;font-size:.8rem;"
            f"    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
            f"    max-width:100%;font-family:\"Manrope\",sans-serif;'>{name}</div>"
            f"  <div style='color:#A0A0A0;font-size:.85rem;margin:3px 0;"
            f"    font-family:\"IBM Plex Mono\",monospace;letter-spacing:.04em;'>"
            f"    {team} · {pos}</div>"
            f"  <div style='color:{clr};font-size:.82rem;font-weight:700;"
            f"    margin-top:4px;font-family:\"IBM Plex Mono\",monospace;'>{val_str}</div>"
            f"</div>",
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
    age      = row.get("age") or 30
    delta    = row.get("value_delta") or 0
    yrs_left = int(row.get("years_left") or 0)
    cap_hit  = float(row.get("cap_hit") or 0)
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
        _icon = "🌙" if _dark else "☀️"
        _label = f"{_icon}  Dark Mode" if _dark else f"{_icon}  Light Mode"
        if st.button(_label, key="_theme_btn", use_container_width=True):
            st.session_state["dark_mode"] = not _dark
            st.rerun()
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        st.markdown(
            f"<div style='color:{KINGS_GOLD};font-size:1.0rem;font-weight:700;"
            "margin-bottom:4px;font-family:\"Bebas Neue\",cursive;letter-spacing:0.04em;'>"
            "NHL Value Model</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.markdown("**Filters**")
        positions = ["All", "C", "L", "R", "D", "F (all)"]
        pos_sel  = st.selectbox("Position", positions, key="sb_pos")
        teams_list = ["All"] + sorted(df["team"].dropna().unique().tolist())
        team_sel = st.selectbox("Team", teams_list, key="sb_team")
        import math
        age_min = math.floor(df["age"].dropna().min())
        age_max = math.ceil(df["age"].dropna().max())
        st.markdown("<div style='font-size:.8rem;color:#888;margin-bottom:2px;font-family:\"IBM Plex Mono\",monospace;letter-spacing:.06em;text-transform:uppercase;'>Age range</div>", unsafe_allow_html=True)
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

        st.markdown("---")

        # Season context
        ctx = load_season_context()
        if ctx:
            mode_icon = "🔀" if ctx.get("use_blend") else "📅"
            st.markdown(f"**{mode_icon} Season**")
            st.caption(ctx.get("description", ""))
            st.markdown("---")

        # Model stats (update dynamically once we have results)
        n_players = df["has_contract_data"].fillna(False).sum()
        st.markdown("**Model**")
        st.caption(
            f"Algorithm: XGBoost  \n"
            f"CV R²: 0.829  \n"
            f"CV RMSE: $1.21M  \n"
            f"Training set: {n_players} players"
        )
        st.markdown("---")

        # Data freshness indicator
        if _refresh_status["error"]:
            st.caption(f"Update error: {_refresh_status['error'][:60]}")
        else:
            pred_path = PROCESSED_DIR / "predictions.csv"
            if pred_path.exists():
                from datetime import datetime
                mtime = datetime.fromtimestamp(pred_path.stat().st_mtime)
                label = "Updating..." if _refresh_status["running"] else mtime.strftime('%b %d %Y %I:%M %p')
                st.caption(f"Data: {label}")

    return filt


# ── Tab 1: League Overview ─────────────────────────────────────────────────────
def tab_overview(df: pd.DataFrame, full_df: pd.DataFrame):
    st.markdown(f"<div style='font-family:\"Bebas Neue\",cursive;font-size:2rem;color:{_T['page_text']};letter-spacing:0.04em;margin:0 0 16px;font-weight:400;'>League-Wide Value vs. Cap Hit</div>", unsafe_allow_html=True)

    df_all = df[df["predicted_value"].notna()].copy()
    df_all = add_delta_pct(df_all)
    df_c   = df_all[df_all["cap_hit"].notna()].copy()   # contracted players
    df_ufa = df_all[df_all["cap_hit"].isna()].copy()    # UFA / unsigned

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Skaters", f"{len(df_all):,}",
              delta=f"{len(df_ufa)} UFA/unsigned", delta_color="off")
    avg_cap = df_c["cap_hit"].mean()
    c2.metric("Avg Cap Hit (contracted)", fmt_m(avg_cap))
    if len(df_c):
        best  = df_c.nlargest(1,  "value_delta").iloc[0]
        worst = df_c.nsmallest(1, "value_delta").iloc[0]
        c3.metric("Most Underpaid", best["name"],  delta=fmt_delta(best["value_delta"]))
        c4.metric("Most Overpaid",  worst["name"], delta=fmt_delta(worst["value_delta"]),
                  delta_color="inverse")

    # ── Scatter: contracted players colored by delta ─────────────────────────
    show_ufa = st.session_state.get("overview_show_ufa", False)

    fig = go.Figure()

    # Trace 1 — contracted players (colored by delta)
    if not df_c.empty:
        fig.add_trace(go.Scatter(
            x=df_c["predicted_value"], y=df_c["cap_hit"],
            mode="markers",
            name="Contracted",
            marker=dict(
                size=7, opacity=0.82,
                color=df_c["value_delta"],
                colorscale="RdYlGn",
                cmid=0,
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

    # Trace 2 — UFA/unsigned players (diamond markers, only when toggled on)
    if show_ufa and not df_ufa.empty:
        fig.add_trace(go.Scatter(
            x=df_ufa["predicted_value"], y=[0] * len(df_ufa),
            mode="markers",
            name="UFA / Unsigned",
            marker=dict(
                symbol="diamond", size=11, opacity=0.85,
                color="#C8A84B",
                line=dict(width=1, color="#fff"),
            ),
            customdata=df_ufa[["name", "team", "pos",
                                "predicted_value"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b> — UFA / Unsigned<br>"
                "%{customdata[1]} · %{customdata[2]}<br>"
                "Predicted Market Value: $%{customdata[3]:,.0f}<br>"
                "<extra></extra>"
            ),
        ))

    # Fair-value line
    if not df_c.empty:
        lo = min(df_c["predicted_value"].min(), df_c["cap_hit"].min()) * 0.95
        hi = max(df_c["predicted_value"].max(), df_c["cap_hit"].max()) * 1.02
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                      line=dict(dash="dash", color="#444", width=1.5))
        fig.add_annotation(x=hi * 0.72, y=hi * 0.78, text="Fair value line",
                           showarrow=False, font=dict(color="#444", size=14))

    fig.update_layout(
        paper_bgcolor=_T["plot_paper"], plot_bgcolor=_T["plot_bg"],
        font=dict(family="'IBM Plex Mono', monospace", color=_T["plot_font"]),
        xaxis=dict(tickformat="$,.0f", title="Predicted Market Value",
                   gridcolor=_T["grid"], zeroline=False),
        yaxis=dict(tickformat="$,.0f", title="Actual Cap Hit (0 = UFA/Unsigned)",
                   gridcolor=_T["grid"]),
        showlegend=False,
        height=600,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    event = st.plotly_chart(fig, use_container_width=True,
                            on_select="rerun", key="overview_scatter")

    # ── UFA toggle + color key sit immediately below the chart, always together ─
    _ufa_label = "🔶  Hide UFA / Unsigned" if show_ufa else "🔶  Show UFA / Unsigned"
    if st.button(_ufa_label, key="overview_ufa_toggle"):
        st.session_state["overview_show_ufa"] = not show_ufa
        st.rerun()
    st.caption(
        "🟢 Below the dashed line = underpaid &nbsp;·&nbsp; "
        "🔴 Above = overpaid &nbsp;·&nbsp; "
        "🔶 Diamonds on x-axis = UFA / no current contract &nbsp;·&nbsp; "
        "Click any marker for a quick summary."
    )

    # ── Show clicked player details below the key area ────────────────────────
    if event and event.get("selection") and event["selection"].get("points"):
        pts = event["selection"]["points"]
        if pts:
            cd = pts[0].get("customdata", [])
            clicked_name = cd[0] if cd else ""
            match = full_df[full_df["name"] == clicked_name]
            if not match.empty:
                p = match.iloc[0]
                st.markdown(f"**Selected: {clicked_name}**")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Team / Pos", f"{p.get('team','?')} · {p.get('pos','?')}")
                mc2.metric("Cap Hit",         fmt_m(p.get("cap_hit")))
                mc3.metric("Predicted Value", fmt_m(p.get("predicted_value")))
                dv = p.get("value_delta")
                mc4.metric("Delta", fmt_delta(dv),
                           delta_color="normal" if (dv or 0) >= 0 else "inverse")

    # ── Most Interesting Players ──────────────────────────────────────────────
    if not df_c.empty:
        st.markdown("---")
        mi1, mi2 = st.columns(2)
        with mi1:
            st.markdown(
                "<div style='color:#1FBFA0;font-size:.85rem;font-weight:700;"
                "letter-spacing:.14em;text-transform:uppercase;margin-bottom:10px;"
                "font-family:\"IBM Plex Mono\",monospace;border-left:2px solid #1FBFA0;"
                "padding-left:8px;'>Top 5 Underpaid</div>",
                unsafe_allow_html=True,
            )
            _mini_player_cards(df_c.nlargest(5, "value_delta"))
        with mi2:
            st.markdown(
                "<div style='color:#E84040;font-size:.85rem;font-weight:700;"
                "letter-spacing:.14em;text-transform:uppercase;margin-bottom:10px;"
                "font-family:\"IBM Plex Mono\",monospace;border-left:2px solid #E84040;"
                "padding-left:8px;'>Top 5 Overpaid</div>",
                unsafe_allow_html=True,
            )
            _mini_player_cards(df_c.nsmallest(5, "value_delta"))


# ── Tab 2: Leaderboards ────────────────────────────────────────────────────────
def tab_leaderboards(df: pd.DataFrame):
    st.markdown(f"<div style='font-family:\"Bebas Neue\",cursive;font-size:2rem;color:{_T['page_text']};letter-spacing:0.04em;margin:0 0 16px;font-weight:400;'>Value Leaderboards</div>", unsafe_allow_html=True)

    df_c   = df[df["cap_hit"].notna() & df["value_delta"].notna()].copy()
    df_c   = add_delta_pct(df_c)
    df_ufa = df[df["cap_hit"].isna() & df["predicted_value"].notna()].copy()

    lc1, lc2, lc3 = st.columns([1, 1, 2])
    n = int(lc1.number_input("Show top N", min_value=5, max_value=30, value=15, step=1, key="lb_n"))
    sort_by = lc2.radio("Sort by", ["% Delta", "$ Delta"], horizontal=True, key="lb_sort")
    pos_opts = ["All", "C", "L", "R", "D", "F (all)"]
    pos_f = lc3.radio("Position", pos_opts, horizontal=True, key="lb_pos")

    if pos_f == "F (all)":
        df_c = df_c[df_c["pos"].isin(["C", "L", "R"])]
    elif pos_f != "All":
        df_c = df_c[df_c["pos"] == pos_f]

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
            font=dict(family="'IBM Plex Mono', monospace", color=_T["plot_font"]), showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
            xaxis=dict(
                tickformat=".1f%" if sort_by == "% Delta" else "$,.0f",
                title=x_lbl, gridcolor=_T["grid"],
            ),
            margin=dict(l=0, r=10, t=30, b=10),
            title=dict(text=title, font=dict(family="'IBM Plex Mono', monospace", color=_T["plot_font"])),
        )
        return fig

    with col1:
        gems = df_c.nlargest(n, sort_col)
        st.plotly_chart(bar_chart(gems, "🟢 Hidden Gems — Most Underpaid", "Greens", False),
                        use_container_width=True)

    with col2:
        over = df_c.nsmallest(n, sort_col)
        st.plotly_chart(bar_chart(over, "🔴 Most Overpaid", "Reds_r", True),
                        use_container_width=True)

    st.markdown("---")

    # Headshot row for top 5 gems and top 5 overpaid
    hs1, hs2 = st.columns(2)
    with hs1:
        st.markdown(
            "<div style='color:#1FBFA0;font-size:.85rem;font-weight:700;"
            "letter-spacing:.14em;text-transform:uppercase;margin-bottom:8px;"
            "font-family:\"IBM Plex Mono\",monospace;border-left:2px solid #1FBFA0;"
            "padding-left:8px;'>Top 5 Underpaid</div>",
            unsafe_allow_html=True,
        )
        _mini_player_cards(df_c.nlargest(5, sort_col))
    with hs2:
        st.markdown(
            "<div style='color:#E84040;font-size:.85rem;font-weight:700;"
            "letter-spacing:.14em;text-transform:uppercase;margin-bottom:8px;"
            "font-family:\"IBM Plex Mono\",monospace;border-left:2px solid #E84040;"
            "padding-left:8px;'>Top 5 Overpaid</div>",
            unsafe_allow_html=True,
        )
        _mini_player_cards(df_c.nsmallest(5, sort_col))

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    _TABLE_CSS = (
        "width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;"
        "font-size:.78rem;background:#1a1a2e;"
    )
    _TH_CSS = (
        "padding:7px 10px;text-align:left;color:#A0A0A0;font-weight:600;"
        "letter-spacing:.08em;text-transform:uppercase;border-bottom:1px solid #252545;"
        "background:#141428;"
    )
    _TD_CSS = "padding:6px 10px;color:#E8E4DC;border-bottom:1px solid #1e1e35;"

    def _html_table(data, delta_col="value_delta"):
        cols_order = ["name", "team", "pos", "age", "cap_hit",
                      "predicted_value", "value_delta", "value_delta_pct"]
        col_labels = {
            "name": "Player", "team": "Team", "pos": "Pos", "age": "Age",
            "cap_hit": "Cap Hit", "predicted_value": "Pred. Value",
            "value_delta": "$ Delta", "value_delta_pct": "% Delta",
        }
        cols = [c for c in cols_order if c in data.columns]
        rows_html = ""
        for _, row in data.iterrows():
            delta_v = row.get(delta_col, 0) or 0
            row_bg  = "background:#1a1a2e;" if delta_v >= 0 else "background:#1a1a2e;"
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
                    color = "#1FBFA0" if (v or 0) >= 0 else "#E84040"
                    txt = f"<span style='color:{color};font-weight:700;'>{fmt_delta(v)}</span>"
                elif c == "value_delta_pct":
                    color = "#1FBFA0" if (v or 0) >= 0 else "#E84040"
                    txt = f"<span style='color:{color};'>{fmt_pct(v)}</span>"
                elif c == "name":
                    txt = f"<span style='color:#E8E4DC;font-weight:700;'>{v}</span>"
                elif c == "team":
                    txt = f"<span style='color:#C8A84B;'>{v}</span>"
                else:
                    txt = str(v) if pd.notna(v) else "?"
                cells += f"<td style='{_TD_CSS}'>{txt}</td>"
            rows_html += f"<tr style='{row_bg}'>{cells}</tr>"
        header = "".join(f"<th style='{_TH_CSS}'>{col_labels.get(c, c)}</th>" for c in cols)
        return (
            f"<div style='background:#1a1a2e;border:1px solid #252545;border-radius:2px;"
            f"overflow:hidden;margin-bottom:4px;'>"
            f"<table style='{_TABLE_CSS}'>"
            f"<thead><tr>{header}</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            f"</table></div>"
        )

    t1, t2 = st.columns(2)
    with t1:
        st.markdown(
            "<div style='color:#1FBFA0;font-size:.78rem;font-weight:700;letter-spacing:.1em;"
            "text-transform:uppercase;font-family:\"IBM Plex Mono\",monospace;margin-bottom:6px;'>"
            "Hidden Gems</div>",
            unsafe_allow_html=True,
        )
        st.markdown(_html_table(df_c.nlargest(n, sort_col)), unsafe_allow_html=True)
    with t2:
        st.markdown(
            "<div style='color:#E84040;font-size:.78rem;font-weight:700;letter-spacing:.1em;"
            "text-transform:uppercase;font-family:\"IBM Plex Mono\",monospace;margin-bottom:6px;'>"
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
            st.markdown(
                "<div style='display:flex;align-items:center;gap:10px;'>"
                "<span style='font-size:1.1rem;font-weight:700;color:#C8A84B;'>"
                "🔶 UFA / Unsigned Players</span>"
                "<span style='background:#8B6914;color:#FFF8E0;padding:2px 8px;"
                "border-radius:2px;font-size:.88rem;'>No current contract — predicted value only</span>"
                "</div>",
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
                            txt = f"<span style='color:#C8A84B;font-weight:700;'>{fmt_m(v)}</span>"
                        elif c == "name":
                            txt = f"<span style='color:#E8E4DC;font-weight:700;'>{v}</span>"
                        elif c == "team":
                            txt = f"<span style='color:#C8A84B;'>{v}</span>"
                        else:
                            txt = str(v) if pd.notna(v) else "?"
                        cells += f"<td style='{_TD_CSS}'>{txt}</td>"
                    rows_html += f"<tr>{cells}</tr>"
                header = "".join(f"<th style='{_TH_CSS}'>{col_labels.get(c, c)}</th>" for c in cols)
                return (
                    f"<div style='background:#1a1a2e;border:1px solid #252545;border-radius:2px;"
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


# ── Tab 3: LA Kings ────────────────────────────────────────────────────────────
def tab_kings(df: pd.DataFrame):
    # Kings-specific header with logo
    kings_logo = team_logo_url("LAK")
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:18px;margin-bottom:12px;"
        f"padding:18px 22px;background:#1a1a2e;border-left:4px solid {KINGS_GOLD};'>"
        f"<img src='{kings_logo}' width='64' height='64' "
        f"style='flex-shrink:0;opacity:.95;' onerror=\"this.style.display='none'\">"
        f"<div>"
        f"<div style='font-size:1.5rem;font-weight:700;color:{KINGS_WHITE};"
        f"font-family:\"Bebas Neue\",cursive;letter-spacing:0.04em;line-height:1.1;'>"
        f"Los Angeles Kings</div>"
        f"<div style='color:#A0A0A0;font-size:.78rem;margin-top:5px;"
        f"font-family:\"IBM Plex Mono\",monospace;letter-spacing:.14em;text-transform:uppercase;'>"
        f"{_season_str(load_season_context())} Roster Analysis &nbsp;·&nbsp; XGBoost Model</div>"
        f"</div></div>",
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
        f"<div style='background:#1a1a2e;border-top:3px solid {KINGS_GOLD};"
        "border-radius:3px;padding:16px 20px;margin-bottom:16px;'>",
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
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Cap Hit vs Predicted Value chart ────────────────────────────────────
    kings_sorted = kings.sort_values("cap_hit", ascending=False)
    fig = go.Figure()
    fig.add_bar(
        name="Cap Hit",
        x=kings_sorted["name"], y=kings_sorted["cap_hit"],
        marker_color=KINGS_SILVER, opacity=0.9,
        text=kings_sorted["cap_hit"].apply(fmt_m),
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
        font=dict(family="'IBM Plex Mono', monospace", color=_T["plot_font"]),
        xaxis=dict(tickangle=-40, gridcolor=_T["grid_alt"]),
        yaxis=dict(tickformat="$,.0f", title="",
                   gridcolor=_T["grid_alt"], zeroline=False),
        title=dict(text="Cap Hit vs. Predicted Market Value",
                   font=dict(color=KINGS_GOLD, size=16)),
        legend=dict(bgcolor=_T["legend_bg"], bordercolor=_T["legend_bg"],
                    orientation="h", y=1.08, x=0.5, xanchor="center"),
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Cap Outlook: next season ─────────────────────────────────────────────
    with st.expander("📅 Cap Outlook — Next Season", expanded=False):
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
            st.markdown(
                f"<div style='color:{KINGS_GOLD};font-weight:600;font-size:.85rem;"
                "margin:10px 0 6px;'>Expiring Contracts</div>",
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
        f"<div style='color:{KINGS_GOLD};font-size:1.0rem;font-weight:700;"
        "margin:14px 0 10px;font-family:\"Bebas Neue\",cursive;letter-spacing:0.04em;'>"
        "Player Breakdown</div>",
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

        ch_str = (fmt_m(ch) + ("*" if is_est else "")) if has_data and pd.notna(ch) else "UFA"
        pv_str = fmt_m(pv) if pd.notna(pv) else "—"
        sig_color = KINGS_SIGNAL_PALETTE.get(signal, "#2C3A40")

        delta_str = "—"
        pct_str   = ""
        if pd.notna(delta):
            sign = "+" if delta >= 0 else ""
            clr  = "#1FBFA0" if delta >= 0 else "#E84040"
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
                f"style='border-radius:50%;object-fit:cover;"
                f"border:2px solid {KINGS_GOLD};flex-shrink:0;' "
                f"onerror=\"this.style.display='none'\">"
            )

        est_badge = (
            f"<span style='background:#2C3A40;color:#B8C4C8;padding:1px 5px;"
            f"border-radius:2px;font-size:.82rem;margin-left:4px;' "
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
            ext_note    = (
                f"<div style='margin-top:5px;font-size:11px;color:#6BBAD4;'>"
                f"✅ Extension signed — {ext_len_s} {ext_ch_s}/yr starting {ext_yr_s}</div>"
            )

        st.markdown(
            f"<div class='kings-card' style='display:flex;align-items:center;gap:16px;'>"
            f"  {hs_html}"
            f"  <div style='flex:1;min-width:0;'>"
            f"    <div style='display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;'>"
            f"      <span style='font-size:1.0rem;font-weight:700;color:{KINGS_WHITE};"
            f"font-family:\"Manrope\",sans-serif;'>{name}</span>"
            f"      {est_badge}"
            f"      <span style='color:#A0A0A0;font-size:.87rem;font-family:\"IBM Plex Mono\",monospace;"
            f"letter-spacing:.04em;'>{pos} · {age_str}</span>"
            f"    </div>"
            f"    <div style='display:flex;gap:22px;margin-top:8px;flex-wrap:wrap;'>"
            f"      <div><div class='stat-label'>Cap Hit</div>"
            f"           <div class='stat-value'>{ch_str}</div></div>"
            f"      <div><div class='stat-label'>Pred. Value</div>"
            f"           <div class='stat-value'>{pv_str}</div></div>"
            f"      <div><div class='stat-label'>Value Delta</div>"
            f"           <div style='font-size:.9rem;font-weight:700;font-family:\"IBM Plex Mono\",monospace;'>"
            f"             {delta_str} {pct_str}</div></div>"
            f"      <div><div class='stat-label'>Expiry</div>"
            f"           <div class='stat-value'>{exp_str} ({exp_st})</div></div>"
            f"      <div><div class='stat-label'>Yrs Left</div>"
            f"           <div class='stat-value'>{yrs_str}</div></div>"
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
def _similar_players(player: pd.Series, df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Return n most similar players by position, age, and predicted value."""
    pid = player.get("player_id")
    pos = player.get("pos", "C")
    age = player.get("age", 28) or 28
    pv  = player.get("predicted_value", 0) or 0

    similar = df[
        (df.get("player_id", df.index) != pid) &
        (df["pos"] == pos) &
        (df["age"].between(age - 4, age + 4))
    ].copy()

    if len(similar) < n:
        similar = df[df.get("player_id", df.index) != pid].copy()

    similar["_sim_score"] = (similar["predicted_value"] - pv).abs()
    return similar.nsmallest(n, "_sim_score").drop(columns=["_sim_score"])


def _player_card(player: pd.Series, df: pd.DataFrame, shap_vals: pd.DataFrame,
                 col_prefix: str = ""):
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

    prior_badge = (
        "" if has_prior else
        "<span style='background:#8B1C1C;color:#fff;padding:2px 7px;"
        "border-radius:2px;font-size:.85rem;margin-left:6px;'>NO PRIOR STATS</span>"
    )
    if not has_contract and exp_status.upper() == "UFA":
        contract_badge = (
            "<span style='background:#3A2018;color:#fff;padding:2px 7px;"
            "border-radius:2px;font-size:.85rem;margin-left:6px;'>UFA / UNSIGNED</span>"
        )
    elif is_estimated:
        contract_badge = (
            "<span style='background:#2C3A40;color:#B8C4C8;padding:2px 7px;"
            "border-radius:2px;font-size:.85rem;margin-left:6px;' "
            "title='Salary estimated — contract data pending verification'>SALARY EST.*</span>"
        )
    else:
        contract_badge = ""

    hs_html = ""
    if pid and pd.notna(pid):
        hs_url = headshot_url(pid, team)
        hs_html = (
            f"<img src='{hs_url}' width='80' height='80' "
            f"style='border-radius:50%;object-fit:cover;"
            f"border:2px solid #C8A84B;flex-shrink:0;' "
            f"onerror=\"this.style.display='none'\">"
        )

    st.markdown(
        f"<div class='player-card'>"
        f"  <div style='display:flex;gap:18px;align-items:center;'>"
        f"    {hs_html}"
        f"    <div>"
        f"      <div style='font-size:1.5rem;font-weight:700;color:#E8E4DC;"
        f"font-family:\"Bebas Neue\",cursive;letter-spacing:0.04em;line-height:1.1;'>"
        f"        {name}{prior_badge}{contract_badge}"
        f"      </div>"
        f"      <div style='color:#A0A0A0;margin-top:5px;font-size:.87rem;"
        f"font-family:\"IBM Plex Mono\",monospace;letter-spacing:.08em;text-transform:uppercase;'>"
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
        st.markdown(
            f"<div style='background:#1a1a2e;border:1px solid #252545;border-radius:3px;"
            f"padding:10px 14px;margin:8px 0;font-size:13px;'>"
            f"<span style='color:#6BBAD4;font-weight:700;'>✅ Extension Signed</span>"
            f"<span style='color:#A0A0A0;margin-left:10px;'>"
            f"{ext_yrs_str} · {ext_ch_str}/yr · {ext_start_str} → {ext_exp_str}"
            f"{' · ' + ext_stat if ext_stat else ''}"
            f"</span></div>",
            unsafe_allow_html=True,
        )

    # Re-sign signal
    signal    = player.get("resign_signal", "—")
    sig_color = RESIGN_PALETTE.get(signal, "#555")
    st.markdown(
        f"<div style='margin:10px 0;display:flex;align-items:center;gap:10px;'>"
        f"<span style='color:#A0A0A0;font-size:.78rem;letter-spacing:.14em;"
        f"font-family:\"IBM Plex Mono\",monospace;text-transform:uppercase;'>Re-sign Signal</span>"
        f"<span class='signal-badge' style='background:{sig_color};color:#fff;'>{signal}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # League rank
    rank     = int((df["predicted_value"] >= (pv or 0)).sum())
    total    = len(df)
    rank_pct = pct_rank(df["predicted_value"], pv or 0)
    st.markdown(
        f"<div style='color:#A0A0A0;font-size:.87rem;font-family:\"IBM Plex Mono\",monospace;"
        f"letter-spacing:.04em;'>LEAGUE RANK BY PREDICTED VALUE: "
        f"<span style='color:#C8A84B;font-weight:700;'>#{rank} of {total}</span> "
        f"&nbsp;·&nbsp; TOP <span style='color:#C8A84B;font-weight:700;'>{100-rank_pct}%</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Percentile bars
    st.markdown("**Stat Percentiles (league-wide)**")
    pct_data = []
    for col, label in PERCENTILE_STATS:
        if col in df.columns and col in player.index and pd.notna(player.get(col)):
            val = player[col]
            pct = pct_rank(df[col], val)
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

    # Value driver breakdown
    if (not shap_vals.empty and "name" in shap_vals.columns
            and name in shap_vals["name"].values):
        row_shap     = shap_vals[shap_vals["name"] == name].iloc[0].drop("name")
        vals_dollars = row_shap.astype(float) * CAP_CEILING
        base         = float(df["predicted_value"].mean())

        pos_factors = vals_dollars[vals_dollars > 0].nlargest(5)
        neg_factors = vals_dollars[vals_dollars < 0].nsmallest(5)
        max_abs     = max(
            pos_factors.abs().max() if not pos_factors.empty else 0,
            neg_factors.abs().max() if not neg_factors.empty else 0,
        )

        st.markdown(
            f"<div style='margin:18px 0 4px 0;font-family:\"Bebas Neue\",cursive;"
            f"font-size:1.1rem;font-weight:700;color:{_T['page_text']};letter-spacing:0.04em;'>"
            f"What drives {name}'s value?</div>"
            f"<div style='color:#A0A0A0;font-size:.87rem;margin-bottom:14px;"
            f"font-family:\"IBM Plex Mono\",monospace;letter-spacing:.04em;'>"
            f"STARTING FROM LEAGUE AVERAGE — FACTORS PUSHING VALUE UP OR DOWN</div>",
            unsafe_allow_html=True,
        )

        # Section 1 — league average
        st.markdown(
            f"<div style='background:#1a1a2e;border-radius:2px;border:1px solid #252545;"
            f"padding:10px 16px;margin-bottom:14px;display:flex;align-items:center;"
            f"justify-content:space-between;'>"
            f"<span style='font-size:.82rem;color:#A0A0A0;font-family:\"IBM Plex Mono\",monospace;"
            f"letter-spacing:.12em;text-transform:uppercase;'>League Average</span>"
            f"<strong style='color:#C8A84B;font-family:\"IBM Plex Mono\",monospace;"
            f"font-size:.95rem;'>${base/1e6:.2f}M</strong></div>",
            unsafe_allow_html=True,
        )

        # Section 2 — side-by-side factors with CSS hover tooltips
        st.markdown("""
<style>
.vd-row{position:relative;margin:5px 0;cursor:default;}
.vd-tip{
  visibility:hidden;opacity:0;
  background:#1a1a2e;color:#E8E4DC;
  border:1px solid #707070;border-radius:2px;
  padding:8px 12px;font-size:11px;line-height:1.6;
  position:absolute;z-index:9999;
  bottom:115%;left:0;
  min-width:240px;max-width:340px;
  white-space:normal;pointer-events:none;
  transition:opacity .12s ease;
  font-family:'Manrope',sans-serif;
}
.vd-row:hover .vd-tip{visibility:visible;opacity:1;}
</style>
""", unsafe_allow_html=True)

        def _factor_row(lbl: str, val: float, max_v: float, positive: bool, tip: str) -> str:
            bar_pct  = int(abs(val) / max_v * 100) if max_v > 0 else 0
            bar_pct  = max(bar_pct, 3)
            color    = "#1FBFA0" if positive else "#E84040"
            sign_str = f"+${val/1e6:.2f}M" if positive else f"-${abs(val)/1e6:.2f}M"
            safe_tip = tip.replace('"', "&quot;").replace("<br>", " ")
            return (
                f"<div class='vd-row'>"
                f"<div class='vd-tip'><strong>{lbl}</strong><br>{safe_tip}</div>"
                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:center;margin-bottom:3px;'>"
                f"<span style='font-size:12px;color:#E8E4DC;'>{lbl}</span>"
                f"<span style='font-size:12px;color:{color};font-weight:600;'>{sign_str}</span>"
                f"</div>"
                f"<div style='height:12px;width:{bar_pct}%;background:{color};"
                f"border-radius:2px;opacity:0.85;'></div>"
                f"</div>"
            )

        col_up, col_dn = st.columns(2)

        with col_up:
            st.markdown(
                "<div style='color:#1FBFA0;font-weight:700;font-size:.85rem;"
                "letter-spacing:.12em;text-transform:uppercase;"
                "font-family:\"IBM Plex Mono\",monospace;margin-bottom:8px;"
                "border-left:2px solid #1FBFA0;padding-left:8px;'>"
                "Pushing value UP</div>",
                unsafe_allow_html=True,
            )
            if pos_factors.empty:
                st.markdown("<span style='color:#A0A0A0;font-size:12px;'>None</span>",
                            unsafe_allow_html=True)
            else:
                rows = "".join(
                    _factor_row(_label(f), v, max_abs, True,
                                _driver_tooltip(f, player, name, True))
                    for f, v in pos_factors.items()
                )
                st.markdown(
                    f"<div style='background:#1a1a2e;border-radius:2px;border:1px solid #252545;"
                    f"padding:10px 12px;'>{rows}</div>",
                    unsafe_allow_html=True,
                )

        with col_dn:
            st.markdown(
                "<div style='color:#E84040;font-weight:700;font-size:.85rem;"
                "letter-spacing:.12em;text-transform:uppercase;"
                "font-family:\"IBM Plex Mono\",monospace;margin-bottom:8px;"
                "border-left:2px solid #E84040;padding-left:8px;'>"
                "Pushing value DOWN</div>",
                unsafe_allow_html=True,
            )
            if neg_factors.empty:
                st.markdown("<span style='color:#A0A0A0;font-size:12px;'>None</span>",
                            unsafe_allow_html=True)
            else:
                rows = "".join(
                    _factor_row(_label(f), v, max_abs, False,
                                _driver_tooltip(f, player, name, False))
                    for f, v in neg_factors.items()
                )
                st.markdown(
                    f"<div style='background:#1a1a2e;border-radius:2px;border:1px solid #252545;"
                    f"padding:10px 12px;'>{rows}</div>",
                    unsafe_allow_html=True,
                )

        # Section 3 — result
        pv_val   = pv or base
        pv_color = "#1FBFA0" if (delta or 0) >= 0 else "#E84040"
        cap_str  = f"${ch/1e6:.2f}M" if ch else "—"
        if delta is not None:
            delta_str = f"+${delta/1e6:.2f}M" if delta >= 0 else f"-${abs(delta)/1e6:.2f}M"
        else:
            delta_str = "—"
        st.markdown(
            f"<div style='background:#1a1a2e;border:1px solid #252545;border-radius:2px;"
            f"border-left:3px solid {pv_color};padding:16px 20px;margin-top:16px;'>"
            f"<div style='font-size:.78rem;color:#A0A0A0;font-family:\"IBM Plex Mono\",monospace;"
            f"letter-spacing:.14em;text-transform:uppercase;margin-bottom:6px;'>"
            f"Estimated Market Value</div>"
            f"<div style='font-size:1.5rem;font-weight:700;color:{pv_color};"
            f"font-family:\"IBM Plex Mono\",monospace;margin-bottom:8px;'>"
            f"${pv_val/1e6:.2f}M</div>"
            f"<div style='font-size:.87rem;color:#A0A0A0;font-family:\"IBM Plex Mono\",monospace;'>"
            f"Current Cap Hit: <span style='color:#5A5A5A;'>{cap_str}</span>"
            f"&nbsp;&nbsp;·&nbsp;&nbsp;"
            f"Difference: <span style='color:{pv_color};font-weight:700;'>{delta_str}</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    # Similar players
    st.markdown("---")
    st.markdown("**Similar Players**")
    sim = _similar_players(player, df, n=3)
    if not sim.empty:
        sim_cols = st.columns(len(sim))
        for col_i, (_, sp) in enumerate(sim.iterrows()):
            sp_pid  = sp.get("player_id")
            sp_name = sp.get("name", "?")
            sp_team = sp.get("team", "?")
            sp_pos  = sp.get("pos", "?")
            sp_age  = sp.get("age")
            sp_ch   = sp.get("cap_hit")
            sp_pv   = sp.get("predicted_value")
            sp_dlt  = sp.get("value_delta")

            sp_hs = ""
            if sp_pid and pd.notna(sp_pid):
                sp_hs_url = headshot_url(sp_pid, sp_team)
                sp_hs = (
                    f"<img src='{sp_hs_url}' width='56' height='56' "
                    f"style='border-radius:50%;object-fit:cover;"
                    f"border:1px solid #252545;' "
                    f"onerror=\"this.style.display='none'\">"
                )
            clr = "#1FBFA0" if (sp_dlt or 0) >= 0 else "#E84040"
            sim_cols[col_i].markdown(
                f"<div style='background:#1a1a2e;border-radius:2px;padding:14px 10px;"
                f"border:1px solid #252545;border-bottom:2px solid {clr};text-align:center;'>"
                f"  {sp_hs}"
                f"  <div style='font-weight:700;color:#E8E4DC;margin-top:7px;"
                f"font-size:.85rem;font-family:\"Manrope\",sans-serif;'>{sp_name}</div>"
                f"  <div style='color:#A0A0A0;font-size:.78rem;margin:3px 0;"
                f"font-family:\"IBM Plex Mono\",monospace;letter-spacing:.06em;'>"
                f"    {sp_team} · {sp_pos}"
                f"    {f'· {sp_age:.0f}' if pd.notna(sp_age) else ''}"
                f"  </div>"
                f"  <div style='margin-top:7px;font-size:.78rem;"
                f"font-family:\"IBM Plex Mono\",monospace;'>"
                f"    <span style='color:#A0A0A0;'>PRED </span>"
                f"    <span style='color:#E8E4DC;'>{fmt_m(sp_pv)}</span>"
                f"  </div>"
                f"  <div style='color:{clr};font-size:.88rem;font-weight:700;"
                f"font-family:\"IBM Plex Mono\",monospace;margin-top:2px;'>"
                f"    {fmt_delta(sp_dlt)}"
                f"  </div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No similar players found.")

    # How is this value calculated?
    with st.expander("How is this value calculated?"):
        st.markdown(f"""
**Model:** XGBoost trained on {int(df['has_contract_data'].fillna(False).sum())} NHL players
with known contracts (CV R² = 0.829, RMSE ≈ $1.21M).

**What goes in:**
- *Current season stats* — points, TOI/game, power-play points, shooting %, G/60, P/60
  (projected to 82-game pace from actual games played)
- *Prior season stats* — same metrics from the previous season, for players with prior data
- *Contract structure* — years remaining, contract length
- *Bio* — age, draft position, draft round
- *Position* — affects TOI benchmarks (D vs F)

**What comes out:**
The model predicts a player's **market-rate cap hit** — what they would command
on the open free-agent market — expressed as a fraction of the current cap ceiling,
then scaled back to dollars.

**Value Delta = Predicted Value − Actual Cap Hit**
- **Positive** → player is worth more than paid → team surplus
- **Negative** → player costs more than model-estimated market rate → team liability

**Important caveats:**
- Defensive metrics (shot blocking, defensive zone starts) are partially captured
  through TOI but not explicitly
- Leadership, locker-room value, and injury history are not modelled
- ELC players (age < 25, cap hit < $1M) will almost always show a large positive delta
  because they are intentionally paid below market
- Small sample sizes (e.g., rookies with < 20 GP) increase prediction uncertainty

The model is retrained nightly from live NHL API stats.
        """)


def tab_player_search(df: pd.DataFrame, shap_vals: pd.DataFrame):
    st.markdown(f"<div style='font-family:\"Bebas Neue\",cursive;font-size:2rem;color:{_T['page_text']};letter-spacing:0.04em;margin:0 0 16px;font-weight:400;'>Player Search</div>", unsafe_allow_html=True)

    all_names = sorted(df["name"].dropna().tolist())
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
    player1   = df[df["name"] == selected1].iloc[0]

    compare = st.checkbox("Compare with another player")
    if compare:
        search2  = st.text_input("Second player", placeholder="e.g. mcdavid, draisaitl…",
                                  key="search2")
        matches2 = [n for n in all_names if search2.lower() in n.lower()] if search2 else all_names
        sel2     = st.selectbox("Select", matches2, key="sel2")
        player2  = df[df["name"] == sel2].iloc[0]

        col_a, col_b = st.columns(2)
        with col_a:
            _player_card(player1, df, shap_vals, "p1")
        with col_b:
            _player_card(player2, df, shap_vals, "p2")
    else:
        _player_card(player1, df, shap_vals)


# ── Tab 5: Model Insights ──────────────────────────────────────────────────────
def tab_insights(df: pd.DataFrame):
    st.markdown(f"<div style='font-family:\"Bebas Neue\",cursive;font-size:2rem;color:{_T['page_text']};letter-spacing:0.04em;margin:0 0 16px;font-weight:400;'>Model Insights — SHAP Feature Importance</div>", unsafe_allow_html=True)
    shap_summary = load_shap_summary()
    shap_vals    = load_shap_values()

    if shap_summary.empty:
        st.info("Run `py -3 pipeline.py` to generate SHAP values.")
        return

    st.markdown(f"<div style='font-family:\"Bebas Neue\",cursive;font-size:1.5rem;color:{_T['page_text']};letter-spacing:0.04em;margin:0 0 12px;font-weight:400;'>What drives predicted player value?</div>", unsafe_allow_html=True)
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
        font=dict(family="'IBM Plex Mono', monospace", color=_T["plot_font"]), showlegend=False, coloraxis_showscale=False,
        xaxis=dict(
            tickvals=tick_vals, ticktext=tick_text,
            title="Avg. Dollar Impact on Prediction", gridcolor=_T["grid"],
        ),
        margin=dict(l=10, r=20, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Mean absolute SHAP value = average dollar impact of each feature on predictions.")

    if not shap_vals.empty and "name" in shap_vals.columns:
        st.markdown("---")
        st.markdown(f"<div style='font-family:\"Bebas Neue\",cursive;font-size:1.5rem;color:{_T['page_text']};letter-spacing:0.04em;margin:0 0 12px;font-weight:400;'>Player-Level Explanation</div>", unsafe_allow_html=True)
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
                font=dict(family="'IBM Plex Mono', monospace", color=_T["plot_font"]), showlegend=False, coloraxis_showscale=False,
                xaxis=dict(
                    tickvals=tick_vals2, ticktext=tick_text2,
                    title="Dollar Impact on Prediction", gridcolor=_T["grid"],
                    zeroline=True, zerolinecolor=_T["zero"], zerolinewidth=2,
                ),
                title=dict(text=f"SHAP Breakdown — {chosen}",
                           font=dict(family="'IBM Plex Mono', monospace", color=_T["plot_font"])),
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

    st.markdown(
        "<div style='height:1px;background:linear-gradient(90deg,#C8A84B 0%,#C8A84B 25%,"
        "#707070 60%,#141414 100%);margin:20px 0 12px;'></div>",
        unsafe_allow_html=True,
    )
    _last_updated = (f" &nbsp;·&nbsp; {last_ts}" if last_ts else "")
    _season_info = f" &nbsp;·&nbsp; {_season_str(load_season_context())} &nbsp;·&nbsp; Cap: ${CAP_CEILING/1e6:.0f}M"
    st.markdown(
        f"<div style='color:#707070;font-size:.78rem;text-align:center;padding:4px 0 8px;"
        f"font-family:\"IBM Plex Mono\",monospace;letter-spacing:.08em;text-transform:uppercase;'>"
        f"NHL API &nbsp;·&nbsp; PuckPedia &nbsp;·&nbsp; "
        f"{n_real} contracts · {n_est} est"
        f"{_last_updated}{_season_info}"
        f"</div>",
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

    # Kick off background data refresh on every cold start (non-blocking)
    start_background_refresh(PROCESSED_DIR)

    # If background job just finished, trigger a rerun so UI picks up fresh mtime
    if _refresh_status["done"]:
        _refresh_status["done"] = False
        st.rerun()

    st.markdown(
        f"<div style='padding:20px 0 0;'>"
        f"<div style='font-family:\"IBM Plex Mono\",monospace;font-size:0.5rem;"
        f"color:#707070;letter-spacing:0.35em;text-transform:uppercase;margin-bottom:14px;'>"
        f"{_season_str(load_season_context())} &nbsp;·&nbsp; XGBOOST + SHAP &nbsp;·&nbsp; LIVE DATA"
        f"</div>"
        f"<div style='font-family:\"Bebas Neue\",cursive;font-size:3.8rem;color:#E8E4DC;"
        f"line-height:0.88;letter-spacing:0.04em;'>"
        f"NHL Player Value<br>"
        f"<span style='color:#C8A84B;'>Model</span>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='height:1px;background:#1C1C1C;margin:18px 0 24px;'></div>",
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
            icon="🔄",
        )
    elif _refresh_status["error"] and _timed_out:
        st.warning("Data fetch timed out. Showing last cached data.", icon="⚠️")

    df        = load_predictions()
    shap_vals = load_shap_values()
    filtered  = sidebar_filters(df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌐 League Overview",
        "📊 Leaderboards",
        "👑 LA Kings",
        "🔍 Player Search",
        "🧠 Model Insights",
    ])

    with tab1:
        tab_overview(filtered, df)
    with tab2:
        tab_leaderboards(filtered)
    with tab3:
        tab_kings(df)
    with tab4:
        tab_player_search(df, shap_vals)
    with tab5:
        tab_insights(df)

    render_footer(df)


if __name__ == "__main__":
    main()
