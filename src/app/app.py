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
_refresh_status: dict = {"running": False, "done": False, "error": None, "started_at": None}
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
        _refresh_status["error"] = None
    except Exception as e:
        _refresh_status["error"] = str(e)
    finally:
        _refresh_status["running"] = False


def start_background_refresh(processed_dir: Path) -> None:
    """Launch background pipeline thread if not already running."""
    with _refresh_lock:
        if _refresh_status["running"]:
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

    toi     = _get("toi_per_g")
    goals   = _get("g", ".0f")
    assists = _get("a", ".0f")
    points  = _get("p", ".0f")
    ppg_v   = _get("ppg")
    age_v   = _get("age", ".0f")
    pm      = _get("plus_minus", "+.0f")
    spct    = _get("shooting_pct")
    g60_v   = _get("g60")
    p60_v   = _get("p60")
    shots_v = _get("shots", ".0f")
    pp_pts_v= _get("pp_pts", ".0f")
    hits_v  = _get("hits", ".0f")
    blk_v   = _get("blocks", ".0f")
    xg_v    = _get("xg")
    fen_v   = _get("fenwick_pct")
    oz_v    = _get("oz_start_pct")
    pp_toi_v= _get("pp_toi")
    pk_toi_v= _get("pk_toi")
    fo_v    = _get("faceoff_pct")
    loc_raw = player.get("length_of_contract")
    loc_v   = int(float(loc_raw)) if loc_raw and not (isinstance(loc_raw, float) and pd.isna(loc_raw)) else None
    dp_v    = _get("draft_position", ".0f")

    is_prior = feat.endswith("_24")
    base_feat = feat[:-3] if is_prior else feat
    prior_pfx = "Last season: " if is_prior else ""

    tips: dict[str, str] = {
        "toi_per_g": (
            f"Averaging {toi} min/game puts {name} among the league's top deployment players, signaling high trust from coaching staff."
            if positive else
            f"Averaging only {toi} min/game suggests a limited role, which the market prices lower."
        ) if toi else (
            "High ice time signals strong trust from coaching staff."
            if positive else
            "Limited ice time suggests a depth or specialized role."
        ),
        "g": (
            f"{name} has scored {goals} goals this season, outpacing most players at their position."
            if positive else
            f"{goals} goals this season is below average for a player with this much ice time."
        ) if goals else (
            "Strong goal-scoring output boosts predicted market value."
            if positive else
            "Below-average goal production pulls predicted value down."
        ),
        "a": (
            f"{name} has {assists} assists this season, showing strong playmaking ability."
            if positive else
            "Fewer assists than expected limits the offensive upside the market rewards."
        ) if assists else (
            "Strong playmaking boosts predicted market value."
            if positive else
            "Below-average assists reduce predicted value."
        ),
        "p": (
            f"{name} has {points} points this season, placing them among the top offensive contributors."
            if positive else
            "Point totals below league norms at this salary tier reduce predicted value."
        ) if points else (
            "Strong point totals boost predicted market value."
            if positive else
            "Below-average point totals reduce predicted value."
        ),
        "ppg": (
            f"{name} is producing at {ppg_v} points per game, an elite offensive pace."
            if positive else
            f"Producing at {ppg_v} points per game is below average for a player at this salary level."
        ) if ppg_v else (
            "Elite points-per-game rate signals consistent offensive production."
            if positive else
            "Below-average points-per-game rate reduces market value."
        ),
        "age": (
            f"At {age_v} years old, {name} is in their prime earning years."
            if positive else
            f"At {age_v} years old, {name} is on the back half of a typical NHL career, which the market discounts."
        ) if age_v else (
            "Prime age is a positive factor in market pricing."
            if positive else
            "Age is a factor the market discounts for older players."
        ),
        "length_of_contract": (
            f"A {loc_v}-year contract signals the team committed to this player long term, which correlates with higher market value."
            if positive else
            "A shorter remaining contract reduces leverage in negotiations, pulling predicted value down."
        ) if loc_v else (
            "Long-term contract commitment correlates with higher market value."
            if positive else
            "Shorter contract length reduces market leverage."
        ),
        "draft_position": (
            f"Being selected {dp_v}th overall reflects high organizational investment and long-term expectations."
            if positive else
            "A later draft position means lower initial expectations, which still factors into market pricing."
        ) if dp_v else (
            "High draft pedigree is a positive factor in market pricing."
            if positive else
            "Draft position factors into long-term market expectations."
        ),
        "draft_year": (
            "Being drafted recently means the player is still in their developmental upside window."
            if positive else
            "Draft year context suggests the player may be past their expected development peak."
        ),
        "year_of_contract": (
            "Early contract years correlate with upside potential that the market rewards."
            if positive else
            "Later contract years reduce remaining upside, pulling predicted value down."
        ),
        "pp_toi": (
            f"{name} averages {pp_toi_v} min of power play time per game, indicating they are a key offensive weapon."
            if positive else
            "Limited power play time suggests a more defensive or depth role."
        ) if pp_toi_v else (
            "Strong power play deployment signals offensive importance to the team."
            if positive else
            "Limited power play time suggests a more defensive or depth role."
        ),
        "pk_toi": (
            f"{name} averages {pk_toi_v} min of PK time per game, showing the team trusts them in high-pressure defensive situations."
            if positive else
            "Minimal penalty kill usage suggests a limited two-way role."
        ) if pk_toi_v else (
            "Strong PK deployment reflects two-way trust from coaching staff."
            if positive else
            "Minimal penalty kill usage suggests a limited two-way role."
        ),
        "fenwick_pct": (
            f"When {name} is on the ice, the team controls {fen_v}% of shot attempts, meaning they drive play in the right direction."
            if positive else
            f"The team is outshot when {name} is on the ice, which advanced metrics penalize."
        ) if fen_v else (
            "Strong shot-attempt control when on the ice is a positive advanced metric."
            if positive else
            "Being outshot when on the ice is penalized by advanced metrics."
        ),
        "xg": (
            f"{name} has generated {xg_v} expected goals this season, meaning their shot quality and volume is elite."
            if positive else
            f"Lower expected goals suggests {name} is not generating high-quality scoring chances."
        ) if xg_v else (
            "Elite expected goals generation reflects strong shot quality and volume."
            if positive else
            "Below-average expected goals suggests limited high-quality scoring chances."
        ),
        "oz_start_pct": (
            f"Starting {oz_v}% of shifts in the offensive zone shows the team uses {name} as an offensive weapon."
            if positive else
            f"Starting most shifts in the defensive zone means {name} faces tougher assignments, which can suppress offensive numbers."
        ) if oz_v else (
            "High offensive zone start % reflects offensive deployment trust."
            if positive else
            "Defensive zone deployment can suppress offensive numbers."
        ),
        "plus_minus": (
            f"A plus/minus of {pm} means {name} is on the ice for more goals for than against."
            if positive else
            f"A plus/minus of {pm} means {name} has been on the ice for more goals against than for."
        ) if pm else (
            "Positive plus/minus reflects strong on-ice goal differential."
            if positive else
            "Negative plus/minus indicates more goals against than for when on the ice."
        ),
        "shooting_pct": (
            f"{name} converts {spct}% of their shots, above the league average, indicating elite finishing ability."
            if positive else
            f"A shooting percentage of {spct}% is below average, suggesting some regression or inefficiency."
        ) if spct else (
            "Above-average shooting % indicates elite finishing ability."
            if positive else
            "Below-average shooting % suggests regression or inefficiency."
        ),
        "hits": (
            f"{name} has recorded {hits_v} hits this season, showing physical presence that teams value."
            if positive else
            "Below-average physical play in terms of hits."
        ) if hits_v else (
            "Strong physical presence is valued by teams."
            if positive else
            "Below-average physical play in terms of hits."
        ),
        "blocks": (
            f"{name} has blocked {blk_v} shots this season, demonstrating defensive commitment."
            if positive else
            "Few blocked shots suggests limited defensive zone presence."
        ) if blk_v else (
            "Strong shot-blocking demonstrates defensive commitment."
            if positive else
            "Few blocked shots suggests limited defensive zone presence."
        ),
        "g60": (
            f"{name} scores at {g60_v} goals per 60 min, showing elite scoring efficiency per unit of ice time."
            if positive else
            f"A goals-per-60 rate of {g60_v} is below what the market expects at this salary level."
        ) if g60_v else (
            "High goals-per-60 rate signals efficient scoring."
            if positive else
            "Below-average goals-per-60 rate reduces market value."
        ),
        "p60": (
            f"{name} produces at {p60_v} points per 60 min, reflecting strong offensive efficiency."
            if positive else
            f"A points-per-60 rate of {p60_v} suggests limited offensive impact relative to ice time."
        ) if p60_v else (
            "Strong points-per-60 reflects elite offensive efficiency."
            if positive else
            "Below-average points-per-60 suggests limited offensive impact."
        ),
        "shots": (
            f"{name} has put {shots_v} shots on net this season, generating consistent offensive pressure."
            if positive else
            "Below-average shot volume limits offensive impact."
        ) if shots_v else (
            "High shot volume generates consistent offensive pressure."
            if positive else
            "Below-average shot volume limits offensive impact."
        ),
        "pp_pts": (
            f"{name} has {pp_pts_v} power play points this season, showing they are a key part of the power play."
            if positive else
            "Limited power play production suggests a smaller role on the man advantage."
        ) if pp_pts_v else (
            "Strong power play point production signals offensive importance."
            if positive else
            "Limited power play production suggests a smaller role on the man advantage."
        ),
        "faceoff_pct": (
            f"Winning {fo_v}% of faceoffs gives {name}'s team better possession starts across the game."
            if positive else
            "A faceoff win rate below 50% means opponents gain more possession starts."
        ) if fo_v else (
            "Strong faceoff win rate gives the team better possession starts."
            if positive else
            "Below-average faceoff win rate cedes possession to opponents."
        ),
        "pim": (
            "Penalty minutes can reflect physical, aggressive play that teams value in certain roles."
            if positive else
            "High penalty minutes hurt on-ice impact by giving opponents power play opportunities."
        ),
    }

    sentence = tips.get(base_feat)
    if sentence is None:
        lbl = _label(feat)
        sentence = (
            f"{lbl} is a positive factor in {name}'s market value."
            if positive else
            f"{lbl} is pulling {name}'s estimated value down."
        )

    full = f"{prior_pfx}{sentence}" if is_prior else sentence
    # Wrap at ~65 chars for tooltip readability
    wrapped = "<br>".join(textwrap.wrap(full, width=65))
    return wrapped


def _driver_chart(
    factors: pd.Series,
    player: pd.Series,
    name: str,
    positive: bool,
    max_abs: float,
) -> "go.Figure | None":
    """Build a Plotly horizontal bar chart for value drivers with hover tooltips."""
    if factors.empty:
        return None

    color   = "#43A047" if positive else "#E53935"
    labels  = [_label(f) for f in factors.index]
    values  = factors.abs().tolist()
    sign    = "+" if positive else "-"
    texts   = [f"{sign}${v/1e6:.2f}M" for v in values]
    tips    = [_driver_tooltip(f, player, name, positive) for f in factors.index]

    fig = go.Figure(go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker=dict(color=color, opacity=0.85, line=dict(color=color, width=0)),
        customdata=[[t] for t in tips],
        hovertemplate="<b>%{y}</b><br>%{customdata[0]}<extra></extra>",
        text=texts,
        textposition="outside",
        textfont=dict(color=color, size=11),
    ))
    fig.update_layout(
        paper_bgcolor="#0D0D1A",
        plot_bgcolor="#0D0D1A",
        font=dict(color="#CCCCDD", size=12),
        height=max(200, len(factors) * 54 + 40),
        xaxis=dict(visible=False, range=[0, max_abs * 1.45]),
        yaxis=dict(title="", autorange="reversed", tickfont=dict(size=12)),
        margin=dict(l=0, r=85, t=5, b=5),
        hoverlabel=dict(
            bgcolor="#1C1C30",
            bordercolor="#4A4A6F",
            font_size=12,
            font_color="#FFFFFF",
            namelength=0,
        ),
        showlegend=False,
    )
    return fig


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
KINGS_BLACK  = "#010101"
KINGS_GOLD   = "#B5975A"
KINGS_SILVER = "#A2AAAD"
KINGS_WHITE  = "#F5F5F5"

# ── Resign signal palettes ─────────────────────────────────────────────────────
RESIGN_PALETTE = {
    "Must Sign":         "#1B5E20",
    "Priority RFA":      "#1565C0",
    "Locked In (Value)": "#4A148C",
    "Fair Deal":         "#37474F",
    "Let Walk":          "#BF360C",
    "Buyout Candidate":  "#880E4F",
    "Monitor":           "#546E7A",
}

KINGS_SIGNAL_PALETTE = {
    "Extension Now":    "#006400",
    "Lock Up":          "#1B5E20",
    "Priority Re-sign": "#0D47A1",
    "Fair Deal":        "#37474F",
    "Monitor":          "#546E7A",
    "Let Walk":         "#BF360C",
    "Buyout Candidate": "#880E4F",
    "UFA":              "#5D4037",
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
st.markdown("""
<style>
  .block-container { padding-top: 3.5rem; padding-bottom: 2rem; }

  /* Metric cards */
  [data-testid="stMetric"] {
      background:#13131F; border-radius:10px;
      padding:14px 18px; border:1px solid #2A2A3E;
  }
  [data-testid="stMetricLabel"] { color:#8888AA; font-size:0.78rem; letter-spacing:.04em; }
  [data-testid="stMetricValue"] { color:#F0F0FF; font-size:1.35rem; font-weight:700; }
  [data-testid="stMetricDelta"] { font-size:0.8rem; }

  /* Player card */
  .player-card {
      background:#13131F; border-radius:12px;
      padding:20px 24px; border:1px solid #2A2A3E;
      margin-bottom:12px;
  }
  .kings-card {
      background:#0D0D14; border-radius:12px;
      padding:20px 24px;
      border-top:3px solid #B5975A; border-left:1px solid #2A2A3E;
      border-right:1px solid #2A2A3E; border-bottom:1px solid #2A2A3E;
      margin-bottom:12px;
  }

  /* Typography helpers */
  .stat-label { color:#777799; font-size:0.72rem; text-transform:uppercase; letter-spacing:.06em; }
  .stat-value { color:#E8E8F8; font-size:1.05rem; font-weight:600; }
  .delta-pos  { color:#4CAF50; font-size:1.5rem; font-weight:800; }
  .delta-neg  { color:#F44336; font-size:1.5rem; font-weight:800; }
  .pct-pos    { color:#81C784; font-size:0.88rem; }
  .pct-neg    { color:#E57373; font-size:0.88rem; }
  .kings-gold { color:#B5975A; font-weight:700; }
  .section-header { color:#B5975A; font-size:1.35rem; font-weight:700; margin-bottom:6px; }

  /* Signal badge */
  .signal-badge {
      display:inline-block; padding:3px 10px;
      border-radius:5px; font-size:.75rem;
      font-weight:600; letter-spacing:.03em;
  }

  /* Sidebar */
  [data-testid="stSidebar"] { background:#0A0A14; }

  /* Hide Streamlit default footer */
  footer { visibility:hidden; }

  /* Tabs */
  [data-baseweb="tab-list"] { gap:6px; }
  [data-baseweb="tab"] { border-radius:6px 6px 0 0; }
</style>
""", unsafe_allow_html=True)

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

        clr     = "#4CAF50" if (delta or 0) >= 0 else "#F44336"
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
            f"<div style='background:#13131F;border-radius:8px;padding:10px 6px;"
            f"text-align:center;border:1px solid #2A2A3E;'>"
            f"  {hs_html}"
            f"  <div style='font-weight:700;color:#EEE;font-size:.8rem;"
            f"    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
            f"    max-width:100%;'>{name}</div>"
            f"  <div style='color:#888;font-size:.7rem;margin:2px 0;'>"
            f"    {logo_html}{team} · {pos}"
            f"    {f'<br>{age_str}' if age_str else ''}</div>"
            f"  <div style='color:{clr};font-size:.82rem;font-weight:700;"
            f"    margin-top:3px;'>{val_str}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def signal_badge(signal: str, palette: dict) -> str:
    color = palette.get(signal, "#37474F")
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
        st.markdown(
            f"<div style='color:{KINGS_GOLD};font-size:1.1rem;font-weight:700;"
            "margin-bottom:4px;'>🏒 NHL Value Model</div>",
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
        age_r = st.slider("Age range", age_min, age_max, (age_min, age_max))

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
    st.markdown("## League-Wide Value vs. Cap Hit")

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

    # Trace 2 — UFA/unsigned players (diamond markers on x-axis)
    if not df_ufa.empty:
        fig.add_trace(go.Scatter(
            x=df_ufa["predicted_value"], y=[0] * len(df_ufa),
            mode="markers",
            name="UFA / Unsigned",
            marker=dict(
                symbol="diamond", size=9, opacity=0.85,
                color="#B5975A",
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
                      line=dict(dash="dash", color="#666", width=1.5))
        fig.add_annotation(x=hi * 0.72, y=hi * 0.78, text="Fair value line",
                           showarrow=False, font=dict(color="#666", size=11))

    fig.update_layout(
        paper_bgcolor="#0A0A14", plot_bgcolor="#111120",
        font=dict(family="Inter, sans-serif", color="#CCCCDD"),
        xaxis=dict(tickformat="$,.0f", title="Predicted Market Value",
                   gridcolor="#1E1E30", zeroline=False),
        yaxis=dict(tickformat="$,.0f", title="Actual Cap Hit (0 = UFA/Unsigned)",
                   gridcolor="#1E1E30"),
        legend=dict(bgcolor="#13131F", bordercolor="#2A2A3E",
                    orientation="h", y=1.04, x=0.5, xanchor="center"),
        height=600,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    event = st.plotly_chart(fig, use_container_width=True,
                            on_select="rerun", key="overview_scatter")

    # Show clicked player mini-card
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

    st.caption(
        "🟢 Below the dashed line = underpaid &nbsp;·&nbsp; "
        "🔴 Above = overpaid &nbsp;·&nbsp; "
        "🔶 Diamonds on x-axis = UFA / no current contract &nbsp;·&nbsp; "
        "Click any marker for a quick summary."
    )

    # ── Most Interesting Players ──────────────────────────────────────────────
    if not df_c.empty:
        st.markdown("---")
        mi1, mi2 = st.columns(2)
        with mi1:
            st.markdown(
                "<div style='color:#4CAF50;font-size:1rem;font-weight:700;"
                "margin-bottom:10px;'>🟢 Top 5 Underpaid</div>",
                unsafe_allow_html=True,
            )
            _mini_player_cards(df_c.nlargest(5, "value_delta"))
        with mi2:
            st.markdown(
                "<div style='color:#F44336;font-size:1rem;font-weight:700;"
                "margin-bottom:10px;'>🔴 Top 5 Overpaid</div>",
                unsafe_allow_html=True,
            )
            _mini_player_cards(df_c.nsmallest(5, "value_delta"))


# ── Tab 2: Leaderboards ────────────────────────────────────────────────────────
def tab_leaderboards(df: pd.DataFrame):
    st.markdown("## Value Leaderboards")

    df_c   = df[df["cap_hit"].notna() & df["value_delta"].notna()].copy()
    df_c   = add_delta_pct(df_c)
    df_ufa = df[df["cap_hit"].isna() & df["predicted_value"].notna()].copy()

    lc1, lc2, lc3 = st.columns([1, 1, 2])
    n = lc1.slider("Show top N", 5, 30, 15, key="lb_n")
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
            paper_bgcolor="#0A0A14", plot_bgcolor="#111120",
            font=dict(color="#CCCCDD"), showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
            xaxis=dict(
                tickformat=".1f%" if sort_by == "% Delta" else "$,.0f",
                title=x_lbl, gridcolor="#1E1E30",
            ),
            margin=dict(l=0, r=10, t=30, b=10),
            title=dict(text=title, font=dict(color="#CCCCEE", size=14)),
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
            "<div style='color:#4CAF50;font-size:.85rem;font-weight:600;"
            "margin-bottom:6px;'>Top 5 Underpaid</div>",
            unsafe_allow_html=True,
        )
        _mini_player_cards(df_c.nlargest(5, sort_col))
    with hs2:
        st.markdown(
            "<div style='color:#F44336;font-size:.85rem;font-weight:600;"
            "margin-bottom:6px;'>Top 5 Overpaid</div>",
            unsafe_allow_html=True,
        )
        _mini_player_cards(df_c.nsmallest(5, sort_col))

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    t1, t2 = st.columns(2)

    def fmt_table(data):
        cols = ["name", "team", "pos", "age", "cap_hit",
                "predicted_value", "value_delta", "value_delta_pct"]
        d = data[[c for c in cols if c in data.columns]].copy()
        d["age"]             = d["age"].apply(lambda v: f"{v:.0f}" if pd.notna(v) else "?")
        d["cap_hit"]         = d["cap_hit"].apply(fmt_m)
        d["predicted_value"] = d["predicted_value"].apply(fmt_m)
        d["value_delta"]     = d["value_delta"].apply(fmt_delta)
        d["value_delta_pct"] = d["value_delta_pct"].apply(fmt_pct)
        return d.rename(columns={
            "name": "Player", "team": "Team", "pos": "Pos", "age": "Age",
            "cap_hit": "Cap Hit", "predicted_value": "Pred. Value",
            "value_delta": "$ Delta", "value_delta_pct": "% Delta",
        })

    with t1:
        st.caption(f"**Hidden Gems** — top {n} by {sort_by}")
        st.dataframe(fmt_table(df_c.nlargest(n, sort_col)),
                     use_container_width=True, hide_index=True)
    with t2:
        st.caption(f"**Overpaid** — bottom {n} by {sort_by}")
        st.dataframe(fmt_table(df_c.nsmallest(n, sort_col)),
                     use_container_width=True, hide_index=True)

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
                "<span style='font-size:1.1rem;font-weight:700;color:#B5975A;'>"
                "🔶 UFA / Unsigned Players</span>"
                "<span style='background:#5D4037;color:#CFD8DC;padding:2px 8px;"
                "border-radius:4px;font-size:.75rem;'>No current contract — predicted value only</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            top_ufa = ufa_filtered.nlargest(min(n, len(ufa_filtered)), "predicted_value")
            ufa_disp = top_ufa[["name", "team", "pos", "age", "predicted_value"]].copy()
            ufa_disp["age"] = ufa_disp["age"].apply(
                lambda v: f"{v:.0f}" if pd.notna(v) else "?"
            )
            ufa_disp["predicted_value"] = ufa_disp["predicted_value"].apply(fmt_m)
            ufa_disp = ufa_disp.rename(columns={
                "name": "Player", "team": "Team", "pos": "Pos", "age": "Age",
                "predicted_value": "Predicted Value",
            })
            st.dataframe(ufa_disp, use_container_width=True, hide_index=True)
            st.caption(
                f"Showing {len(top_ufa)} of {len(ufa_filtered)} UFA/unsigned players — "
                "sorted by predicted market value. These players have no active cap charge."
            )


# ── Tab 3: LA Kings ────────────────────────────────────────────────────────────
def tab_kings(df: pd.DataFrame):
    # Kings-specific header with logo
    kings_logo = team_logo_url("LAK")
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:16px;margin-bottom:8px;"
        f"padding:16px 20px;background:#0D0D14;border-radius:10px;"
        f"border-left:4px solid {KINGS_GOLD};'>"
        f"<img src='{kings_logo}' width='72' height='72' "
        f"style='flex-shrink:0;' onerror=\"this.style.display='none'\">"
        f"<div>"
        f"<div style='font-size:1.6rem;font-weight:800;color:{KINGS_WHITE};'>"
        f"Los Angeles Kings — {_season_str(load_season_context())} Roster Analysis</div>"
        f"<div style='color:{KINGS_SILVER};font-size:.9rem;margin-top:2px;'>"
        f"Live NHL API · PuckPedia Contract Database · XGBoost Model</div>"
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
        f"<div style='background:#0D0D14;border-top:3px solid {KINGS_GOLD};"
        "border-radius:10px;padding:16px 20px;margin-bottom:16px;'>",
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
        textposition="outside", textfont=dict(size=9, color=KINGS_SILVER),
    )
    fig.add_bar(
        name="Predicted Value",
        x=kings_sorted["name"], y=kings_sorted["predicted_value"],
        marker_color=KINGS_GOLD, opacity=0.9,
    )
    fig.update_layout(
        barmode="group",
        paper_bgcolor="#0A0A14", plot_bgcolor="#0D0D14",
        font=dict(family="Inter, sans-serif", color="#CCCCDD"),
        xaxis=dict(tickangle=-40, gridcolor="#1A1A28"),
        yaxis=dict(tickformat="$,.0f", title="",
                   gridcolor="#1A1A28", zeroline=False),
        title=dict(text="Cap Hit vs. Predicted Market Value",
                   font=dict(color=KINGS_GOLD, size=15)),
        legend=dict(bgcolor="#13131F", bordercolor="#2A2A3E",
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
        f"<div style='color:{KINGS_GOLD};font-size:1.1rem;font-weight:700;"
        "margin:12px 0 8px;'>Player Breakdown</div>",
        unsafe_allow_html=True,
    )

    tbl = kings_all.sort_values("value_delta", ascending=False, na_position="last").copy()

    def _render_kings_group(group_df, group_label):
        """Render one positional group (Forwards / Defensemen) as player cards."""
        if group_df.empty:
            return
        st.markdown(
            f"<div style='color:{KINGS_SILVER};font-size:.8rem;font-weight:600;"
            f"letter-spacing:.08em;text-transform:uppercase;margin:14px 0 6px;"
            f"padding-left:4px;border-left:3px solid {KINGS_GOLD};'>"
            f"{group_label}</div>",
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
        sig_color = KINGS_SIGNAL_PALETTE.get(signal, "#37474F")

        delta_str = "—"
        pct_str   = ""
        if pd.notna(delta):
            sign = "+" if delta >= 0 else ""
            clr  = "#4CAF50" if delta >= 0 else "#F44336"
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
            f"<span style='background:#37474F;color:#CFD8DC;padding:1px 5px;"
            f"border-radius:3px;font-size:.65rem;margin-left:4px;' "
            f"title='Salary estimated — contract data pending verification'>est*</span>"
        ) if is_est else ""

        st.markdown(
            f"<div class='kings-card' style='display:flex;align-items:center;gap:16px;'>"
            f"  {hs_html}"
            f"  <div style='flex:1;min-width:0;'>"
            f"    <div style='display:flex;align-items:baseline;gap:8px;'>"
            f"      <span style='font-size:1.05rem;font-weight:700;color:{KINGS_WHITE};'>{name}</span>"
            f"      {est_badge}"
            f"      <span style='color:{KINGS_SILVER};font-size:.85rem;'>{pos} · Age {age_str}</span>"
            f"    </div>"
            f"    <div style='display:flex;gap:24px;margin-top:6px;flex-wrap:wrap;'>"
            f"      <div><div class='stat-label'>Cap Hit</div>"
            f"           <div class='stat-value'>{ch_str}</div></div>"
            f"      <div><div class='stat-label'>Pred. Value</div>"
            f"           <div class='stat-value'>{pv_str}</div></div>"
            f"      <div><div class='stat-label'>Value Delta</div>"
            f"           <div style='font-size:.95rem;font-weight:700;'>"
            f"             {delta_str} {pct_str}</div></div>"
            f"      <div><div class='stat-label'>Expiry</div>"
            f"           <div class='stat-value'>{exp_str} ({exp_st})</div></div>"
            f"      <div><div class='stat-label'>Yrs Left</div>"
            f"           <div class='stat-value'>{yrs_str}</div></div>"
            f"    </div>"
            f"  </div>"
            f"  <div style='text-align:center;flex-shrink:0;'>"
            f"    <div class='stat-label'>Signal</div>"
            f"    <span style='background:{sig_color};color:#fff;padding:4px 10px;"
            f"border-radius:6px;font-size:.8rem;font-weight:700;display:inline-block;"
            f"margin-top:4px;white-space:nowrap;'>{signal}</span>"
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
    st.markdown(
        f"<div style='color:{KINGS_SILVER};font-size:.8rem;margin-bottom:8px;'>"
        "Re-sign Signal Legend</div>",
        unsafe_allow_html=True,
    )
    leg_cols = st.columns(len(KINGS_SIGNAL_PALETTE))
    for i, (lbl, clr) in enumerate(KINGS_SIGNAL_PALETTE.items()):
        leg_cols[i].markdown(
            f"<span style='background:{clr};color:#fff;padding:3px 8px;"
            f"border-radius:4px;font-size:.72rem;white-space:nowrap;'>{lbl}</span>",
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
        "<span style='background:#B71C1C;color:#fff;padding:2px 7px;"
        "border-radius:3px;font-size:.68rem;margin-left:6px;'>NO PRIOR STATS</span>"
    )
    if not has_contract and exp_status.upper() == "UFA":
        contract_badge = (
            "<span style='background:#5D4037;color:#fff;padding:2px 7px;"
            "border-radius:3px;font-size:.68rem;margin-left:6px;'>UFA / UNSIGNED</span>"
        )
    elif is_estimated:
        contract_badge = (
            "<span style='background:#37474F;color:#CFD8DC;padding:2px 7px;"
            "border-radius:3px;font-size:.68rem;margin-left:6px;' "
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
            f"border:2px solid #B5975A;flex-shrink:0;' "
            f"onerror=\"this.style.display='none'\">"
        )

    st.markdown(
        f"<div class='player-card'>"
        f"  <div style='display:flex;gap:16px;align-items:center;'>"
        f"    {hs_html}"
        f"    <div>"
        f"      <div style='font-size:1.4rem;font-weight:800;color:#EEE;'>"
        f"        {name}{prior_badge}{contract_badge}"
        f"      </div>"
        f"      <div style='color:#888;margin-top:3px;font-size:.9rem;'>"
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

    # Re-sign signal
    signal    = player.get("resign_signal", "—")
    sig_color = RESIGN_PALETTE.get(signal, "#555")
    st.markdown(
        f"<div style='margin:8px 0;'>"
        f"<span style='color:#777799;font-size:.75rem;letter-spacing:.04em;'>"
        f"RE-SIGN SIGNAL &nbsp;</span>"
        f"<span style='background:{sig_color};color:#fff;padding:4px 12px;"
        f"border-radius:6px;font-size:.88rem;font-weight:600;'>{signal}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # League rank
    rank     = int((df["predicted_value"] >= (pv or 0)).sum())
    total    = len(df)
    rank_pct = pct_rank(df["predicted_value"], pv or 0)
    st.markdown(
        f"<div style='color:#9999BB;font-size:.88rem;'>League rank by predicted value: "
        f"<strong style='color:#DDD;'>#{rank} of {total}</strong> "
        f"(top <strong style='color:#DDD;'>{100-rank_pct}%</strong>)</div>",
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
        fig_pct.update_traces(textposition="inside", textfont_size=11)
        fig_pct.update_layout(
            paper_bgcolor="#0A0A14", plot_bgcolor="#111120", font_color="#CCCCDD",
            showlegend=False, coloraxis_showscale=False,
            xaxis=dict(range=[0, 100], title="Percentile", gridcolor="#1E1E30"),
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
            f"<div style='margin:16px 0 4px 0;font-size:1.05rem;font-weight:700;color:#EEE;'>"
            f"What drives {name}'s value?</div>"
            f"<div style='color:#777799;font-size:12px;margin-bottom:12px;'>"
            f"Starting from the league average salary, here is what pushes this player's value up or down.</div>",
            unsafe_allow_html=True,
        )

        # Section 1 — league average
        st.markdown(
            f"<div style='background:#1A1A2E;border-radius:8px;padding:10px 16px;"
            f"margin-bottom:14px;font-size:14px;color:#AAAACC;'>"
            f"League Average &nbsp;<strong style='color:#DDD;'>${base/1e6:.2f}M</strong></div>",
            unsafe_allow_html=True,
        )

        # Section 2 — side-by-side factors
        col_up, col_dn = st.columns(2)

        with col_up:
            st.markdown(
                "<div style='color:#43A047;font-weight:700;font-size:13px;"
                "margin-bottom:4px;'>↑ Pushing value UP</div>",
                unsafe_allow_html=True,
            )
            fig_up = _driver_chart(pos_factors, player, name, True, max_abs)
            if fig_up:
                st.plotly_chart(fig_up, use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.markdown("<span style='color:#555566;font-size:12px;'>None</span>",
                            unsafe_allow_html=True)

        with col_dn:
            st.markdown(
                "<div style='color:#E53935;font-weight:700;font-size:13px;"
                "margin-bottom:4px;'>↓ Pushing value DOWN</div>",
                unsafe_allow_html=True,
            )
            fig_dn = _driver_chart(neg_factors, player, name, False, max_abs)
            if fig_dn:
                st.plotly_chart(fig_dn, use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.markdown("<span style='color:#555566;font-size:12px;'>None</span>",
                            unsafe_allow_html=True)

        # Section 3 — result
        pv_val   = pv or base
        pv_color = "#43A047" if (delta or 0) >= 0 else "#E53935"
        cap_str  = f"${ch/1e6:.2f}M" if ch else "—"
        if delta is not None:
            delta_str = f"+${delta/1e6:.2f}M" if delta >= 0 else f"-${abs(delta)/1e6:.2f}M"
        else:
            delta_str = "—"
        st.markdown(
            f"<div style='background:#13131F;border:1px solid #2A2A3F;border-radius:10px;"
            f"padding:16px 20px;margin-top:16px;'>"
            f"<div style='font-size:1.25rem;font-weight:800;color:{pv_color};margin-bottom:6px;'>"
            f"Estimated Market Value: ${pv_val/1e6:.2f}M</div>"
            f"<div style='font-size:13px;color:#777799;'>"
            f"Current Cap Hit: <span style='color:#AAAACC;'>{cap_str}</span>"
            f"&nbsp;&nbsp;·&nbsp;&nbsp;"
            f"Difference: <span style='color:{pv_color};font-weight:600;'>{delta_str}</span>"
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
                    f"border:1px solid #2A2A3E;' "
                    f"onerror=\"this.style.display='none'\">"
                )
            clr = "#4CAF50" if (sp_dlt or 0) >= 0 else "#F44336"
            sim_cols[col_i].markdown(
                f"<div style='background:#13131F;border-radius:8px;padding:12px;"
                f"border:1px solid #2A2A3E;text-align:center;'>"
                f"  {sp_hs}"
                f"  <div style='font-weight:700;color:#EEE;margin-top:6px;"
                f"font-size:.88rem;'>{sp_name}</div>"
                f"  <div style='color:#888;font-size:.75rem;'>"
                f"    {sp_team} · {sp_pos}"
                f"    {f'· Age {sp_age:.0f}' if pd.notna(sp_age) else ''}"
                f"  </div>"
                f"  <div style='margin-top:6px;font-size:.8rem;'>"
                f"    <span style='color:#AAA;'>Pred: </span>"
                f"    <span style='color:#EEE;'>{fmt_m(sp_pv)}</span>"
                f"  </div>"
                f"  <div style='color:{clr};font-size:.78rem;font-weight:700;'>"
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
    st.markdown("## Player Search")

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
    st.markdown("## Model Insights — SHAP Feature Importance")
    shap_summary = load_shap_summary()
    shap_vals    = load_shap_values()

    if shap_summary.empty:
        st.info("Run `py -3 pipeline.py` to generate SHAP values.")
        return

    st.markdown("### What drives predicted player value?")
    top_n = st.slider("Features to show", 5, min(30, len(shap_summary)), 15)
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
        paper_bgcolor="#0A0A14", plot_bgcolor="#111120",
        font=dict(color="#CCCCDD"), showlegend=False, coloraxis_showscale=False,
        xaxis=dict(
            tickvals=tick_vals, ticktext=tick_text,
            title="Avg. Dollar Impact on Prediction", gridcolor="#1E1E30",
        ),
        margin=dict(l=10, r=20, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Mean absolute SHAP value = average dollar impact of each feature on predictions.")

    if not shap_vals.empty and "name" in shap_vals.columns:
        st.markdown("---")
        st.markdown("### Player-Level Explanation")
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
                paper_bgcolor="#0A0A14", plot_bgcolor="#111120",
                font=dict(color="#CCCCDD"), showlegend=False, coloraxis_showscale=False,
                xaxis=dict(
                    tickvals=tick_vals2, ticktext=tick_text2,
                    title="Dollar Impact on Prediction", gridcolor="#1E1E30",
                    zeroline=True, zerolinecolor="#555566", zerolinewidth=2,
                ),
                title=dict(text=f"SHAP Breakdown — {chosen}",
                           font=dict(color="#CCCCEE", size=14)),
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

    st.markdown("---")
    _last_updated = (f" &nbsp;·&nbsp; Last updated: <strong>{last_ts}</strong>" if last_ts else "")
    _season_info = f"&nbsp;·&nbsp; {_season_str(load_season_context())} Season &nbsp;·&nbsp; Cap ceiling: ${CAP_CEILING/1e6:.1f}M"
    st.markdown(
        f"<div style='color:#555566;font-size:.78rem;text-align:center;padding:8px 0 4px;'>"
        f"Data: <strong>NHL API</strong> (rosters, stats, headshots) &nbsp;·&nbsp; "
        f"<strong>PuckPedia</strong> (contract database) &nbsp;·&nbsp; "
        f"{n_real} real contracts · {n_est} estimated"
        f"{_last_updated}{_season_info}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Kick off background data refresh on every cold start (non-blocking)
    start_background_refresh(PROCESSED_DIR)

    # If background job just finished, trigger a rerun so UI picks up fresh mtime
    if _refresh_status["done"]:
        _refresh_status["done"] = False
        st.rerun()

    st.markdown(
        f"<div style='display:flex;align-items:baseline;gap:12px;margin-bottom:0;'>"
        f"<h1 style='margin:0;color:#F0F0FF;font-size:1.8rem;'>🏒 NHL Player Value Model</h1>"
        f"<span style='color:#666688;font-size:.9rem;'>"
        f"{_season_str(load_season_context())} · XGBoost + SHAP · Live Data</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:4px;background:linear-gradient(90deg,#B5975A,#A2AAAD,#010101);border-radius:2px;margin-bottom:16px;'></div>", unsafe_allow_html=True)

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
