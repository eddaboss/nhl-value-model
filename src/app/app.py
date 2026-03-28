"""
NHL Player Value Model — Streamlit App
Tabs: League Overview | Leaderboards | LA Kings | Player Search | Model Insights
"""
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from src.data.load import load_and_merge, CAP_CEILING
from src.features.build import build_features, get_feature_matrix, resign_label
from src.models.train import load_model

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NHL Value Model",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
  .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

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
MODELS_DIR    = Path(__file__).parents[2] / "models"

_KEEP_COLS = [
    "name", "team", "pos", "age",
    "cap_hit", "predicted_value", "value_delta",
    "expiry_status", "expiry_year", "years_left", "length_of_contract",
    "gp", "g", "a", "p", "ppg",
    "toi_per_g", "plus_minus", "pim",
    "g60", "p60", "pp_pts", "shots", "shooting_pct",
    "resign_signal", "player_id",
    "has_contract_data", "has_prior_market_data", "is_estimated",
]


# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)   # refresh live data every hour
def load_predictions() -> pd.DataFrame:
    """
    Build predictions from live sources every hour.
    - NHL API stats:  fetched live (24-hour disk cache in data/raw/)
    - Contracts:      read from data/contracts.db (committed to repo)
    - Model:          loaded from models/xgb.pkl (committed to repo)
    No manual pipeline run needed — the app always serves fresh data.
    """
    df_raw, _ctx = load_and_merge()
    df = build_features(df_raw)
    X, _ = get_feature_matrix(df)

    model = load_model("xgb")
    df["predicted_value"] = model.predict(X) * CAP_CEILING
    df["value_delta"] = df.apply(
        lambda r: r["predicted_value"] - r["cap_hit"]
        if r.get("has_contract_data") else None, axis=1
    )
    df["resign_signal"] = df.apply(resign_label, axis=1)

    out = df[[c for c in _KEEP_COLS if c in df.columns]].copy()
    for col in ["cap_hit", "predicted_value", "value_delta"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "is_estimated" not in out.columns:
        out["is_estimated"] = False
    out["is_estimated"] = out["is_estimated"].fillna(False).astype(bool)
    return out


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


def headshot_url(player_id, team: str = "") -> str:
    team = str(team).upper().strip() if team else ""
    if team:
        return f"https://assets.nhle.com/mugs/nhl/20252026/{team}/{int(player_id)}.png"
    return f"https://assets.nhle.com/mugs/nhl/20252026/{int(player_id)}.png"


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

        # Last updated
        lu = load_last_updated()
        if lu:
            ts = (lu.get("nightly_stats") or lu.get("weekly_contracts") or {}).get("timestamp", "")
            if ts:
                from datetime import datetime, timezone
                try:
                    dt = datetime.fromisoformat(ts).astimezone()
                    st.caption(f"Last refreshed: {dt.strftime('%b %d %Y %I:%M %p')}")
                except Exception:
                    st.caption(f"Last refreshed: {ts[:10]}")

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
        f"Los Angeles Kings — 2025-26 Roster Analysis</div>"
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

    # SHAP waterfall
    if (not shap_vals.empty and "name" in shap_vals.columns
            and name in shap_vals["name"].values):
        st.markdown("**What drives this player's predicted value?**")
        row_shap = shap_vals[shap_vals["name"] == name].iloc[0].drop("name")
        vals     = row_shap.astype(float)
        top12    = vals.abs().nlargest(12)
        base     = float(df["predicted_value"].mean())
        features = [f.replace("_", " ").title() for f in top12.index]
        shap_v   = top12.values.tolist()
        measure  = ["absolute"] + ["relative"] * len(shap_v) + ["total"]
        x_labels = ["Base Value"] + features + ["Predicted Value"]
        y_vals   = [base] + shap_v + [(pv or base)]

        fig_shap = go.Figure(go.Waterfall(
            orientation="h", measure=measure,
            x=y_vals, y=x_labels,
            connector=dict(line=dict(color="#333", width=1)),
            increasing=dict(marker_color="#4CAF50"),
            decreasing=dict(marker_color="#F44336"),
            totals=dict(marker_color=KINGS_GOLD),
            textposition="outside",
            text=[
                (f"+{fmt_m(v)}" if v > 0 else fmt_m(v)) if i not in (0, len(y_vals) - 1)
                else fmt_m(v)
                for i, v in enumerate(y_vals)
            ],
        ))
        fig_shap.update_layout(
            paper_bgcolor="#0A0A14", plot_bgcolor="#111120", font_color="#CCCCDD",
            height=max(400, len(features) * 38 + 80),
            xaxis=dict(tickformat="$,.0f", title="$ Value", gridcolor="#1E1E30"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=0, r=80, t=20, b=10),
            title=dict(text=f"SHAP Waterfall — {name}", font=dict(color="#CCCCEE", size=13)),
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption(
            "🟢 Green = feature pushes value UP · 🔴 Red = pushes DOWN · "
            "Starting from league average predicted value."
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
- *Prior season stats* — same metrics from 2024-25, for players with prior data
- *Contract structure* — years remaining, contract length
- *Bio* — age, draft position, draft round
- *Position* — affects TOI benchmarks (D vs F)

**What comes out:**
The model predicts a player's **market-rate cap hit** — what they would command
on the open free-agent market — expressed as a fraction of the current cap ceiling
($95.5M), then scaled back to dollars.

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
    top["feature"] = top["feature"].str.replace("_", " ").str.title()

    fig = px.bar(
        top.sort_values("mean_abs_shap"),
        x="mean_abs_shap", y="feature", orientation="h",
        color="mean_abs_shap", color_continuous_scale="Tealgrn",
        labels={"mean_abs_shap": "Mean |SHAP| ($)", "feature": ""},
        height=max(350, top_n * 28),
    )
    fig.update_layout(
        paper_bgcolor="#0A0A14", plot_bgcolor="#111120",
        font=dict(color="#CCCCDD"), showlegend=False, coloraxis_showscale=False,
        xaxis=dict(tickformat="$,.0f", gridcolor="#1E1E30"),
        margin=dict(l=0, r=10, t=10, b=10),
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
            pdf   = pd.DataFrame({
                "Feature": [f.replace("_", " ").title() for f in top12],
                "SHAP":    row[top12].values,
            }).sort_values("SHAP")

            fig2 = px.bar(
                pdf, x="SHAP", y="Feature", orientation="h",
                color="SHAP", color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                labels={"SHAP": "SHAP Value ($)", "Feature": ""},
                height=420,
            )
            fig2.update_layout(
                paper_bgcolor="#0A0A14", plot_bgcolor="#111120",
                font=dict(color="#CCCCDD"), showlegend=False, coloraxis_showscale=False,
                xaxis=dict(tickformat="$,.0f", gridcolor="#1E1E30"),
                title=dict(text=f"SHAP Breakdown — {chosen}",
                           font=dict(color="#CCCCEE", size=13)),
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
    st.markdown(
        f"<div style='color:#555566;font-size:.78rem;text-align:center;padding:8px 0 4px;'>"
        f"Data: <strong>NHL API</strong> (rosters, stats, headshots) &nbsp;·&nbsp; "
        f"<strong>PuckPedia</strong> (contract database) &nbsp;·&nbsp; "
        f"{n_real} real contracts · {n_est} estimated"
        + (f" &nbsp;·&nbsp; Last updated: <strong>{last_ts}</strong>" if last_ts else "")
        + f"&nbsp;·&nbsp; 2025-26 Season &nbsp;·&nbsp; Cap ceiling: $95.5M"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    st.markdown(
        f"<div style='display:flex;align-items:baseline;gap:12px;margin-bottom:0;'>"
        f"<h1 style='margin:0;color:#F0F0FF;font-size:1.8rem;'>🏒 NHL Player Value Model</h1>"
        f"<span style='color:#666688;font-size:.9rem;'>"
        f"2025-26 · XGBoost + SHAP · Live Data</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:4px;background:linear-gradient(90deg,#B5975A,#A2AAAD,#010101);border-radius:2px;margin-bottom:16px;'></div>", unsafe_allow_html=True)

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
