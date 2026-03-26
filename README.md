# NHL Player Value Model

An interactive Streamlit app that estimates the **true market value** of every active NHL skater using an XGBoost model trained on live contract and performance data.

**Live app:** *(add your Streamlit Cloud URL here after deployment)*

---

## What it does

For each of the ~749 active NHL skaters, the model predicts what their contract would be worth on the open market and compares that to their actual cap hit. The result is a **value delta** — the difference between what a player *should* cost and what they *actually* cost.

- **Positive delta** → team surplus (underpaid relative to production)
- **Negative delta** → team liability (paid above model-estimated market rate)

The app includes five views:

| Tab | What you get |
|-----|-------------|
| **League Overview** | Interactive scatter of cap hit vs predicted value for all 749 players. UFA/unsigned players shown separately as gold diamonds. |
| **Leaderboards** | Ranked lists of most underpaid and most overpaid players with headshots, position filter, and % delta sort. |
| **LA Kings** | Full Kings roster analysis with cap summary, next-season outlook, and color-coded re-sign recommendations. |
| **Player Search** | Detailed player card: headshot, contract breakdown, league rank, stat percentiles, SHAP waterfall, and 3 most similar players. |
| **Model Insights** | Feature importance (SHAP) and per-player explanations. |

---

## Data sources

| Source | What it provides | How often refreshed |
|--------|-----------------|---------------------|
| [NHL API](https://api-web.nhle.com/v1) | Rosters, per-player stats (GP, G, A, TOI, PP pts, S%), draft info, headshots | Nightly |
| [PuckPedia](https://puckpedia.com) | Cap hit, contract length, expiry year, expiry status (UFA/RFA) | Weekly |

No scraped CSVs are committed to this repo. All data is fetched live from public APIs and cached locally in `data/raw/`.

---

## Methodology

### Model

**Algorithm:** XGBoost with 5-fold cross-validation
**Target:** cap hit as a fraction of the salary cap ceiling ($95.5M for 2025-26), then rescaled to dollars
**Training set:** ~635 players with verified contracts
**CV R²:** 0.829 | **CV RMSE:** ~$1.21M

### Feature engineering

The model uses 32 features across five groups:

**Performance (current season)**
- Points, goals, assists, PPG
- TOI/game, PP points, shots, shooting %
- G/60 and P/60 (rate stats normalized for ice time)
- All stats projected to 82-game pace from actual GP

**Performance (prior season)**
- Same metrics from 2024-25
- Blended with current-season stats when fewer than 25% of regular-season games have been played (see season blending below)

**Contract structure**
- Years remaining on current contract
- Total contract length
- Year of current contract

**Bio**
- Age (as of Sep 30 of the season end year)
- Draft position (overall pick; 999 = undrafted)
- Draft round tier (top-10 / round-1 / after-round-1 / undrafted)

**Position**
- Forward (C/L/R) vs defenseman affects all TOI-based benchmarks

### Season blending

Early in the season (before 25% of total team-games played), current-season stats are statistically noisy. The model blends prior-year and current-year stats:

```
blend_weight = (avg games played) / (season length)
blended_stat = blend_weight × current_stat_projected + (1 - blend_weight) × prior_stat
```

After the 25% threshold, only current-season projected stats are used. This prevents early-season outlier performances from distorting valuations.

### Why cap hit instead of total contract value

Total contract value (TCV) is inflated by signing bonuses and performance bonuses that don't translate year-to-year. **Cap hit** is the annual average value (AAV) — what the team actually pays against the salary cap each year. It's the honest, comparable unit of NHL contract value.

### Contract estimation

For players where contract data cannot be scraped (typically fringe roster players on PTOs or AHL callups), the model estimates salary from position/TOI tier medians computed from real contracts in the database:

- **ELC** (age < 25, <3 NHL seasons): $975K
- **Top-line F/D** (TOI ≥ 20 min): median of real top-line contracts
- **Mid-tier** (15–20 min): position median
- **Bottom-6/7-8** (< 15 min): position median

Estimated salaries are marked with an asterisk (*) throughout the app.

---

## Running locally

### Prerequisites

- Python 3.11+
- ~2 GB disk (for model, caches, contracts DB)

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/nhl-value-model.git
cd nhl-value-model
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### First run (build predictions)

```bash
# Rebuild predictions from live NHL API + contracts DB
py -3 pipeline.py --refresh

# Launch the app
py -3 -m streamlit run src/app/app.py
```

The pipeline takes ~3–5 minutes on first run (fetching stats for 822 players). Subsequent runs use the cache and complete in ~30 seconds.

### Automated refresh

```bash
# Start the scheduler (nightly stats + weekly contract check)
py -3 scheduler.py

# Or run manually:
py -3 scheduler.py --run-stats       # stats refresh + retrain now
py -3 scheduler.py --run-contracts   # weekly contract check now
```

| Schedule | What runs |
|----------|-----------|
| Daily 11 PM PST | NHL API stats refresh + model retrain |
| Sunday midnight PST | Incremental contract re-scrape (expiring/stale players only) + retrain |

---

## Project structure

```
nhl-value-model/
├── src/
│   ├── app/app.py              # Streamlit app (5 tabs)
│   ├── data/
│   │   ├── nhl_api.py          # NHL API client + roster/stats cache
│   │   ├── contracts_db.py     # SQLite contracts DB layer
│   │   ├── exhaustive_scraper.py  # 7-source contract scraper
│   │   ├── puckpedia_scraper.py   # Parallel PuckPedia scraper
│   │   └── load.py             # Data merge pipeline
│   ├── features/build.py       # Feature engineering
│   └── models/
│       ├── train.py            # CV, model training, XGB tuning
│       └── explain.py          # SHAP artifacts
├── data/
│   ├── contracts.db            # SQLite: all contract data
│   ├── raw/                    # NHL API caches (JSON)
│   └── processed/              # Predictions CSV + SHAP values
├── models/xgb.pkl              # Trained XGBoost pipeline
├── pipeline.py                 # End-to-end training pipeline
├── scheduler.py                # APScheduler nightly/weekly refresh
├── contract_overrides.json     # Manual contract overrides
└── requirements.txt
```

---

## Notes

- The model is descriptive, not prescriptive. It tells you what the market *has* paid for similar production — not what a player *deserves*.
- Players on ELCs (entry-level contracts) will almost always show large positive deltas; this is expected and reflects the structure of the CBA, not model error.
- The value delta for UFA/unsigned players is not computed (no contract to compare against), but their predicted market value is shown.
- This project was built as a portfolio piece for a Data Scientist application to the Los Angeles Kings.

---

*Data: NHL API + PuckPedia · Model: XGBoost · 2025-26 season · Cap ceiling: $95.5M*
