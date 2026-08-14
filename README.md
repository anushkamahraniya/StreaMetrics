# 🎬 StreaMetrics — AI Creator Monetization & Cohort Analytics

An end-to-end analytics stack that turns raw watch-event logs into creator payouts, virality scores, retention curves, and a statistically validated growth lever — all explorable in a live Streamlit dashboard.

---

## 📌 Problem Statement & Business Value

StreaMetrics is a fictional AI video platform that acquires users through four channels and lets viewers remix AI-generated clips, paying creators along the way. Leadership needs three questions answered with data, not gut feel:

- **Which creators and content actually drive watch time, revenue, and remixes** — so payouts and promotion go where they work?
- **Which AI models and genres win with viewers**, to steer the content roadmap?
- **Which acquisition channels bring in users who stick around**, so growth spend chases durable retention instead of cheap signups?

**Real-world failure this prevents:** without this pipeline, teams optimize for the wrong signal — e.g. rewarding the channel with the best Day-1 retention (Paid Social, 74.8%) while ignoring that it collapses to 1.9% by Day 60, or crediting a creator's raw view count instead of what viewers actually *do* with their videos. The Chi-Square test also converts "early remixing feels like it helps retention" from a hunch into a validated (p ≈ 6.9×10⁻⁸) activation metric worth building onboarding around.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3 |
| **Database** | SQLite (`streaming_analytics.db`) |
| **Data / Analysis** | pandas, numpy, scipy (Chi-Square test) |
| **Visualization** | Plotly Express, matplotlib, seaborn |
| **App / Dashboard** | Streamlit |
| **SQL** | Window functions — `DENSE_RANK()`, `LAG()` |

---

## ✨ Key Features & Edge Cases Handled

- **Reproducible synthetic data** — `generate_data.py` uses a fixed seed (42), so the full dataset regenerates identically every run.
- **Referential integrity enforced at load** — `PRAGMA foreign_keys = ON` plus explicit `FOREIGN KEY` constraints on `watch_events` catch orphaned `user_id` / `video_id` references before they reach analytics.
- **No SQLite CLI in the environment** — `build_database.py` and `run_analytics_queries.py` re-implement the DDL/query execution through Python's built-in `sqlite3` module, so the pipeline still runs end-to-end where the `sqlite3` binary isn't installed; the original `.sql` files remain portable and runnable directly wherever it is.
- **Zero-selection & empty-result guards** — the Streamlit sidebar filters can be cleared to zero options; the app detects this and shows a warning instead of crashing on an empty dataframe.
- **Divide-by-zero protection** — Virality Score and the Forecast Simulator both guard against a video with 0 views before dividing.
- **Widget-state clamping** — the ROI calculator clamps its projection to 0 when Streamlit's slider state hasn't caught up with a raised minimum, so projections never go negative.
- **In-progress cohorts aren't miscounted as decline** — trailing months in the growth trend are still inside their 90-day observation window; the app calls this out explicitly instead of implying a drop-off.
- **Correlation vs. causation flagged in-product** — the A/B test result is explicitly labeled as an observed association, not a randomized experiment, right in the UI.

---

## 🏗️ Architecture Overview

```
generate_data.py  →  users.csv / videos.csv / watch_events.csv
        │
        ▼
build_database.py  →  streaming_analytics.db  (users, videos, watch_events)
        │
        ├──▶ run_analytics_queries.py   (creator leaderboard, MoM growth, virality rank)
        ├──▶ cohort_analysis.py         (Day 1/7/14/30/60 retention matrix + heatmap)
        ├──▶ ab_test_analysis.py        (Chi-Square: early remix vs. Day-30 retention)
        └──▶ app.py                     (Streamlit — 7 tabs, live filters, forecast simulator)
```

**Schema:** `users(user_id PK)` and `videos(video_id PK)` are referenced by `watch_events` via foreign keys on `user_id` and `video_id`. Every engagement signal (like, share, remix, completion rate, ad revenue) lives at the event grain in `watch_events`, so all downstream metrics are aggregations over that one table.

---

## 🚀 Setup & Execution Guide

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data (seed=42, fully reproducible)
python python/generate_data.py

# 3. Build the SQLite warehouse
python python/build_database.py

# 4. Run the SQL analytics suite
python python/run_analytics_queries.py

# 5. Run cohort retention analysis (exports heatmap to screenshots/)
python python/cohort_analysis.py

# 6. Run the A/B test (Chi-Square: early remix vs. retention)
python python/ab_test_analysis.py

# 7. Launch the dashboard
streamlit run python/app.py
```

---

## 📊 Sample Results

**Day-30 retention by acquisition channel:**

| Channel | Day 1 | Day 7 | Day 30 | Day 60 |
|---|---|---|---|---|
| Paid Social | 74.8% | 71.6% | 20.4% | 1.9% |
| Search Ads | 65.8% | 62.8% | 34.6% | 9.7% |
| Organic | 58.9% | 65.1% | 36.3% | 17.3% |
| Creator Referral | 47.2% | 56.8% | **41.0%** | **27.1%** |

**Virality Score by AI model:**

| AI Model | Videos | Avg Score | Max Score |
|---|---|---|---|
| Flux | 114 | **54.70** | 98.27 |
| Sora 2 | 183 | 53.51 | 106.22 |
| Kling 2.0 | 177 | 52.28 | 106.38 |
| Seedance Pro | 126 | 52.01 | 107.62 |

**Early remix → Day-30 retention (Chi-Square test):**

| Segment | n | Retention Rate |
|---|---|---|
| Early Remixer | 550 | **42.73%** |
| Passive Viewer | 1,450 | 29.86% |

p ≈ 6.9 × 10⁻⁸ — statistically significant at 90/95/99% confidence. Early remixers retain ~1.4x more than passive viewers.
