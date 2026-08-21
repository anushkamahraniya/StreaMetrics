# StreaMetrics — Creator Monetization & Cohort Analytics

**Live dashboard:** [streametrics.streamlit.app](https://streametrics.streamlit.app)

A pipeline that turns raw watch-event logs from a fictional AI video platform into creator payouts, virality scores, retention curves, and a statistically validated growth signal — explorable in a Streamlit dashboard.

It answers three questions: which creators/content actually drive revenue, which AI models/genres win with viewers, and which acquisition channels bring in users who stick around (not just sign up cheap). For example, Paid Social has strong Day-1 retention (74.8%) but collapses to 1.9% by Day 60, while Creator Referral starts weaker and ends up retaining the most. A Chi-Square test also confirms early remixing predicts retention (p ≈ 6.9×10⁻⁸).

---

## Tech stack

Python 3 · SQLite · pandas / numpy / scipy · Plotly / matplotlib / seaborn · Streamlit

---

## Architecture

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

Data is seeded (42) for reproducibility, foreign keys are enforced at load, and the pipeline runs without needing the `sqlite3` CLI installed.

---

## Setup

```bash
pip install -r requirements.txt
python python/generate_data.py
python python/build_database.py
python python/run_analytics_queries.py
python python/cohort_analysis.py
python python/ab_test_analysis.py
streamlit run python/app.py
```

---

## Sample results

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

p ≈ 6.9 × 10⁻⁸ — early remixers retain about 1.4x more than passive viewers.
