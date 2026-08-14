"""
StreaMetrics | Product & Creator Health Analytics — Streamlit App

Reads data/streaming_analytics.db and presents core pipeline analyses, 
virality scoring, A/B testing insights, and interactive scenario forecasting.
"""

import os
import sqlite3

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
from ab_test_analysis import compute_ab_test_results, significance_verdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "streaming_analytics.db")

CONFIDENCE_LEVEL_OPTIONS = {"90% Confidence": 0.90, "95% Confidence": 0.95, "99% Confidence": 0.99}

# Economic constants (mirrored from python/generate_data.py)
BASE_CPM_USD_PER_SEC = 0.012
AVG_CREDIT_PER_RECREATION_USD = 2.75

RETENTION_DAY = 30
RETENTION_TOLERANCE = 3

st.set_page_config(
    page_title="StreaMetrics Analytics",
    page_icon="🎬",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Enhanced Visual Styling & Animations (Human-Centric & Interactive UI)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Modern ambient background */
    .stApp {
        background-color: #FAFAFC;
    }
    
    /* Interactive Elevated Cards & Metric Displays */
    [class*="st-key-card_"],
    div[data-testid="stMetric"],
    div[data-testid="stDataFrameResizable"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        padding: 14px !important;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.12), 0 4px 6px -2px rgba(79, 70, 229, 0.05) !important;
        border-color: #C7D2FE !important;
    }

    /* Vibrant Tag Chips for Multiselect */
    div[data-testid="stMultiSelectTagsContainer"] > span,
    div[data-testid="stMultiSelectTagsContainer"] > div {
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%) !important;
        color: #4338CA !important;
        border: 1px solid #C7D2FE !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    
    div[data-testid="stMultiSelectTagsContainer"] svg {
        fill: #4338CA !important;
    }

    /* Hero Branding Styling */
    .hero-title {
        font-size: 3.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4338CA 0%, #6366F1 50%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        text-align: center;
    }
    
    .hero-subtitle {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Data Loading & Preparation
# =============================================================================
@st.cache_data(ttl="1h", show_spinner="Gathering platform data...")
def load_core_tables(db_path: str):
    with sqlite3.connect(db_path) as conn:
        users = pd.read_sql_query("SELECT * FROM users", conn, parse_dates=["signup_date"])
        videos = pd.read_sql_query("SELECT * FROM videos", conn)
        watch_events = pd.read_sql_query(
            "SELECT * FROM watch_events", conn, parse_dates=["event_timestamp"]
        )
    return users, videos, watch_events


@st.cache_data(show_spinner=False)
def enrich_events(watch_events: pd.DataFrame, videos: pd.DataFrame) -> pd.DataFrame:
    return watch_events.merge(
        videos[["video_id", "creator_id", "genre", "ai_model_used"]], on="video_id", how="left"
    )


users_df, videos_df, watch_events_df = load_core_tables(DB_PATH)
events_df = enrich_events(watch_events_df, videos_df)

GENRES = sorted(videos_df["genre"].unique())
AI_MODELS = sorted(videos_df["ai_model_used"].unique())


# =============================================================================
# Global Sidebar Controls
# =============================================================================
with st.sidebar:
    st.markdown("### 🎛️ Customize View")
    selected_models = st.multiselect(
        "AI Generator Models", options=AI_MODELS, default=AI_MODELS, key="global_ai_model_filter"
    )
    selected_genres = st.multiselect(
        "Content Genres", options=GENRES, default=GENRES, key="global_genre_filter"
    )
    completion_range = st.slider(
        "Viewer Completion Rate (%)",
        min_value=0,
        max_value=100,
        value=(0, 100),
        key="global_completion_filter",
        help="Focus on videos watched to a specific completion threshold.",
    )
    st.caption("✨ Adjusting these controls dynamically updates all analytics tabs across the application.")

if not selected_models or not selected_genres:
    st.warning(
        "Please select at least one AI Model and one Genre in the sidebar to populate insights.",
        icon="⚠️",
    )
    st.stop()

filtered_events = events_df[
    events_df["ai_model_used"].isin(selected_models)
    & events_df["genre"].isin(selected_genres)
    & (events_df["completion_rate"] * 100 >= completion_range[0])
    & (events_df["completion_rate"] * 100 <= completion_range[1])
]

if filtered_events.empty:
    st.warning(
        "No matching video sessions found. Try broadening your filter settings in the sidebar.",
        icon="🔍",
    )
    st.stop()


def build_video_stats(events: pd.DataFrame) -> pd.DataFrame:
    stats = events.groupby("video_id").agg(
        creator_id=("creator_id", "first"),
        genre=("genre", "first"),
        ai_model_used=("ai_model_used", "first"),
        total_views=("event_id", "count"),
        total_recreations=("recreated_by_user", "sum"),
        total_shares=("shared_by_user", "sum"),
        total_likes=("liked_by_user", "sum"),
        avg_completion_rate=("completion_rate", "mean"),
    ).reset_index()

    stats["virality_score"] = (
        (
            (stats["total_recreations"] * 3.0 + stats["total_shares"] * 2.0 + stats["total_likes"])
            / stats["total_views"]
        )
        * stats["avg_completion_rate"]
        * 100
    ).round(2)
    stats["avg_completion_rate"] = stats["avg_completion_rate"].round(4)
    stats["genre_virality_rank"] = (
        stats.groupby("genre")["virality_score"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    return stats


def compute_cohort_matrix(users: pd.DataFrame, events: pd.DataFrame):
    retention_days = [1, 7, 14, 30, 60]
    tolerance = {1: 1, 7: 2, 14: 2, 30: 3, 60: 5}

    merged = events.merge(
        users[["user_id", "signup_date", "acquisition_channel"]], on="user_id", how="left"
    )
    merged["days_since_signup"] = (
        merged["event_timestamp"].dt.normalize() - merged["signup_date"]
    ).dt.days

    flags = pd.DataFrame({"user_id": users["user_id"]})
    flags["acquisition_channel"] = users["acquisition_channel"]
    for day_n in retention_days:
        tol = tolerance[day_n]
        mask = (merged["days_since_signup"] >= day_n - tol) & (
            merged["days_since_signup"] <= day_n + tol
        )
        retained_ids = set(merged.loc[mask, "user_id"].unique())
        flags[f"Day {day_n}"] = flags["user_id"].isin(retained_ids)

    day_cols = [f"Day {d}" for d in retention_days]
    matrix = (flags.groupby("acquisition_channel")[day_cols].mean() * 100).round(1)
    sizes = flags.groupby("acquisition_channel")["user_id"].count()
    matrix = matrix.loc[sizes.sort_values(ascending=False).index]
    return matrix, sizes


def download_csv_button(df: pd.DataFrame, file_name: str, key: str):
    st.download_button(
        "Export Data (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        icon="📥",
        key=key,
    )


# Palette for charts
LIGHT_CHART_COLORS = ["#4F46E5", "#0284C7", "#059669", "#D97706", "#DB2777"]
LIGHT_CHART_SEQUENTIAL = ["#EEF2FF", "#4338CA"]


def apply_light_chart_theme(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=LIGHT_CHART_COLORS,
        font=dict(family="Inter, sans-serif", color="#1E293B"),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


# =============================================================================
# Dashboard Header
# =============================================================================
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")

if os.path.exists(LOGO_PATH):
    st.logo(LOGO_PATH, size="large")

st.markdown(
    '<p class="hero-title">AI Creator Monetization &amp; Cohort Analytics</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="hero-subtitle">Real-time insights on audience engagement, creator earnings, and AI model impact.</p>',
    unsafe_allow_html=True,
)

(
    overview_tab,
    creator_tab,
    content_tab,
    cohort_tab,
    virality_tab,
    abtest_tab,
    simulator_tab,
) = st.tabs(
    [
        "📊 Executive Summary",
        "🏆 Creator Earnings",
        "⚙️ AI Model Breakdown",
        "🔄 User Loyalty & Retention",
        "🚀 Virality Spotlight",
        "🧪 Behavioral Experiments",
        "🔮 Forecast Simulator",
    ]
)


# =============================================================================
# Executive Summary
# =============================================================================
with overview_tab:
    st.caption("A high-level view of overall platform growth, engagement duration, and monetization.")

    total_watch_hours = filtered_events["watch_time_sec"].sum() / 3600
    total_ad_revenue = filtered_events["ad_revenue_usd"].sum()
    total_recreations = int(filtered_events["recreated_by_user"].sum())
    active_users_in_view = filtered_events["user_id"].nunique()

    with st.container(horizontal=True):
        st.metric("Active Viewers", f"{active_users_in_view:,}", help="Unique users watching within the selected scope.")
        st.metric("Total Watch Time", f"{total_watch_hours:,.0f} hrs", help="Cumulative video viewing duration.")
        st.metric("Ad Revenue Generated", f"${total_ad_revenue:,.0f}", help="Estimated total ad revenue collected.")
        st.metric("Community Remixes", f"{total_recreations:,}", help="Total number of derivative videos created.")

    monthly = filtered_events.copy()
    monthly["month"] = monthly["event_timestamp"].dt.to_period("M").dt.to_timestamp()
    monthly_agg = (
        monthly.groupby("month").agg(watch_hours=("watch_time_sec", lambda s: s.sum() / 3600))
        .reset_index()
    )

    with st.container(border=True, key="card_overview_trend"):
        st.markdown("### 📈 How is audience viewing time trending month-over-month?")
        fig = px.line(
            monthly_agg,
            x="month",
            y="watch_hours",
            markers=True,
            labels={"month": "Month", "watch_hours": "Stream Duration (Hours)"},
            color_discrete_sequence=LIGHT_CHART_COLORS,
        )
        apply_light_chart_theme(fig)
        fig.update_traces(hovertemplate="%{x|%B %Y}<br><b>%{y:,.1f} hrs</b> watched<extra></extra>")
        fig.update_layout(height=360, hovermode="x unified")
        st.plotly_chart(fig, theme=None)
        st.caption("💡 *Note: Recent months appear lower due to ongoing 90-day cohort observation windows, rather than a decline in actual user activity.*")


# =============================================================================
# Creator Earnings
# =============================================================================
with creator_tab:
    st.markdown("### Creator Leaderboard & Monetization")
    st.caption("Track top-performing creators based on total watch time, community remixes, and direct credit payouts.")

    creator_stats = (
        filtered_events.groupby("creator_id")
        .agg(
            video_count=("video_id", "nunique"),
            total_watch_hours=("watch_time_sec", lambda s: s.sum() / 3600),
            total_ad_revenue=("ad_revenue_usd", "sum"),
            total_recreations=("recreated_by_user", "sum"),
            total_credits_earned=("creator_credit_earned", "sum"),
        )
        .reset_index()
    )
    creator_stats["rank_by_watch_time"] = creator_stats["total_watch_hours"].rank(
        method="dense", ascending=False
    ).astype(int)
    creator_stats["rank_by_ad_revenue"] = creator_stats["total_ad_revenue"].rank(
        method="dense", ascending=False
    ).astype(int)
    creator_stats["rank_by_recreations"] = creator_stats["total_recreations"].rank(
        method="dense", ascending=False
    ).astype(int)
    creator_stats = creator_stats.sort_values("total_watch_hours", ascending=False)

    search_col, download_col = st.columns([3, 1])
    with search_col:
        creator_search = st.text_input(
            "Search Creator ID", placeholder="Search creator (e.g. CREATOR_079)...", key="creator_search"
        )
    creator_display = creator_stats
    if creator_search:
        creator_display = creator_stats[
            creator_stats["creator_id"].str.contains(creator_search, case=False, na=False)
        ]
    with download_col:
        st.write("")
        download_csv_button(creator_display, "creator_leaderboard.csv", "download_creator_csv")

    st.caption(f"Displaying **{len(creator_display):,}** of **{len(creator_stats):,}** creators — click column headers to reorder.")
    st.dataframe(
        creator_display,
        hide_index=True,
        column_config={
            "creator_id": st.column_config.TextColumn("Creator Name / ID", pinned=True),
            "video_count": st.column_config.NumberColumn("Videos Published"),
            "total_watch_hours": st.column_config.NumberColumn("Stream Duration (hrs)", format="%.2f"),
            "total_ad_revenue": st.column_config.NumberColumn("Ad Revenue Generated ($)", format="$%.2f"),
            "total_recreations": st.column_config.NumberColumn("Remixes Generated"),
            "total_credits_earned": st.column_config.NumberColumn("Creator Payout ($)", format="$%.2f"),
            "rank_by_watch_time": st.column_config.NumberColumn("Watch Time Rank"),
            "rank_by_ad_revenue": st.column_config.NumberColumn("Ad Revenue Rank"),
            "rank_by_recreations": st.column_config.NumberColumn("Remix Rank"),
        },
    )


# =============================================================================
# AI Model Breakdown
# =============================================================================
with content_tab:
    st.markdown("### Generator Performance Comparison")
    st.caption("Evaluate how content created with Sora 2, Flux, Kling 2.0, and Seedance Pro engages audiences.")

    engagement = (
        filtered_events.groupby(["genre", "ai_model_used"])
        .agg(
            total_views=("event_id", "count"),
            avg_completion_rate=("completion_rate", "mean"),
            avg_recreation_rate=("recreated_by_user", "mean"),
            total_ad_revenue=("ad_revenue_usd", "sum"),
        )
        .reset_index()
        .sort_values(["genre", "total_views"], ascending=[True, False])
    )

    st.dataframe(
        engagement,
        hide_index=True,
        column_config={
            "genre": st.column_config.TextColumn("Content Genre", pinned=True),
            "ai_model_used": st.column_config.TextColumn("AI Model"),
            "total_views": st.column_config.NumberColumn("Total Views"),
            "avg_completion_rate": st.column_config.NumberColumn("Avg. Completion Rate", format="percent"),
            "avg_recreation_rate": st.column_config.NumberColumn("Avg. Remix Rate", format="percent"),
            "total_ad_revenue": st.column_config.NumberColumn("Ad Revenue ($)", format="$%.2f"),
        },
    )

    with st.container(border=True, key="card_content_chart"):
        st.markdown("### 📊 Total Views Generated by AI Engine & Genre")
        fig = px.bar(
            engagement,
            x="genre",
            y="total_views",
            color="ai_model_used",
            barmode="group",
            labels={"genre": "Genre", "total_views": "Total Views", "ai_model_used": "AI Model"},
            color_discrete_sequence=LIGHT_CHART_COLORS,
        )
        apply_light_chart_theme(fig)
        fig.update_layout(height=360, legend_title_text="AI Engine")
        st.plotly_chart(fig, theme=None)


# =============================================================================
# User Loyalty & Retention
# =============================================================================
with cohort_tab:
    st.markdown("### Audience Retention & Loyalty Analysis")
    st.caption("Tracking user drop-off over a 60-day window across different acquisition channels.")

    cohort_matrix, cohort_sizes = compute_cohort_matrix(users_df, filtered_events)

    if cohort_matrix.empty:
        st.info("No retention records match your current filter settings.", icon="ℹ️")
    else:
        st.markdown("### 🗺️ Where do users stay or drop off over 60 days?")
        fig = px.imshow(
            cohort_matrix,
            text_auto=".1f",
            color_continuous_scale=LIGHT_CHART_SEQUENTIAL,
            labels=dict(x="Retention Timeline", y="Acquisition Channel", color="Retention Rate (%)"),
            aspect="auto",
        )
        apply_light_chart_theme(fig)
        fig.update_traces(
            hovertemplate="Channel: <b>%{y}</b><br>%{x}: <b>%{z:.1f}%</b> retained<extra></extra>"
        )
        fig.update_layout(height=360)
        st.plotly_chart(fig, theme=None)
        st.caption(
            "📌 **Cohort Base Sizes:** "
            + " • ".join(f"**{ch.replace('_', ' ').title()}**: {n:,} users" for ch, n in cohort_sizes.items())
        )

    st.caption(
        "💡 **Key Finding:** *Paid Social* brings in quick initial traffic (strong Day 1 retention) but sees faster drop-offs long-term. In contrast, *Creator Referrals* demonstrate steady, long-term audience loyalty up to Day 60."
    )


# =============================================================================
# Virality Spotlight
# =============================================================================
with virality_tab:
    st.markdown("### Virality & Content Resonance Analysis")
    st.caption(
        "Evaluating video performance using a balanced weighting of completion rate, user shares, and remixes."
    )
    
    with st.expander("ℹ️ How is the Virality Score calculated?", expanded=False):
        st.write(
            """
            $$\\text{Virality Score} = \\left( \\frac{(\\text{Remixes} \\times 3) + (\\text{Shares} \\times 2) + \\text{Likes}}{\\text{Total Views}} \\right) \\times \\text{Completion Rate} \\times 100$$
            
            * **Remixes (3x weight):** Highest value action, driving derivative content creation.
            * **Shares (2x weight):** Expands reach beyond existing audience circles.
            * **Likes (1x weight):** Expresses immediate user appreciation.
            * **Completion Rate Multiplier:** Ensures lower-view videos with highly engaged audiences are fairly represented.
            """
        )

    video_stats_df = build_video_stats(filtered_events)
    tab_genres = sorted(video_stats_df["genre"].unique())
    tab_models = sorted(video_stats_df["ai_model_used"].unique())

    filter_row = st.container(horizontal=True)
    with filter_row:
        drill_genres = st.multiselect(
            "Filter Genre",
            options=tab_genres,
            default=tab_genres,
            key="virality_genre_filter",
            help="Narrow down videos within the current view scope.",
        )
        drill_models = st.multiselect(
            "Filter AI Model",
            options=tab_models,
            default=tab_models,
            key="virality_model_filter",
            help="Narrow down videos within the current view scope.",
        )

    if not drill_genres or not drill_models:
        st.info("Select at least one genre and AI model to display virality insights.", icon="ℹ️")
        filtered_stats = video_stats_df.iloc[0:0]
    else:
        filtered_stats = video_stats_df[
            video_stats_df["genre"].isin(drill_genres)
            & video_stats_df["ai_model_used"].isin(drill_models)
        ]

    with st.container(horizontal=True):
        st.metric("Analyzed Videos", f"{len(filtered_stats):,}")
        st.metric(
            "Average Virality",
            f"{filtered_stats['virality_score'].mean():.1f}" if len(filtered_stats) else "—",
        )
        st.metric(
            "Peak Virality Score",
            f"{filtered_stats['virality_score'].max():.1f}" if len(filtered_stats) else "—",
        )

    col_leaderboard, col_chart = st.columns([3, 2])

    with col_leaderboard:
        with st.container(border=True, key="card_virality_leaderboard"):
            st.markdown("### 🔥 Top Viral Videos")
            video_search = st.text_input(
                "Search Video or Creator ID",
                placeholder="Search (e.g. VID_0291 or CREATOR_083)...",
                key="virality_search",
            )
            leaderboard = filtered_stats.sort_values(["genre", "genre_virality_rank"])[
                [
                    "genre",
                    "genre_virality_rank",
                    "video_id",
                    "creator_id",
                    "ai_model_used",
                    "total_views",
                    "total_likes",
                    "total_shares",
                    "total_recreations",
                    "avg_completion_rate",
                    "virality_score",
                ]
            ]
            if video_search:
                mask = leaderboard["video_id"].str.contains(
                    video_search, case=False, na=False
                ) | leaderboard["creator_id"].str.contains(video_search, case=False, na=False)
                leaderboard = leaderboard[mask]

            st.caption(f"Showing **{len(leaderboard):,}** matching videos — click columns to re-sort.")
            st.dataframe(
                leaderboard,
                hide_index=True,
                column_config={
                    "genre": st.column_config.TextColumn("Genre"),
                    "genre_virality_rank": st.column_config.NumberColumn("Genre Rank"),
                    "video_id": st.column_config.TextColumn("Video ID", pinned=True),
                    "creator_id": st.column_config.TextColumn("Creator"),
                    "ai_model_used": st.column_config.TextColumn("AI Model"),
                    "total_views": st.column_config.NumberColumn("Views"),
                    "total_likes": st.column_config.NumberColumn("Likes"),
                    "total_shares": st.column_config.NumberColumn("Shares"),
                    "total_recreations": st.column_config.NumberColumn("Remixes"),
                    "avg_completion_rate": st.column_config.NumberColumn("Completion Rate", format="percent"),
                    "virality_score": st.column_config.NumberColumn("Virality Score", format="%.2f"),
                },
            )
            download_csv_button(leaderboard, "virality_leaderboard.csv", "download_virality_csv")

    with col_chart:
        with st.container(border=True, key="card_virality_chart"):
            st.markdown("### 🤖 Virality Score by AI Model")
            model_avg = (
                filtered_stats.groupby("ai_model_used")["virality_score"]
                .mean()
                .round(2)
                .reset_index()
                .sort_values("virality_score", ascending=False)
            )
            fig = px.bar(
                model_avg,
                x="virality_score",
                y="ai_model_used",
                color="ai_model_used",
                orientation="h",
                labels={"virality_score": "Average Virality Score", "ai_model_used": "AI Model"},
                text="virality_score",
                color_discrete_sequence=LIGHT_CHART_COLORS,
            )
            apply_light_chart_theme(fig)
            fig.update_traces(
                texttemplate="<b>%{text:.2f}</b>",
                hovertemplate="Model: <b>%{y}</b><br>Avg Score: <b>%{x:.2f}</b><extra></extra>",
            )
            fig.update_layout(
                height=320, yaxis={"categoryorder": "total ascending"}, showlegend=False
            )
            st.plotly_chart(fig, theme=None)


# =============================================================================
# Behavioral Experiments (A/B Test) - Human-Centered Refactor
# =============================================================================
@st.cache_data(ttl="1h", show_spinner="Evaluating statistical test results...")
def load_ab_test_results(users: pd.DataFrame, events: pd.DataFrame):
    return compute_ab_test_results(users=users, watch_events=events)


with abtest_tab:
    st.markdown("### Behavioral Experiment: Early Remixing vs. Retention")
    st.caption("Testing whether users who remix a video within their first 7 days demonstrate stronger 30-day retention.")

    ab_results = load_ab_test_results(users_df, filtered_events)

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        with st.container(border=True, key="card_ab_chart"):
            st.markdown("### 🧪 Day-30 Retention Rate Comparison")
            
            # Interactive Plotly chart with styled gradient contrast
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Passive Viewers", "Early Remixers"],
                y=[ab_results["passive_retention_rate"] * 100, ab_results["recreator_retention_rate"] * 100],
                text=[f"{ab_results['passive_retention_rate']:.1%}", f"{ab_results['recreator_retention_rate']:.1%}"],
                textposition='auto',
                marker=dict(
                    color=['#94A3B8', '#4F46E5'],
                    line=dict(color='#334155', width=1)
                ),
                hovertemplate="User Segment: <b>%{x}</b><br>Day 30 Retention: <b>%{y:.1f}%</b><extra></extra>"
            ))

            apply_light_chart_theme(fig)
            fig.update_layout(
                height=320, 
                yaxis=dict(title="Retained (%)", ticksuffix="%"),
                showlegend=False
            )
            st.plotly_chart(fig, theme=None)

    with col_table:
        with st.container(border=True, key="card_ab_table"):
            st.markdown("### 📊 User Count Breakdown")
            contingency_display = ab_results["contingency"].rename(
                index={"Early Recreator": "Early Remixers"}
            )
            st.dataframe(contingency_display)
            st.caption(
                f"• **Early Remixers:** {ab_results['n_recreators']:,} users\n\n"
                f"• **Passive Viewers:** {ab_results['n_passive']:,} users"
            )

    st.markdown("---")
    st.markdown("### 🎯 How sure do you want to be about this result?")
    
    confidence_choice = st.radio(
        "Confidence Level Threshold",
        options=list(CONFIDENCE_LEVEL_OPTIONS.keys()),
        index=1,
        horizontal=True,
        key="ab_confidence_level",
        help="Higher confidence requires stronger mathematical proof before declaring a real behavioral difference."
    )
    confidence_level = CONFIDENCE_LEVEL_OPTIONS[confidence_choice]
    verdict = significance_verdict(ab_results["p_value"], confidence_level)

    # Convert scientific notation (e.g. 6.89e-08) into plain human language
    raw_p = ab_results["p_value"]
    if raw_p < 0.001:
        p_value_display = "< 0.001 (Overwhelming Proof)"
    elif raw_p < 0.01:
        p_value_display = f"{raw_p:.3f} (Very Strong Proof)"
    else:
        p_value_display = f"{raw_p:.3f}"

    with st.container(horizontal=True):
        st.metric(
            label="Experiment Outcome",
            value="🎉 Confirmed Difference" if verdict["is_significant"] else "⏳ Inconclusive",
            help="Tells you whether the gap in retention is real or just random luck."
        )
        st.metric(
            label="Statistical Odds (p-value)",
            value=p_value_display,
            help="Measures the chance that this difference happened by accident. Numbers below 0.05 mean high confidence."
        )
        st.metric(
            label="Target Threshold",
            value=f"{verdict['alpha']} ({confidence_choice})",
            help="The maximum risk of false alarm allowed for this test."
        )
        st.metric(
            label="Retention Boost",
            value=f"+{ab_results['lift_pct_points']:.1f}% Points",
            delta=f"{ab_results['relative_lift_pct']:+.1f}% overall lift",
            help="How much higher the Day-30 retention rate is for Early Remixers compared to Passive Viewers."
        )

    st.write("")

    if verdict["is_significant"]:
        st.success(
            f"🚀 **Key Takeaway:** Early remixing works! Users who create a remix during their first 7 days are **{ab_results['relative_lift_pct']:.1f}% more likely** to stick around after 30 days. We recommend testing remix prompts in your user onboarding flow.",
            icon="✅",
        )
    else:
        st.warning(
            "💡 **Key Takeaway:** Under these filter settings, there isn't enough proof to confirm that early remixing keeps users around longer. Try broadening sidebar filters to evaluate a larger user segment.",
            icon="⚠️",
        )

    st.info(
        f"**How to read this:** At a **{confidence_choice}** setting, any p-value below **{verdict['alpha']}** proves the difference isn't random coincidence. "
        "*Keep in mind: this tracks real user habits (correlation), not a forced test where users were randomly required to remix (causation).*",
        icon="🧠"
    )

    # -------------------------------------------------------------------------
    # Interactive ROI Calculator (Bonus Engagement Feature)
    # -------------------------------------------------------------------------
    st.write("")
    with st.expander("🧮 Estimate Business Impact (Interactive ROI Calculator)", expanded=False):
        st.markdown("##### If we nudge more new users to remix in Week 1, what happens to revenue?")
        
        col_a, col_b = st.columns(2)
        with col_a:
            new_monthly_users = st.slider("Monthly New User Signups", 1000, 100000, 25000, step=1000, key="roi_signups")
            current_remix_rate = st.slider("Current % of Users Who Remix in Week 1", 5, 50, 15, key="roi_curr_rate")
            target_remix_rate = st.slider("Target % of Users Who Remix in Week 1", current_remix_rate, 80, 30, key="roi_target_rate")
        
        # Projection math based on retention lift. Clamped to 0: if the target
        # slider's stored value is left below a since-raised current-rate
        # slider (Streamlit doesn't retroactively re-clamp widget state when
        # min_value changes), this keeps the projection from going negative.
        additional_remixers = max(0.0, new_monthly_users * ((target_remix_rate - current_remix_rate) / 100))
        retained_users_gained = int(additional_remixers * (ab_results["lift_pct_points"] / 100))
        est_annual_value = retained_users_gained * 12 * 5.00  # assuming $5 estimated annual LTV per retained user

        with col_b:
            st.metric("Gained Retained Users (Monthly)", f"+{retained_users_gained:,} users")
            st.metric("Projected Annual Revenue Lift", f"${est_annual_value:,.2f}", delta="Estimated ARR Impact")
            if target_remix_rate < current_remix_rate:
                st.caption(
                    "Set the target rate above the current rate to see a positive projection."
                )


# =============================================================================
# Forecast Simulator
# =============================================================================
with simulator_tab:
    st.markdown("### Video Performance & Monetization Forecast")
    st.caption(
        "Model potential outcomes for a video concept to estimate virality and earnings using platform baseline metrics."
    )

    input_col, result_col = st.columns([2, 3])

    with input_col:
        with st.container(border=True, key="card_sim_inputs"):
            st.markdown("### 🎛️ Video Parameters")
            sim_video_length = st.slider("Video Duration (seconds)", 15, 300, 120, key="sim_length")
            sim_model = st.selectbox("AI Generator Model", options=AI_MODELS, key="sim_model")
            sim_genre = st.selectbox(
                "Target Genre",
                options=["All Genres"] + GENRES,
                key="sim_genre",
                help="Compare expected performance against existing videos in this genre.",
            )
            sim_monetized = st.toggle("Enable Ad Monetization", value=True, key="sim_monetized")
            
            st.markdown("---")
            st.markdown("### 👥 Expected Audience Engagement")
            sim_views = st.slider("Projected View Count", 10, 500, 50, key="sim_views")
            sim_completion_pct = st.slider("Expected Completion Rate (%)", 0, 100, 55, key="sim_completion")
            sim_like_pct = st.slider("Expected Like Rate (% of Viewers)", 0, 100, 30, key="sim_like_rate")
            sim_share_pct = st.slider("Expected Share Rate (% of Viewers)", 0, 100, 10, key="sim_share_rate")
            sim_recreate_pct = st.slider("Expected Remix Rate (% of Viewers)", 0, 100, 15, key="sim_recreate_rate")

    sim_likes = sim_views * sim_like_pct / 100
    sim_shares = sim_views * sim_share_pct / 100
    sim_recreations = sim_views * sim_recreate_pct / 100
    sim_completion_rate = sim_completion_pct / 100

    sim_virality_score = (
        ((sim_recreations * 3.0 + sim_shares * 2.0 + sim_likes) / sim_views)
        * sim_completion_rate
        * 100
        if sim_views > 0
        else 0.0
    )
    sim_watch_time_per_view = sim_completion_rate * sim_video_length
    sim_ad_revenue = (
        sim_views * sim_watch_time_per_view * BASE_CPM_USD_PER_SEC if sim_monetized else 0.0
    )
    sim_creator_credits = sim_recreations * AVG_CREDIT_PER_RECREATION_USD

    peer_scores = video_stats_df[video_stats_df["ai_model_used"] == sim_model]
    if sim_genre != "All Genres":
        peer_scores = peer_scores[peer_scores["genre"] == sim_genre]
    percentile = (
        (peer_scores["virality_score"] < sim_virality_score).mean() * 100
        if len(peer_scores)
        else float("nan")
    )

    with result_col:
        with st.container(horizontal=True):
            st.metric("Forecasted Virality", f"{sim_virality_score:.1f}")
            st.metric("Estimated Ad Revenue", f"${sim_ad_revenue:,.2f}")
            st.metric("Estimated Creator Payout", f"${sim_creator_credits:,.2f}")

        with st.container(border=True, key="card_sim_results"):
            model_label = sim_model if sim_genre == "All Genres" else f"{sim_model} ({sim_genre})"
            st.markdown(f"### 📊 Benchmark Comparison against {model_label} Videos")
            if len(peer_scores) >= 5:
                fig = px.histogram(
                    peer_scores,
                    x="virality_score",
                    nbins=20,
                    labels={"virality_score": "Virality Score"},
                    color_discrete_sequence=LIGHT_CHART_COLORS,
                )
                apply_light_chart_theme(fig)
                fig.add_vline(
                    x=sim_virality_score,
                    line_width=3,
                    line_dash="dash",
                    line_color="#DB2777",
                    annotation_text="Your Video Projection",
                    annotation_position="top left",
                )
                fig.update_layout(height=340, showlegend=False)
                st.plotly_chart(fig, theme=None)
                st.caption(
                    f"🌟 This scenario scores higher than **{percentile:.0f}%** of the "
                    f"**{len(peer_scores):,}** existing **{model_label}** videos matching your active filters."
                )
            else:
                st.info(
                    "Insufficient benchmark data available for this combination. Try broadening your sidebar filters or selecting a different model/genre.",
                    icon="ℹ️",
                )

    st.caption(
        f"💡 Estimates are modeled using platform baseline rates: **${BASE_CPM_USD_PER_SEC}/sec** ad rate and **~${AVG_CREDIT_PER_RECREATION_USD}** average remix payout credit."
    )