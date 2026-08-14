"""
StreaMetrics - Synthetic Data Generator
Generates 3 realistic relational CSV datasets for product & growth analytics:
  - data/users.csv
  - data/videos.csv
  - data/watch_events.csv

Seed is fixed at 42 for full reproducibility.
"""

import os
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. USERS
# ----------------------------------------------------------------------------
N_USERS = 2000

# Signup dates spread across 2025-01-01 through 2026-06-30 (weighted toward growth over time)
signup_start = pd.Timestamp("2025-01-01")
signup_end = pd.Timestamp("2026-06-30")
signup_range_days = (signup_end - signup_start).days

# Weight signups so later periods have more volume (platform growth curve)
day_offsets = np.arange(signup_range_days + 1)
growth_weights = np.linspace(0.3, 1.0, len(day_offsets)) ** 2
growth_weights = growth_weights / growth_weights.sum()
chosen_offsets = rng.choice(day_offsets, size=N_USERS, p=growth_weights)
signup_dates = signup_start + pd.to_timedelta(chosen_offsets, unit="D")

acquisition_channels = ["Organic", "Creator_Referral", "Paid_Social", "Search_Ads"]
acquisition_probs = [0.35, 0.25, 0.25, 0.15]
acquisition_channel = rng.choice(acquisition_channels, size=N_USERS, p=acquisition_probs)

subscription_plans = ["Free", "Pro", "Creator_Tier"]
subscription_probs = [0.65, 0.25, 0.10]
subscription_plan = rng.choice(subscription_plans, size=N_USERS, p=subscription_probs)

user_ids = [f"USR_{i:04d}" for i in range(1, N_USERS + 1)]

users_df = pd.DataFrame({
    "user_id": user_ids,
    "signup_date": signup_dates.strftime("%Y-%m-%d"),
    "acquisition_channel": acquisition_channel,
    "subscription_plan": subscription_plan,
})

users_df.to_csv(os.path.join(DATA_DIR, "users.csv"), index=False)

# ----------------------------------------------------------------------------
# 2. VIDEOS
# ----------------------------------------------------------------------------
N_VIDEOS = 600
N_CREATORS = 120

creator_ids_pool = [f"CREATOR_{i:03d}" for i in range(1, N_CREATORS + 1)]
video_creator_id = rng.choice(creator_ids_pool, size=N_VIDEOS)

genres = ["Sci-Fi", "AI Animation", "AI Shorts", "Drama", "Horror", "Documentaries"]
genre_probs = [0.20, 0.25, 0.20, 0.15, 0.10, 0.10]
video_genre = rng.choice(genres, size=N_VIDEOS, p=genre_probs)

video_duration_sec = rng.integers(15, 301, size=N_VIDEOS)

ai_models = ["Kling 2.0", "Sora 2", "Flux", "Seedance Pro"]
ai_model_probs = [0.30, 0.30, 0.20, 0.20]
ai_model_used = rng.choice(ai_models, size=N_VIDEOS, p=ai_model_probs)

ai_generated_ratio = np.round(rng.uniform(0.60, 1.0, size=N_VIDEOS), 3)

monetized = rng.choice([0, 1], size=N_VIDEOS, p=[0.35, 0.65])

video_ids = [f"VID_{i:04d}" for i in range(1, N_VIDEOS + 1)]

videos_df = pd.DataFrame({
    "video_id": video_ids,
    "creator_id": video_creator_id,
    "genre": video_genre,
    "video_duration_sec": video_duration_sec,
    "ai_model_used": ai_model_used,
    "ai_generated_ratio": ai_generated_ratio,
    "monetized": monetized,
})

videos_df.to_csv(os.path.join(DATA_DIR, "videos.csv"), index=False)

# ----------------------------------------------------------------------------
# 3. WATCH EVENTS
# ----------------------------------------------------------------------------
N_EVENTS = 15000

users_lookup = users_df.set_index("user_id")
videos_lookup = videos_df.set_index("video_id")

# --- Channel-driven engagement decay -----------------------------------------
# Each acquisition channel has a characteristic "engagement half-life": the
# average number of days a user keeps generating watch events after signup.
# Creator referrals arrive pre-qualified by trust in a specific creator and
# stick around longest; paid social traffic is lowest-intent and drops off
# fastest. Sampling each event's days-since-signup from an exponential
# distribution keyed on this scale produces realistic, decaying, channel-
# differentiated retention curves for the Step 3 cohort analysis (as opposed
# to sampling event timing uniformly at random, which yields no retention
# decay at all).
CHANNEL_ENGAGEMENT_SCALE_DAYS = {
    "Creator_Referral": 30.0,
    "Organic": 20.0,
    "Search_Ads": 15.0,
    "Paid_Social": 10.0,
}

user_channel_map = users_df.set_index("user_id")["acquisition_channel"]

# Allocate exactly N_EVENTS events across users, weighted by a per-user
# "activity level" (most users are casual, some are power users).
activity_weight = rng.gamma(shape=2.0, scale=1.0, size=N_USERS)
activity_weight = activity_weight / activity_weight.sum()
events_per_user = rng.multinomial(N_EVENTS, activity_weight)

event_user_id_chunks = []
days_since_signup_chunks = []
for idx, user_id in enumerate(user_ids):
    n = events_per_user[idx]
    if n == 0:
        continue
    scale = CHANNEL_ENGAGEMENT_SCALE_DAYS[user_channel_map.loc[user_id]]
    days = np.clip(np.floor(rng.exponential(scale=scale, size=n)), 0, 90).astype(int)
    event_user_id_chunks.append(np.full(n, user_id))
    days_since_signup_chunks.append(days)

event_user_id = np.concatenate(event_user_id_chunks)
days_since_signup = np.concatenate(days_since_signup_chunks)

# Shuffle so events aren't grouped by user in file order
shuffle_idx = rng.permutation(len(event_user_id))
event_user_id = event_user_id[shuffle_idx]
days_since_signup = days_since_signup[shuffle_idx]

event_video_id = rng.choice(video_ids, size=len(event_user_id))

# Event timestamp: signup_date + days_since_signup (channel-decayed) + time of day
user_signup_ts = pd.to_datetime(users_lookup.loc[event_user_id, "signup_date"]).values
seconds_of_day = rng.integers(0, 86400, size=len(event_user_id))
event_timestamp = (
    pd.to_datetime(user_signup_ts)
    + pd.to_timedelta(days_since_signup, unit="D")
    + pd.to_timedelta(seconds_of_day, unit="s")
)

N_EVENTS = len(event_user_id)  # exact count after multinomial allocation (== 15000)

video_duration_for_event = videos_lookup.loc[event_video_id, "video_duration_sec"].to_numpy()

# Watch time: beta-distributed completion behavior (many finish early, some binge to completion)
completion_fraction = rng.beta(a=2.0, b=1.8, size=N_EVENTS)
watch_time_sec = np.minimum(
    video_duration_for_event,
    np.round(completion_fraction * video_duration_for_event).astype(int),
)
watch_time_sec = np.maximum(watch_time_sec, 1)

completion_rate = np.round(watch_time_sec / video_duration_for_event, 4)

# Recreation likelihood increases with completion rate
recreate_prob = np.clip(0.03 + 0.20 * completion_rate, 0, 0.35)
recreated_by_user = (rng.uniform(size=N_EVENTS) < recreate_prob).astype(int)

# Likes and shares: lighter-weight engagement signals than a full recreation.
# Both scale with completion rate (independently of recreation) so the
# Module 1 Virality Score has real signal to differentiate content on.
like_prob = np.clip(0.15 + 0.35 * completion_rate, 0, 0.60)
liked_by_user = (rng.uniform(size=N_EVENTS) < like_prob).astype(int)

share_prob = np.clip(0.05 + 0.15 * completion_rate, 0, 0.25)
shared_by_user = (rng.uniform(size=N_EVENTS) < share_prob).astype(int)

# Ad revenue: driven by watch time (CPM-style), only for monetized videos
video_monetized_for_event = videos_lookup.loc[event_video_id, "monetized"].to_numpy()
base_cpm = 0.012  # USD per second watched, roughly
ad_revenue_noise = rng.normal(1.0, 0.15, size=N_EVENTS)
ad_revenue_usd = np.where(
    video_monetized_for_event == 1,
    np.round(np.maximum(watch_time_sec * base_cpm * ad_revenue_noise, 0), 4),
    0.0,
)

# Creator credit earned: only when the clip was recreated
creator_credit_earned = np.where(
    recreated_by_user == 1,
    np.round(rng.uniform(0.50, 5.00, size=N_EVENTS), 2),
    0.0,
)

event_ids = [f"EVT_{i:06d}" for i in range(1, N_EVENTS + 1)]

watch_events_df = pd.DataFrame({
    "event_id": event_ids,
    "user_id": event_user_id,
    "video_id": event_video_id,
    "event_timestamp": event_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
    "watch_time_sec": watch_time_sec,
    "completion_rate": completion_rate,
    "recreated_by_user": recreated_by_user,
    "liked_by_user": liked_by_user,
    "shared_by_user": shared_by_user,
    "ad_revenue_usd": ad_revenue_usd,
    "creator_credit_earned": creator_credit_earned,
})

watch_events_df.to_csv(os.path.join(DATA_DIR, "watch_events.csv"), index=False)

# ----------------------------------------------------------------------------
print("Data generation complete.")
print(f"  users.csv         -> {len(users_df):,} rows")
print(f"  videos.csv        -> {len(videos_df):,} rows")
print(f"  watch_events.csv  -> {len(watch_events_df):,} rows")
