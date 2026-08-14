-- ============================================================================
-- StreaMetrics | Streaming Analytics — Advanced Analytics Queries
-- Target: SQLite (data/streaming_analytics.db)
--
-- Run with:
--   sqlite3 data/streaming_analytics.db < sql/02_analytics_queries.sql
-- ============================================================================

.headers on
.mode column


-- ============================================================================
-- QUERY 1: CREATOR LEADERBOARD
-- Ranks creators using DENSE_RANK() on total watch time, total ad revenue,
-- and recreation count — surfaces who is actually driving engagement, revenue,
-- and viral remix behavior (not always the same creators).
-- ============================================================================
WITH creator_stats AS (
    SELECT
        v.creator_id,
        COUNT(DISTINCT v.video_id)                 AS video_count,
        SUM(we.watch_time_sec) / 3600.0             AS total_watch_hours,
        SUM(we.ad_revenue_usd)                      AS total_ad_revenue,
        SUM(we.recreated_by_user)                   AS total_recreations,
        SUM(we.creator_credit_earned)               AS total_credits_earned,
        ROUND(AVG(we.completion_rate), 4)           AS avg_completion_rate
    FROM videos v
    JOIN watch_events we ON we.video_id = v.video_id
    GROUP BY v.creator_id
)
SELECT
    creator_id,
    video_count,
    ROUND(total_watch_hours, 2)      AS total_watch_hours,
    ROUND(total_ad_revenue, 2)       AS total_ad_revenue,
    total_recreations,
    ROUND(total_credits_earned, 2)   AS total_credits_earned,
    avg_completion_rate,
    DENSE_RANK() OVER (ORDER BY total_watch_hours DESC)   AS rank_by_watch_time,
    DENSE_RANK() OVER (ORDER BY total_ad_revenue DESC)    AS rank_by_ad_revenue,
    DENSE_RANK() OVER (ORDER BY total_recreations DESC)   AS rank_by_recreations
FROM creator_stats
ORDER BY total_watch_hours DESC
LIMIT 20;


-- ============================================================================
-- QUERY 2: CONTENT & AI MODEL ENGAGEMENT
-- Aggregates total views, average completion rate, and average CTR
-- (approximated as recreation rate — the share of views that convert into a
-- recreate/remix action) grouped by genre and ai_model_used.
-- ============================================================================
SELECT
    v.genre,
    v.ai_model_used,
    COUNT(we.event_id)                                        AS total_views,
    ROUND(AVG(we.completion_rate), 4)                          AS avg_completion_rate,
    ROUND(AVG(we.recreated_by_user) * 100, 2)                  AS avg_recreation_ctr_pct,
    ROUND(SUM(we.ad_revenue_usd), 2)                           AS total_ad_revenue,
    ROUND(AVG(v.ai_generated_ratio), 3)                        AS avg_ai_generated_ratio
FROM videos v
JOIN watch_events we ON we.video_id = v.video_id
GROUP BY v.genre, v.ai_model_used
ORDER BY v.genre, total_views DESC;


-- ============================================================================
-- QUERY 3: MONTH-OVER-MONTH (MoM) GROWTH
-- Uses LAG() over a monthly-aggregated CTE to compute MoM watch-hour growth
-- and MoM monetization (ad revenue) growth.
-- ============================================================================
WITH monthly_stats AS (
    SELECT
        strftime('%Y-%m', we.event_timestamp)         AS activity_month,
        SUM(we.watch_time_sec) / 3600.0                AS watch_hours,
        SUM(we.ad_revenue_usd)                         AS ad_revenue,
        COUNT(DISTINCT we.user_id)                     AS active_users
    FROM watch_events we
    GROUP BY activity_month
),
monthly_growth AS (
    SELECT
        activity_month,
        ROUND(watch_hours, 2)   AS watch_hours,
        ROUND(ad_revenue, 2)    AS ad_revenue,
        active_users,
        LAG(watch_hours) OVER (ORDER BY activity_month)  AS prev_watch_hours,
        LAG(ad_revenue) OVER (ORDER BY activity_month)   AS prev_ad_revenue
    FROM monthly_stats
)
SELECT
    activity_month,
    watch_hours,
    ad_revenue,
    active_users,
    CASE
        WHEN prev_watch_hours IS NULL OR prev_watch_hours = 0 THEN NULL
        ELSE ROUND((watch_hours - prev_watch_hours) / prev_watch_hours * 100, 2)
    END AS mom_watch_hours_growth_pct,
    CASE
        WHEN prev_ad_revenue IS NULL OR prev_ad_revenue = 0 THEN NULL
        ELSE ROUND((ad_revenue - prev_ad_revenue) / prev_ad_revenue * 100, 2)
    END AS mom_revenue_growth_pct
FROM monthly_growth
ORDER BY activity_month;


-- ============================================================================
-- QUERY 4: VIRALITY & DECAY SCORE - PER-VIDEO LEADERBOARD (Module 1)
-- Custom engagement score blending remix depth (recreations), share reach,
-- and lightweight approval (likes), scaled by how much of the video people
-- actually watched:
--
--   Virality Score = (((Recreations * 3.0) + (Shares * 2.0) + Likes)
--                       / Total Views) * Avg Completion Rate * 100
--
-- Recreations are weighted highest (3x) because a remix is the deepest form
-- of engagement StreaMetrics can capture - it creates new monetizable content and
-- directly compounds the platform's content graph. Shares (2x) extend reach
-- without creating new content; likes (1x) are the lowest-effort signal.
-- Dividing by total_views normalizes for a video's raw popularity (a video
-- with 500 lukewarm views shouldn't outrank one with 50 fiercely engaged
-- views), and multiplying by avg_completion_rate penalizes clickbait content
-- that gets engagement actions without being genuinely watched.
--
-- DENSE_RANK() OVER (PARTITION BY genre ORDER BY virality_score DESC) ranks
-- each video within its own genre (so Horror isn't compared to Documentaries
-- on the same absolute scale), surfacing the top 5 per genre.
-- ============================================================================
WITH video_stats AS (
    SELECT
        v.video_id,
        v.creator_id,
        v.genre,
        v.ai_model_used,
        COUNT(we.event_id)              AS total_views,
        SUM(we.recreated_by_user)       AS total_recreations,
        SUM(we.shared_by_user)          AS total_shares,
        SUM(we.liked_by_user)           AS total_likes,
        AVG(we.completion_rate)         AS avg_completion_rate
    FROM videos v
    JOIN watch_events we ON we.video_id = v.video_id
    GROUP BY v.video_id
),
virality_scores AS (
    SELECT
        video_id,
        creator_id,
        genre,
        ai_model_used,
        total_views,
        total_likes,
        total_shares,
        total_recreations,
        ROUND(avg_completion_rate, 4) AS avg_completion_rate,
        ROUND(
            ((total_recreations * 3.0 + total_shares * 2.0 + total_likes) / total_views)
            * avg_completion_rate * 100,
        2) AS virality_score
    FROM video_stats
),
ranked_scores AS (
    SELECT
        *,
        DENSE_RANK() OVER (PARTITION BY genre ORDER BY virality_score DESC) AS genre_virality_rank
    FROM virality_scores
)
SELECT
    genre,
    genre_virality_rank,
    video_id,
    creator_id,
    ai_model_used,
    total_views,
    total_likes,
    total_shares,
    total_recreations,
    avg_completion_rate,
    virality_score
FROM ranked_scores
WHERE genre_virality_rank <= 5
ORDER BY genre, genre_virality_rank;


-- ============================================================================
-- QUERY 5: VIRALITY SCORE BY AI MODEL (Module 1)
-- Aggregates per-video Virality Scores by ai_model_used to identify which
-- generation model produces the highest-performing content on average, with
-- DENSE_RANK() over the aggregated averages to rank the four models.
-- ============================================================================
WITH video_stats AS (
    SELECT
        v.video_id,
        v.ai_model_used,
        COUNT(we.event_id)              AS total_views,
        SUM(we.recreated_by_user)       AS total_recreations,
        SUM(we.shared_by_user)          AS total_shares,
        SUM(we.liked_by_user)           AS total_likes,
        AVG(we.completion_rate)         AS avg_completion_rate
    FROM videos v
    JOIN watch_events we ON we.video_id = v.video_id
    GROUP BY v.video_id
),
virality_scores AS (
    SELECT
        ai_model_used,
        ((total_recreations * 3.0 + total_shares * 2.0 + total_likes) / total_views)
            * avg_completion_rate * 100 AS virality_score
    FROM video_stats
)
SELECT
    ai_model_used,
    COUNT(*)                                AS video_count,
    ROUND(AVG(virality_score), 2)           AS avg_virality_score,
    ROUND(MAX(virality_score), 2)           AS max_virality_score,
    ROUND(MIN(virality_score), 2)           AS min_virality_score,
    DENSE_RANK() OVER (ORDER BY AVG(virality_score) DESC) AS model_rank
FROM virality_scores
GROUP BY ai_model_used
ORDER BY avg_virality_score DESC;
