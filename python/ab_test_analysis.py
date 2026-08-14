"""
StreaMetrics - Module 2: A/B Testing & Hypothesis Testing

Chi-Square test of independence evaluating whether recreating/remixing an AI
video within the first 7 days after signup ("early recreation") is
associated with higher Day 30 retention, compared to users who did not
recreate in that window ("passive viewers").

Hypotheses
----------
H0 (null):        Day 30 retention is independent of early-recreation status.
H1 (alternative): Day 30 retention is associated with early-recreation status.

Group definitions
------------------
- Early Recreator : >= 1 watch_event with recreated_by_user == 1 where
                     days_since_signup is in [0, 7].
- Passive Viewer   : everyone else (did not recreate within the first 7 days
                     - includes users who watched without recreating and
                     users with no activity at all in that window).

Day 30 retention uses the same windowed definition as cohort_analysis.py
(>= 1 watch_event with days_since_signup in [27, 33]) for methodological
consistency with the Module 0 cohort analysis.

Caveat: this is an observational behavioral-cohort comparison, not a
randomized controlled trial. Users who choose to recreate early are
plausibly more engaged in ways that also independently drive retention
(selection bias) - the test establishes association, not proven causation.
See README.md for the full write-up.
"""

import os

import pandas as pd
from scipy.stats import chi2_contingency

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

EARLY_WINDOW_DAYS = 7
RETENTION_DAY = 30
RETENTION_TOLERANCE = 3  # matches cohort_analysis.py's Day 30 window: [27, 33]

CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]


def compute_ab_test_results(
    data_dir: str = DATA_DIR,
    users: pd.DataFrame | None = None,
    watch_events: pd.DataFrame | None = None,
) -> dict:
    """Compute the full Chi-Square A/B test result set.

    Loads `users.csv` / `watch_events.csv` from `data_dir` when `users` /
    `watch_events` aren't supplied (the standalone script path). The
    Streamlit app instead passes in its already-loaded, globally-filtered
    DataFrames so the A/B test respects the sidebar filters without a
    second disk read.
    """
    if users is None:
        users = pd.read_csv(os.path.join(data_dir, "users.csv"), parse_dates=["signup_date"])
    if watch_events is None:
        watch_events = pd.read_csv(
            os.path.join(data_dir, "watch_events.csv"), parse_dates=["event_timestamp"]
        )

    events = watch_events.merge(users[["user_id", "signup_date"]], on="user_id", how="left")
    events["days_since_signup"] = (
        events["event_timestamp"].dt.normalize() - events["signup_date"]
    ).dt.days

    early_recreator_ids = set(
        events.loc[
            (events["recreated_by_user"] == 1) & (events["days_since_signup"] <= EARLY_WINDOW_DAYS),
            "user_id",
        ].unique()
    )

    retained_mask = (events["days_since_signup"] >= RETENTION_DAY - RETENTION_TOLERANCE) & (
        events["days_since_signup"] <= RETENTION_DAY + RETENTION_TOLERANCE
    )
    retained_ids = set(events.loc[retained_mask, "user_id"].unique())

    cohort = pd.DataFrame({"user_id": users["user_id"]})
    cohort["is_early_recreator"] = cohort["user_id"].isin(early_recreator_ids)
    cohort["is_retained_day30"] = cohort["user_id"].isin(retained_ids)

    contingency = pd.crosstab(
        cohort["is_early_recreator"].map({True: "Early Recreator", False: "Passive Viewer"}),
        cohort["is_retained_day30"].map({True: "Retained (Day 30)", False: "Not Retained"}),
    ).reindex(
        index=["Early Recreator", "Passive Viewer"],
        columns=["Retained (Day 30)", "Not Retained"],
    )

    chi2_stat, p_value, dof, expected = chi2_contingency(contingency.values, correction=True)

    n_recreators = int(contingency.loc["Early Recreator"].sum())
    n_passive = int(contingency.loc["Passive Viewer"].sum())
    recreator_retention_rate = contingency.loc["Early Recreator", "Retained (Day 30)"] / n_recreators
    passive_retention_rate = contingency.loc["Passive Viewer", "Retained (Day 30)"] / n_passive
    lift_pct_points = (recreator_retention_rate - passive_retention_rate) * 100
    relative_lift_pct = (
        (recreator_retention_rate - passive_retention_rate) / passive_retention_rate * 100
        if passive_retention_rate > 0
        else float("nan")
    )

    return {
        "contingency": contingency,
        "expected": expected,
        "chi2_stat": float(chi2_stat),
        "p_value": float(p_value),
        "dof": int(dof),
        "n_recreators": n_recreators,
        "n_passive": n_passive,
        "recreator_retention_rate": recreator_retention_rate,
        "passive_retention_rate": passive_retention_rate,
        "lift_pct_points": lift_pct_points,
        "relative_lift_pct": relative_lift_pct,
    }


def format_p_value(p_value: float) -> str:
    """Fixed-point for readable p-values, scientific notation once it would round to 0."""
    return f"{p_value:.6f}" if p_value >= 0.0001 else f"{p_value:.4e}"


def significance_verdict(p_value: float, confidence_level: float) -> dict:
    """Statistical conclusion (reject / fail to reject H0) at a given confidence level."""
    alpha = round(1 - confidence_level, 2)
    return {
        "confidence_level": confidence_level,
        "alpha": alpha,
        "is_significant": p_value < alpha,
    }


def format_recommendation(results: dict, confidence_level: float) -> str:
    """Plain-language statistical conclusion + business recommendation for a confidence level."""
    verdict = significance_verdict(results["p_value"], confidence_level)
    conf_pct = int(confidence_level * 100)
    recreator_pct = results["recreator_retention_rate"] * 100
    passive_pct = results["passive_retention_rate"] * 100

    if verdict["is_significant"] and results["lift_pct_points"] > 0:
        return (
            f"Reject H0 at {conf_pct}% confidence (p = {format_p_value(results['p_value'])} < "
            f"alpha = {verdict['alpha']}). Early recreation is significantly associated with "
            f"higher Day 30 retention ({recreator_pct:.1f}% vs. {passive_pct:.1f}%, a "
            f"{results['lift_pct_points']:+.1f}-point lift). Recommendation: treat 'recreated "
            f"within 7 days' as a leading activation metric - invest in onboarding nudges "
            f"(remix prompts, template challenges, creator-credit teasers) that drive first-week "
            f"recreation, since it is statistically linked to a more durable user base."
        )
    elif verdict["is_significant"]:
        return (
            f"Reject H0 at {conf_pct}% confidence (p = {format_p_value(results['p_value'])} < "
            f"alpha = {verdict['alpha']}), but the direction does not favor recreators "
            f"({recreator_pct:.1f}% vs. {passive_pct:.1f}%). Investigate before acting on this "
            f"result - the association is real but does not support incentivizing recreation."
        )
    else:
        return (
            f"Fail to reject H0 at {conf_pct}% confidence (p = {format_p_value(results['p_value'])} >= "
            f"alpha = {verdict['alpha']}). The observed retention difference "
            f"({recreator_pct:.1f}% vs. {passive_pct:.1f}%) is not statistically significant at "
            f"this confidence level - do not treat early recreation as a proven retention driver "
            f"without more data."
        )


def main():
    results = compute_ab_test_results()

    print("=" * 90)
    print("MODULE 2: A/B TEST - EARLY RECREATION vs. DAY 30 RETENTION (Chi-Square Test)")
    print("=" * 90)
    print()
    print("H0: Day 30 retention is independent of early-recreation status (first 7 days).")
    print("H1: Day 30 retention is associated with early-recreation status.")
    print()
    print("Contingency table (observed):")
    print(results["contingency"].to_string())
    print()
    print(
        f"Group sizes: Early Recreators = {results['n_recreators']:,}  |  "
        f"Passive Viewers = {results['n_passive']:,}"
    )
    print()
    print(f"Day 30 retention rate - Early Recreators: {results['recreator_retention_rate']*100:.2f}%")
    print(f"Day 30 retention rate - Passive Viewers:  {results['passive_retention_rate']*100:.2f}%")
    print(
        f"Lift: {results['lift_pct_points']:+.2f} points "
        f"({results['relative_lift_pct']:+.1f}% relative)"
    )
    print()
    print(f"Chi-Square statistic: {results['chi2_stat']:.4f}")
    print(f"Degrees of freedom:   {results['dof']}")
    print(f"p-value:              {format_p_value(results['p_value'])}")
    print("(Yates' continuity correction applied, standard for 2x2 contingency tables)")
    print()
    print("-" * 90)
    print("CONCLUSIONS AT EACH CONFIDENCE LEVEL")
    print("-" * 90)
    for level in CONFIDENCE_LEVELS:
        print(f"\n[{int(level*100)}% confidence]")
        print(format_recommendation(results, level))
    print()


if __name__ == "__main__":
    main()
