CREATE OR REPLACE VIEW `engagement_logistic_baseline_input`
AS
SELECT
    dataset_split,
    label_will_miss_goal_next_7_days
        AS will_miss_goal_next_7_days,

    age_band_as_of,
    membership_months_as_of,
    activity_level_as_of,
    reward_profile_as_of,

    weekly_goal_as_of,
    weekly_active_minutes_so_far_as_of,
    goal_completion_percentage_as_of,
    previous_goal_streak_as_of,
    previous_failed_goals_as_of,
    days_since_last_app_session_as_of,

    available_day_count_28d,
    unavailable_day_count_28d,

    avg_daily_steps_28d,
    stddev_daily_steps_28d,
    avg_daily_steps_7d,
    avg_daily_steps_prior_7d,
    daily_steps_trend_7d,

    sum_active_minutes_28d,
    avg_active_minutes_28d,
    stddev_active_minutes_28d,
    avg_active_minutes_7d,
    avg_active_minutes_prior_7d,
    active_minutes_trend_7d,
    active_day_count_28d,

    avg_sleep_hours_28d,
    avg_sleep_hours_7d,
    sleep_observation_count_28d,
    sleep_missing_day_count_28d,

    sum_app_sessions_28d,
    avg_app_sessions_28d,
    avg_app_sessions_7d,
    avg_app_sessions_prior_7d,
    app_sessions_trend_7d,
    app_session_observation_count_28d,
    app_sessions_missing_day_count_28d,

    rewards_viewed_28d,
    rewards_redeemed_28d,
    reward_redemption_rate_28d,

    interventions_received_28d,
    interventions_opened_28d,
    interventions_clicked_28d,
    intervention_open_rate_28d,
    intervention_click_rate_28d,

    step_outlier_day_count_28d,
    late_record_count_28d,
    activity_level_change_count_28d,
    avg_goal_completion_percentage_28d

FROM `vitality_engagement_dev.engagement_modeling_split`;
