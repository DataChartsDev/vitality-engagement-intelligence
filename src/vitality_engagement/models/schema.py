"""Leakage-safe schema definitions for Python model development."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

SOURCE_TARGET_COLUMN: Final = "label_will_miss_goal_next_7_days"
TARGET_COLUMN: Final = "will_miss_goal_next_7_days"
SPLIT_COLUMN: Final = "dataset_split"

IDENTIFIER_COLUMNS: Final = (
    "member_id",
    "prediction_date",
)

CATEGORICAL_FEATURE_COLUMNS: Final = (
    "age_band_as_of",
    "activity_level_as_of",
    "reward_profile_as_of",
)

MODEL_FEATURE_COLUMNS: Final = (
    "age_band_as_of",
    "membership_months_as_of",
    "activity_level_as_of",
    "reward_profile_as_of",
    "weekly_goal_as_of",
    "weekly_active_minutes_so_far_as_of",
    "goal_completion_percentage_as_of",
    "previous_goal_streak_as_of",
    "previous_failed_goals_as_of",
    "days_since_last_app_session_as_of",
    "available_day_count_28d",
    "unavailable_day_count_28d",
    "avg_daily_steps_28d",
    "stddev_daily_steps_28d",
    "avg_daily_steps_7d",
    "avg_daily_steps_prior_7d",
    "daily_steps_trend_7d",
    "sum_active_minutes_28d",
    "avg_active_minutes_28d",
    "stddev_active_minutes_28d",
    "avg_active_minutes_7d",
    "avg_active_minutes_prior_7d",
    "active_minutes_trend_7d",
    "active_day_count_28d",
    "avg_sleep_hours_28d",
    "avg_sleep_hours_7d",
    "sleep_observation_count_28d",
    "sleep_missing_day_count_28d",
    "sum_app_sessions_28d",
    "avg_app_sessions_28d",
    "avg_app_sessions_7d",
    "avg_app_sessions_prior_7d",
    "app_sessions_trend_7d",
    "app_session_observation_count_28d",
    "app_sessions_missing_day_count_28d",
    "rewards_viewed_28d",
    "rewards_redeemed_28d",
    "reward_redemption_rate_28d",
    "interventions_received_28d",
    "interventions_opened_28d",
    "interventions_clicked_28d",
    "intervention_open_rate_28d",
    "intervention_click_rate_28d",
    "step_outlier_day_count_28d",
    "late_record_count_28d",
    "activity_level_change_count_28d",
    "avg_goal_completion_percentage_28d",
)

NUMERIC_FEATURE_COLUMNS: Final = tuple(
    column for column in MODEL_FEATURE_COLUMNS if column not in CATEGORICAL_FEATURE_COLUMNS
)

EXPORT_COLUMNS: Final = (
    *IDENTIFIER_COLUMNS,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    *MODEL_FEATURE_COLUMNS,
)

PROHIBITED_PREDICTOR_COLUMNS: Final = (
    "member_id",
    "prediction_date",
    "feature_window_start",
    "feature_window_end",
    "max_source_activity_date",
    "max_source_available_date",
    "dataset_split",
    "label_will_miss_goal_next_7_days",
    "will_miss_goal_next_7_days",
    "future_7_day_active_minutes",
    "next_week_goal_completed",
    "intervention_profile_as_of",
)

EXPECTED_SPLIT_ROW_COUNTS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "train": 46_000,
        "validation": 15_500,
        "test": 11_000,
        "scoring": 3_500,
    }
)

EXPECTED_TOTAL_ROW_COUNT: Final = sum(EXPECTED_SPLIT_ROW_COUNTS.values())
EXPECTED_MEMBER_COUNT: Final = 500
