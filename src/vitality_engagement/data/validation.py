"""Validate generated modelling datasets."""

from typing import Final

import pandas as pd

from vitality_engagement.data.schema import GenerationConfig

REQUIRED_MODELING_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "member_id",
        "date",
        "activity_level",
        "daily_steps",
        "active_minutes",
        "sleep_hours",
        "weekly_goal",
        "app_sessions",
        "future_7_day_active_minutes",
        "next_week_goal_completed",
        "will_miss_goal_next_7_days",
        "sleep_hours_missing",
        "app_sessions_missing",
        "is_step_outlier",
        "is_late_record",
        "record_delay_days",
        "available_date",
        "activity_level_changed",
    }
)

VALID_ACTIVITY_LEVELS: Final[frozenset[str]] = frozenset(
    {
        "low",
        "moderate",
        "high",
    }
)

TARGET_COLUMNS: Final[tuple[str, ...]] = (
    "future_7_day_active_minutes",
    "next_week_goal_completed",
    "will_miss_goal_next_7_days",
)


def validate_modeling_dataset(
    data: pd.DataFrame,
    config: GenerationConfig,
) -> None:
    """Validate the structure and consistency of a modelling dataset.

    Args:
        data: Complete synthetic modelling dataset.
        config: Configuration used to generate the dataset.

    Raises:
        ValueError: If any validation rule fails.
    """
    _validate_required_columns(data)
    _validate_duplicate_member_dates(data)
    _validate_expected_size(data, config)
    _validate_member_count(data, config)
    _validate_activity_levels(data)
    _validate_missingness_indicators(data)
    _validate_step_outliers(data)
    _validate_record_delays(data)
    _validate_future_targets(data, config)


def _validate_required_columns(
    data: pd.DataFrame,
) -> None:
    """Validate that every required modelling column exists."""
    missing_columns = REQUIRED_MODELING_COLUMNS - set(data.columns)

    if not missing_columns:
        return

    missing_text = ", ".join(sorted(missing_columns))

    raise ValueError(f"Missing required modelling columns: {missing_text}")


def _validate_duplicate_member_dates(
    data: pd.DataFrame,
) -> None:
    """Reject duplicate member-date records."""
    duplicate_count = int(
        data.duplicated(
            subset=["member_id", "date"],
        ).sum()
    )

    if duplicate_count == 0:
        return

    raise ValueError(f"Found {duplicate_count} duplicate member-date rows.")


def _validate_expected_size(
    data: pd.DataFrame,
    config: GenerationConfig,
) -> None:
    """Validate the expected total number of rows."""
    expected_rows = config.member_count * config.day_count

    if len(data) == expected_rows:
        return

    raise ValueError(f"Expected {expected_rows} rows, but received {len(data)}.")


def _validate_member_count(
    data: pd.DataFrame,
    config: GenerationConfig,
) -> None:
    """Validate the number of unique members."""
    actual_member_count = data["member_id"].nunique()

    if actual_member_count == config.member_count:
        return

    raise ValueError(
        "Generated member count does not match the configuration. "
        f"Expected {config.member_count}, "
        f"but received {actual_member_count}."
    )


def _validate_activity_levels(
    data: pd.DataFrame,
) -> None:
    """Reject unsupported activity-level categories."""
    observed_activity_levels = set(data["activity_level"].dropna().astype(str))

    invalid_activity_levels = observed_activity_levels - VALID_ACTIVITY_LEVELS

    if not invalid_activity_levels:
        return

    invalid_text = ", ".join(sorted(invalid_activity_levels))

    raise ValueError(f"Invalid activity levels found: {invalid_text}")


def _validate_missingness_indicators(
    data: pd.DataFrame,
) -> None:
    """Validate that missingness flags match observed values."""
    sleep_missing_matches = data["sleep_hours"].isna() == data["sleep_hours_missing"].astype(bool)

    if not sleep_missing_matches.all():
        raise ValueError("Sleep missingness indicator does not match sleep values.")

    app_missing_matches = data["app_sessions"].isna() == data["app_sessions_missing"].astype(bool)

    if not app_missing_matches.all():
        raise ValueError("App-session missingness indicator does not match values.")


def _validate_step_outliers(
    data: pd.DataFrame,
) -> None:
    """Validate that step-outlier flags match step values."""
    step_outlier_matches = data["daily_steps"].gt(35000) == data["is_step_outlier"].astype(bool)

    if step_outlier_matches.all():
        return

    raise ValueError("Step-outlier indicator does not match step values.")


def _validate_record_delays(
    data: pd.DataFrame,
) -> None:
    """Validate late-record flags, delay values, and availability dates."""
    delay_values_valid = data["record_delay_days"].between(0, 3).all()

    if not delay_values_valid:
        raise ValueError("Record delays must remain between zero and three days.")

    late_record_matches = data["record_delay_days"].gt(0) == data["is_late_record"].astype(bool)

    if not late_record_matches.all():
        raise ValueError("Late-record indicator does not match record delays.")

    expected_available_date = data["date"] + pd.to_timedelta(
        data["record_delay_days"],
        unit="D",
    )

    if expected_available_date.equals(data["available_date"]):
        return

    raise ValueError("Available dates do not match record delays.")


def _validate_future_targets(
    data: pd.DataFrame,
    config: GenerationConfig,
) -> None:
    """Validate future-window target availability and consistency."""
    if config.day_count < 7:
        return

    final_rows = (
        data.sort_values(
            ["member_id", "date"],
        )
        .groupby(
            "member_id",
            sort=False,
        )
        .tail(7)
    )

    if not final_rows[list(TARGET_COLUMNS)].isna().all().all():
        raise ValueError("Final seven member-days must have missing targets.")

    if config.day_count == 7:
        return

    labelled_rows = (
        data.sort_values(
            ["member_id", "date"],
        )
        .groupby(
            "member_id",
            sort=False,
        )
        .head(config.day_count - 7)
    )

    if labelled_rows["will_miss_goal_next_7_days"].isna().any():
        raise ValueError("Complete future windows must have target labels.")

    expected_miss_target = (
        labelled_rows["future_7_day_active_minutes"] < labelled_rows["weekly_goal"]
    )

    actual_miss_target = labelled_rows["will_miss_goal_next_7_days"].astype(bool)

    if not expected_miss_target.equals(actual_miss_target):
        raise ValueError("Future goal-miss target does not match future activity.")
