"""Tests for baseline daily engagement generation."""

from pandas.testing import assert_frame_equal

from vitality_engagement.data.generate_engagement import (
    generate_daily_engagement,
)
from vitality_engagement.data.schema import GenerationConfig


def test_daily_engagement_has_expected_shape_and_columns() -> None:
    """Daily engagement should contain one record per member-date."""
    config = GenerationConfig(
        member_count=3,
        day_count=4,
        random_seed=42,
    )

    engagement = generate_daily_engagement(config)

    assert engagement.shape == (12, 12)

    assert list(engagement.columns) == [
        "member_id",
        "age_band",
        "membership_months",
        "activity_level",
        "reward_profile",
        "intervention_profile",
        "date",
        "daily_steps",
        "active_minutes",
        "sleep_hours",
        "weekly_goal",
        "app_sessions",
    ]


def test_daily_behaviour_remains_within_expected_ranges() -> None:
    """Baseline behavioural values should remain plausible."""
    engagement = generate_daily_engagement(
        GenerationConfig(
            member_count=100,
            day_count=30,
            random_seed=42,
        )
    )

    assert (
        engagement["daily_steps"]
        .between(
            0,
            35000,
        )
        .all()
    )

    assert (
        engagement["active_minutes"]
        .between(
            0,
            300,
        )
        .all()
    )

    assert (
        engagement["sleep_hours"]
        .between(
            4.0,
            10.5,
        )
        .all()
    )

    assert engagement["app_sessions"].ge(0).all()

    assert (
        engagement["weekly_goal"]
        .between(
            140,
            450,
        )
        .all()
    )

    assert engagement["weekly_goal"].mod(10).eq(0).all()
    assert not engagement.isna().any().any()


def test_weekly_goal_matches_activity_level_band() -> None:
    """Weekly goals should remain appropriate for activity level."""
    engagement = generate_daily_engagement(
        GenerationConfig(
            member_count=500,
            day_count=7,
            random_seed=42,
        )
    )

    low_activity = engagement[engagement["activity_level"] == "low"]

    moderate_activity = engagement[engagement["activity_level"] == "moderate"]

    high_activity = engagement[engagement["activity_level"] == "high"]

    assert (
        low_activity["weekly_goal"]
        .between(
            140,
            170,
        )
        .all()
    )

    assert (
        moderate_activity["weekly_goal"]
        .between(
            220,
            270,
        )
        .all()
    )

    assert (
        high_activity["weekly_goal"]
        .between(
            310,
            390,
        )
        .all()
    )


def test_weekly_goal_is_stable_within_member() -> None:
    """Each member should retain one goal throughout the dataset."""
    engagement = generate_daily_engagement(
        GenerationConfig(
            member_count=100,
            day_count=30,
            random_seed=42,
        )
    )

    goals_per_member = engagement.groupby("member_id")["weekly_goal"].nunique()

    assert goals_per_member.eq(1).all()


def test_daily_engagement_is_reproducible() -> None:
    """Identical configurations should produce identical records."""
    config = GenerationConfig(
        member_count=25,
        day_count=14,
        random_seed=17,
    )

    first_result = generate_daily_engagement(config)
    second_result = generate_daily_engagement(config)

    assert_frame_equal(
        first_result,
        second_result,
    )


def test_daily_engagement_changes_with_random_seed() -> None:
    """Different seeds should produce different daily behaviour."""
    first_result = generate_daily_engagement(
        GenerationConfig(
            member_count=25,
            day_count=14,
            random_seed=17,
        )
    )

    second_result = generate_daily_engagement(
        GenerationConfig(
            member_count=25,
            day_count=14,
            random_seed=18,
        )
    )

    assert not first_result.equals(second_result)
