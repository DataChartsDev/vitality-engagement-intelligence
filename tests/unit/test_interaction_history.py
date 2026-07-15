"""Tests for synthetic interaction-history generation."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from vitality_engagement.data.generate_engagement import (
    generate_daily_engagement,
)
from vitality_engagement.data.interaction_history import (
    add_interaction_history,
    calculate_days_since_last_app_session,
)
from vitality_engagement.data.schema import GenerationConfig


def test_app_session_recency_is_calculated_correctly() -> None:
    """Recency should reset when an app session occurs."""
    app_sessions = pd.Series(
        [0, 0, 2, 0, 0, 1, 0],
        dtype="int64",
    )

    result = calculate_days_since_last_app_session(app_sessions)

    assert result.tolist() == [
        1,
        2,
        0,
        1,
        2,
        0,
        1,
    ]


def test_reward_actions_are_logically_consistent() -> None:
    """Viewed and redeemed rewards should remain consistent."""
    config = GenerationConfig(
        member_count=100,
        day_count=30,
        random_seed=42,
    )

    engagement = generate_daily_engagement(config)
    result = add_interaction_history(
        engagement,
        config,
    )

    assert result["rewards_viewed"].ge(0).all()
    assert result["rewards_redeemed"].ge(0).all()

    assert (result["rewards_viewed"] <= result["app_sessions"]).all()

    assert (result["rewards_redeemed"] <= result["rewards_viewed"]).all()


def test_intervention_fields_are_consistent() -> None:
    """Intervention metadata should agree with assignment."""
    config = GenerationConfig(
        member_count=200,
        day_count=60,
        random_seed=42,
    )

    engagement = generate_daily_engagement(config)
    result = add_interaction_history(
        engagement,
        config,
    )

    sent = result[result["intervention_received"]]
    not_sent = result[~result["intervention_received"]]

    assert not sent.empty
    assert sent["intervention_type"].ne("none").all()
    assert sent["intervention_opened"].notna().all()
    assert sent["intervention_clicked"].notna().all()

    assert not_sent["intervention_type"].eq("none").all()
    assert not_sent["intervention_opened"].isna().all()
    assert not_sent["intervention_clicked"].isna().all()


def test_clicked_interventions_were_opened() -> None:
    """An intervention cannot be clicked without being opened."""
    config = GenerationConfig(
        member_count=300,
        day_count=90,
        random_seed=42,
    )

    engagement = generate_daily_engagement(config)
    result = add_interaction_history(
        engagement,
        config,
    )

    sent = result[result["intervention_received"]]

    clicked = sent["intervention_clicked"].fillna(False).astype(bool)

    opened = sent["intervention_opened"].fillna(False).astype(bool)

    assert (~clicked | opened).all()


def test_interventions_only_occur_on_mondays() -> None:
    """The synthetic programme should use weekly opportunities."""
    config = GenerationConfig(
        member_count=200,
        day_count=60,
        random_seed=42,
    )

    engagement = generate_daily_engagement(config)
    result = add_interaction_history(
        engagement,
        config,
    )

    sent = result[result["intervention_received"]]

    assert not sent.empty
    assert (sent["date"].dt.dayofweek == 0).all()


def test_interaction_history_is_reproducible() -> None:
    """Identical inputs should produce identical interaction history."""
    config = GenerationConfig(
        member_count=50,
        day_count=30,
        random_seed=17,
    )

    engagement = generate_daily_engagement(config)

    first_result = add_interaction_history(
        engagement,
        config,
    )

    second_result = add_interaction_history(
        engagement,
        config,
    )

    assert_frame_equal(first_result, second_result)


def test_interaction_history_rejects_missing_columns() -> None:
    """Missing required fields should produce a clear error."""
    engagement = pd.DataFrame(
        {
            "member_id": ["M000001"],
            "date": pd.to_datetime(["2025-01-01"]),
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required interaction columns",
    ):
        add_interaction_history(
            engagement,
            GenerationConfig(
                member_count=1,
                day_count=1,
            ),
        )
