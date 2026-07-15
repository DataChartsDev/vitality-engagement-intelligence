"""Generate synthetic app, reward, and intervention history."""

from typing import Final

import numpy as np
import pandas as pd

from vitality_engagement.data.schema import GenerationConfig

REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "member_id",
        "date",
        "app_sessions",
        "reward_profile",
        "intervention_profile",
    }
)

REWARD_VIEW_PROBABILITIES: Final[dict[str, float]] = {
    "low": 0.08,
    "medium": 0.18,
    "high": 0.35,
}

REWARD_REDEMPTION_PROBABILITIES: Final[dict[str, float]] = {
    "low": 0.05,
    "medium": 0.15,
    "high": 0.30,
}

INTERVENTION_OPEN_PROBABILITIES: Final[dict[str, float]] = {
    "low": 0.25,
    "medium": 0.50,
    "high": 0.75,
}

INTERVENTION_CLICK_PROBABILITIES: Final[dict[str, float]] = {
    "low": 0.08,
    "medium": 0.18,
    "high": 0.35,
}

INTERVENTION_TYPES: Final[tuple[str, ...]] = (
    "progress_reminder",
    "smaller_goal_suggestion",
    "reward_reminder",
    "social_encouragement",
    "educational_message",
    "recovery_suggestion",
)

INTERVENTION_TYPE_PROBABILITIES: Final[tuple[float, ...]] = (
    0.25,
    0.15,
    0.20,
    0.15,
    0.10,
    0.15,
)


def calculate_days_since_last_app_session(
    app_sessions: pd.Series,
) -> pd.Series:
    """Calculate app-session recency within one member's history."""
    recency_values: list[int] = []
    days_since_session = 0

    for session_count in app_sessions.astype("int64").tolist():
        if session_count > 0:
            days_since_session = 0
        else:
            days_since_session += 1

        recency_values.append(days_since_session)

    return pd.Series(
        recency_values,
        index=app_sessions.index,
        dtype="int64",
    )


def add_interaction_history(
    engagement: pd.DataFrame,
    config: GenerationConfig,
) -> pd.DataFrame:
    """Add app recency, reward actions, and intervention history.

    Intervention assignment occurs randomly on weekly opportunity days.
    This supports a later synthetic uplift-modelling demonstration.

    Args:
        engagement: Daily synthetic engagement records.
        config: Reproducible data-generation configuration.

    Returns:
        Engagement records with interaction-history fields.

    Raises:
        ValueError: If required columns are missing.
    """
    missing_columns = REQUIRED_COLUMNS - set(engagement.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required interaction columns: {missing_text}")

    result = (
        engagement.sort_values(
            ["member_id", "date"],
        )
        .reset_index(drop=True)
        .copy()
    )

    rng = np.random.default_rng(config.random_seed + 3)
    row_count = len(result)

    result["days_since_last_app_session"] = (
        result.groupby(
            "member_id",
            sort=False,
        )["app_sessions"]
        .transform(calculate_days_since_last_app_session)
        .astype("int64")
    )

    app_sessions = result["app_sessions"].astype("int64").to_numpy()

    reward_view_probability = (
        result["reward_profile"].map(REWARD_VIEW_PROBABILITIES).astype(float).to_numpy()
    )

    rewards_viewed = rng.binomial(
        n=app_sessions,
        p=reward_view_probability,
    ).astype(np.int64)

    reward_redemption_probability = (
        result["reward_profile"].map(REWARD_REDEMPTION_PROBABILITIES).astype(float).to_numpy()
    )

    rewards_redeemed = rng.binomial(
        n=rewards_viewed,
        p=reward_redemption_probability,
    ).astype(np.int64)

    weekly_opportunity = result["date"].dt.dayofweek.to_numpy() == 0

    intervention_received = weekly_opportunity & (rng.random(row_count) < 0.35)

    proposed_intervention_types = rng.choice(
        INTERVENTION_TYPES,
        size=row_count,
        p=INTERVENTION_TYPE_PROBABILITIES,
    )

    intervention_type = np.where(
        intervention_received,
        proposed_intervention_types,
        "none",
    )

    open_probability = (
        result["intervention_profile"].map(INTERVENTION_OPEN_PROBABILITIES).astype(float).to_numpy()
    )

    opened_values = intervention_received & (rng.random(row_count) < open_probability)

    click_probability = (
        result["intervention_profile"]
        .map(INTERVENTION_CLICK_PROBABILITIES)
        .astype(float)
        .to_numpy()
    )

    clicked_values = opened_values & (rng.random(row_count) < click_probability)

    intervention_opened = pd.Series(
        pd.NA,
        index=result.index,
        dtype="boolean",
    )

    intervention_clicked = pd.Series(
        pd.NA,
        index=result.index,
        dtype="boolean",
    )

    intervention_opened.loc[intervention_received] = opened_values[intervention_received]

    intervention_clicked.loc[intervention_received] = clicked_values[intervention_received]

    return result.assign(
        rewards_viewed=rewards_viewed,
        rewards_redeemed=rewards_redeemed,
        intervention_received=intervention_received,
        intervention_type=intervention_type,
        intervention_opened=intervention_opened,
        intervention_clicked=intervention_clicked,
    )
