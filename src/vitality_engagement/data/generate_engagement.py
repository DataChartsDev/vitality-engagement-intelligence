"""Generate synthetic daily engagement records."""

from typing import Final

import numpy as np
import pandas as pd

from vitality_engagement.data.behaviour_patterns import (
    generate_engagement_trajectory,
)
from vitality_engagement.data.generate_members import generate_members
from vitality_engagement.data.goal_metrics import (
    add_goal_metrics_and_target,
)
from vitality_engagement.data.interaction_history import (
    add_interaction_history,
)
from vitality_engagement.data.schema import GenerationConfig

STEP_BASELINES: Final[dict[str, float]] = {
    "low": 4500.0,
    "moderate": 7500.0,
    "high": 11000.0,
}

WEEKLY_GOALS: Final[dict[str, int]] = {
    "low": 120,
    "moderate": 180,
    "high": 240,
}

APP_SESSION_RATES: Final[dict[str, float]] = {
    "low": 0.9,
    "moderate": 1.5,
    "high": 2.1,
}

SLEEP_ADJUSTMENTS: Final[dict[str, float]] = {
    "18-24": 0.2,
    "25-34": 0.1,
    "35-44": 0.0,
    "45-54": -0.1,
    "55+": -0.2,
}


def generate_member_day_skeleton(
    config: GenerationConfig,
) -> pd.DataFrame:
    """Create one row for every synthetic member and calendar date."""
    members = generate_members(config)

    dates = pd.DataFrame(
        {
            "date": pd.date_range(
                start=config.start_date,
                periods=config.day_count,
                freq="D",
            )
        }
    )

    skeleton = members.merge(dates, how="cross")

    return skeleton.sort_values(
        ["member_id", "date"],
        ignore_index=True,
    )


def generate_daily_engagement(
    config: GenerationConfig,
) -> pd.DataFrame:
    """Add baseline daily behavioural signals to the member-day skeleton."""
    engagement = generate_member_day_skeleton(config)
    rng = np.random.default_rng(config.random_seed + 1)

    trajectory = generate_engagement_trajectory(
        engagement,
        config,
    )

    time_multiplier = trajectory["engagement_multiplier"].astype(float).to_numpy()

    row_count = len(engagement)

    activity_baseline = engagement["activity_level"].map(STEP_BASELINES).astype(float).to_numpy()

    member_activity_factors = np.repeat(
        rng.normal(
            loc=1.0,
            scale=0.12,
            size=config.member_count,
        ),
        config.day_count,
    )

    weekend_mask = engagement["date"].dt.dayofweek.to_numpy() >= 5

    weekend_activity_factor = np.where(
        weekend_mask,
        0.88,
        1.0,
    )

    daily_activity_noise = np.clip(
        rng.normal(
            loc=1.0,
            scale=0.18,
            size=row_count,
        ),
        0.45,
        1.65,
    )

    daily_steps = np.clip(
        np.rint(
            activity_baseline
            * member_activity_factors
            * weekend_activity_factor
            * time_multiplier
            * daily_activity_noise
        ),
        0,
        35000,
    ).astype(np.int64)

    active_minutes = np.clip(
        np.rint(
            daily_steps / 170.0
            + rng.normal(
                loc=0.0,
                scale=7.0,
                size=row_count,
            )
        ),
        0,
        300,
    ).astype(np.int64)

    age_sleep_adjustment = engagement["age_band"].map(SLEEP_ADJUSTMENTS).astype(float).to_numpy()

    sleep_hours = np.clip(
        7.3
        + age_sleep_adjustment
        + np.where(weekend_mask, 0.25, 0.0)
        + rng.normal(
            loc=0.0,
            scale=0.65,
            size=row_count,
        ),
        4.0,
        10.5,
    )

    weekly_goal = engagement["activity_level"].map(WEEKLY_GOALS).astype("int64")

    app_session_rate = engagement["activity_level"].map(APP_SESSION_RATES).astype(float).to_numpy()

    app_sessions = rng.poisson(
        app_session_rate * np.where(weekend_mask, 0.85, 1.0) * np.clip(time_multiplier, 0.60, 1.40)
    ).astype(np.int64)

    return engagement.assign(
        daily_steps=daily_steps,
        active_minutes=active_minutes,
        sleep_hours=np.round(sleep_hours, 1),
        weekly_goal=weekly_goal,
        app_sessions=app_sessions,
    )


def generate_modeling_dataset(
    config: GenerationConfig,
) -> pd.DataFrame:
    """Generate daily engagement records with future outcome labels."""
    engagement = generate_daily_engagement(config)

    engagement_with_history = add_interaction_history(
        engagement,
        config,
    )

    return add_goal_metrics_and_target(engagement_with_history)

    """Generate daily engagement records with future outcome labels."""
    engagement = generate_daily_engagement(config)

    return add_goal_metrics_and_target(engagement)
