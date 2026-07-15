"""Generate behavioural patterns that change over time."""

import numpy as np
import pandas as pd

from vitality_engagement.data.schema import GenerationConfig


def generate_engagement_trajectory(
    skeleton: pd.DataFrame,
    config: GenerationConfig,
) -> pd.DataFrame:
    """Generate reproducible time-varying engagement factors.

    The trajectory includes:

    - gradual disengagement for a subset of members;
    - temporary new-member enthusiasm;
    - seasonal activity variation.

    Args:
        skeleton: Member-day data ordered by member and date.
        config: Reproducible data-generation configuration.

    Returns:
        A DataFrame containing latent engagement trajectory values.

    Raises:
        ValueError: If the skeleton does not have the expected row count.
    """
    expected_rows = config.member_count * config.day_count

    if len(skeleton) != expected_rows:
        message = f"Expected {expected_rows} member-day rows, but received {len(skeleton)}."
        raise ValueError(message)

    rng = np.random.default_rng(config.random_seed + 2)

    day_index = np.tile(
        np.arange(config.day_count),
        config.member_count,
    )

    member_is_disengager = rng.random(config.member_count) < 0.25

    disengagement_slopes = rng.uniform(
        low=-0.0030,
        high=-0.0010,
        size=config.member_count,
    )

    stable_slopes = rng.normal(
        loc=0.0,
        scale=0.0003,
        size=config.member_count,
    )

    member_slopes = np.where(
        member_is_disengager,
        disengagement_slopes,
        stable_slopes,
    )

    row_slopes = np.repeat(
        member_slopes,
        config.day_count,
    )

    trajectory_factor = np.clip(
        1.0 + row_slopes * day_index,
        0.55,
        1.20,
    )

    membership_months = skeleton["membership_months"].astype("int64").to_numpy()

    new_member_factor = np.where(
        membership_months <= 3,
        1.0 + 0.18 * np.exp(-day_index / 45.0),
        1.0,
    )

    day_of_year = skeleton["date"].dt.dayofyear.to_numpy()

    seasonal_factor = 1.0 + 0.06 * np.sin(2.0 * np.pi * (day_of_year - 30) / 365.25)

    engagement_multiplier = np.clip(
        trajectory_factor * new_member_factor * seasonal_factor,
        0.50,
        1.50,
    )

    return pd.DataFrame(
        {
            "member_id": skeleton["member_id"].to_numpy(),
            "date": skeleton["date"].to_numpy(),
            "engagement_multiplier": np.round(
                engagement_multiplier,
                4,
            ),
            "is_gradual_disengager": np.repeat(
                member_is_disengager,
                config.day_count,
            ),
        }
    )
