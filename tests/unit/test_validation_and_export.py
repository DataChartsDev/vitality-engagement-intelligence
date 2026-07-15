"""Tests for dataset validation and export."""

from pathlib import Path

import pandas as pd
import pytest

from vitality_engagement.data.export_dataset import (
    export_modeling_dataset,
)
from vitality_engagement.data.generate_engagement import (
    generate_modeling_dataset,
)
from vitality_engagement.data.schema import GenerationConfig
from vitality_engagement.data.validation import (
    validate_modeling_dataset,
)


def test_generated_dataset_passes_validation() -> None:
    """A complete generated dataset should pass validation."""
    config = GenerationConfig(
        member_count=20,
        day_count=30,
        random_seed=42,
    )

    data = generate_modeling_dataset(config)

    validate_modeling_dataset(data, config)


def test_validation_rejects_duplicate_member_dates() -> None:
    """Duplicate member-date keys should be rejected."""
    config = GenerationConfig(
        member_count=10,
        day_count=20,
        random_seed=42,
    )

    data = generate_modeling_dataset(config)

    duplicated_data = pd.concat(
        [
            data,
            data.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate member-date rows",
    ):
        validate_modeling_dataset(
            duplicated_data,
            config,
        )


def test_validation_rejects_missing_required_column() -> None:
    """Required modelling columns cannot be removed."""
    config = GenerationConfig(
        member_count=10,
        day_count=20,
        random_seed=42,
    )

    data = generate_modeling_dataset(config)
    incomplete_data = data.drop(columns=["daily_steps"])

    with pytest.raises(
        ValueError,
        match="Missing required modelling columns",
    ):
        validate_modeling_dataset(
            incomplete_data,
            config,
        )


def test_validation_rejects_indicator_mismatch() -> None:
    """Missingness indicators must agree with observed values."""
    config = GenerationConfig(
        member_count=10,
        day_count=20,
        random_seed=42,
    )

    data = generate_modeling_dataset(config)
    data.loc[0, "sleep_hours_missing"] = True

    with pytest.raises(
        ValueError,
        match="Sleep missingness indicator",
    ):
        validate_modeling_dataset(data, config)


def test_export_creates_csv_and_parquet_files(
    tmp_path: Path,
) -> None:
    """Export should create readable CSV and Parquet files."""
    config = GenerationConfig(
        member_count=5,
        day_count=20,
        random_seed=42,
    )

    csv_path, parquet_path = export_modeling_dataset(
        config,
        tmp_path,
    )

    assert csv_path.exists()
    assert parquet_path.exists()

    csv_data = pd.read_csv(csv_path)
    parquet_data = pd.read_parquet(parquet_path)

    assert len(csv_data) == 100
    assert len(parquet_data) == 100
    assert list(csv_data.columns) == list(parquet_data.columns)


def test_default_target_rate_is_useful_for_modeling() -> None:
    """The synthetic target should not be vanishingly rare."""
    config = GenerationConfig(
        member_count=500,
        day_count=180,
        random_seed=42,
    )

    data = generate_modeling_dataset(config)

    target_rate = data["will_miss_goal_next_7_days"].dropna().astype(bool).mean()

    assert 0.12 <= target_rate <= 0.30
