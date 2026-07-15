"""Tests for synthetic-data schema definitions."""

import pytest
from pydantic import ValidationError

from vitality_engagement.data.schema import GenerationConfig


def test_default_generation_config() -> None:
    """Default configuration should be small and reproducible."""
    config = GenerationConfig()

    assert config.member_count == 500
    assert config.day_count == 180
    assert config.random_seed == 42


def test_generation_config_accepts_valid_values() -> None:
    """Explicit valid values should be retained."""
    config = GenerationConfig(
        member_count=100,
        day_count=30,
        random_seed=7,
    )

    assert config.member_count == 100
    assert config.day_count == 30
    assert config.random_seed == 7


def test_generation_config_rejects_zero_members() -> None:
    """A dataset cannot contain zero members."""
    with pytest.raises(ValidationError):
        GenerationConfig(member_count=0)
