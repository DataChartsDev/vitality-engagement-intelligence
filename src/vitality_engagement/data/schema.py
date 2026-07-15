"""Schema definitions for synthetic engagement data."""

from enum import StrEnum

from pydantic import BaseModel, Field, PositiveInt


class ActivityLevel(StrEnum):
    """Synthetic baseline activity categories."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class ResponseProfile(StrEnum):
    """Synthetic behavioural-response categories."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GenerationConfig(BaseModel):
    """Configuration for reproducible synthetic-data generation."""

    member_count: PositiveInt = 500
    day_count: PositiveInt = 180
    random_seed: int = Field(default=42, ge=0)
