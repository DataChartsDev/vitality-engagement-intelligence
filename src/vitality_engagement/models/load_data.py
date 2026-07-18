"""Load leakage-safe chronological modelling data from local Parquet."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Final, Literal

import pandas as pd

from vitality_engagement.models.export_features import validate_export_frame
from vitality_engagement.models.schema import (
    EXPECTED_MEMBER_COUNT,
    EXPECTED_SPLIT_ROW_COUNTS,
    EXPORT_COLUMNS,
    IDENTIFIER_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    SPLIT_COLUMN,
    TARGET_COLUMN,
)

DEFAULT_MODELING_DATA_PATH: Final = Path("data/modeling/engagement_modeling_split.parquet")

LabelledSplitName = Literal["train", "validation", "test"]

EXPECTED_SPLIT_ORDER: Final = (
    "train",
    "validation",
    "test",
    "scoring",
)


class ModelDataValidationError(ValueError):
    """Raised when modelling data violates chronological loading rules."""


@dataclass(frozen=True)
class LabelledModelSplit:
    """A labelled chronological model-development split."""

    name: LabelledSplitName
    identifiers: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series[bool]

    def __post_init__(self) -> None:
        """Validate that all split components contain the same rows."""
        row_count = len(self.features)

        if len(self.identifiers) != row_count or len(self.target) != row_count:
            raise ModelDataValidationError(
                f"{self.name} split components have inconsistent row counts."
            )


@dataclass(frozen=True)
class ScoringModelSplit:
    """An unlabelled operational scoring split."""

    identifiers: pd.DataFrame
    features: pd.DataFrame

    def __post_init__(self) -> None:
        """Validate that scoring components contain the same rows."""
        if len(self.identifiers) != len(self.features):
            raise ModelDataValidationError("Scoring split components have inconsistent row counts.")


@dataclass(frozen=True)
class ChronologicalModelingData:
    """All Stage 3 chronological partitions prepared for Python modelling."""

    train: LabelledModelSplit
    validation: LabelledModelSplit
    test: LabelledModelSplit
    scoring: ScoringModelSplit


def _validate_split_block_order(frame: pd.DataFrame) -> None:
    """Ensure split rows occur in one uninterrupted chronological sequence."""
    split_values = frame[SPLIT_COLUMN].astype(str).tolist()

    actual_blocks = tuple(split_name for split_name, _ in groupby(split_values))

    if actual_blocks != EXPECTED_SPLIT_ORDER:
        raise ModelDataValidationError(
            "Dataset splits are not arranged in the required chronological order."
        )


def _select_split(
    frame: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Return an isolated copy of one chronological split."""
    return frame.loc[frame[SPLIT_COLUMN].eq(split_name)].copy()


def _select_identifiers(frame: pd.DataFrame) -> pd.DataFrame:
    """Return member and prediction-date audit columns."""
    return frame.loc[:, list(IDENTIFIER_COLUMNS)].reset_index(drop=True)


def _select_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only approved model predictor columns."""
    return frame.loc[:, list(MODEL_FEATURE_COLUMNS)].reset_index(drop=True)


def _build_labelled_split(
    frame: pd.DataFrame,
    split_name: LabelledSplitName,
) -> LabelledModelSplit:
    """Build one labelled model-development split."""
    split_frame = _select_split(frame, split_name)
    target = split_frame[TARGET_COLUMN].astype(bool).reset_index(drop=True)

    return LabelledModelSplit(
        name=split_name,
        identifiers=_select_identifiers(split_frame),
        features=_select_features(split_frame),
        target=target,
    )


def build_chronological_modeling_data(
    frame: pd.DataFrame,
    *,
    expected_split_row_counts: Mapping[str, int] = EXPECTED_SPLIT_ROW_COUNTS,
    expected_member_count: int = EXPECTED_MEMBER_COUNT,
) -> ChronologicalModelingData:
    """Validate a modelling frame and build its chronological partitions."""
    validate_export_frame(
        frame,
        expected_split_row_counts=expected_split_row_counts,
        expected_member_count=expected_member_count,
    )
    _validate_split_block_order(frame)

    scoring_frame = _select_split(frame, "scoring")

    return ChronologicalModelingData(
        train=_build_labelled_split(frame, "train"),
        validation=_build_labelled_split(frame, "validation"),
        test=_build_labelled_split(frame, "test"),
        scoring=ScoringModelSplit(
            identifiers=_select_identifiers(scoring_frame),
            features=_select_features(scoring_frame),
        ),
    )


def load_chronological_modeling_data(
    path: Path = DEFAULT_MODELING_DATA_PATH,
    *,
    expected_split_row_counts: Mapping[str, int] = EXPECTED_SPLIT_ROW_COUNTS,
    expected_member_count: int = EXPECTED_MEMBER_COUNT,
) -> ChronologicalModelingData:
    """Load, validate, and partition the exported modelling dataset."""
    if not path.is_file():
        raise FileNotFoundError(f"Modelling data file does not exist: {path}")

    frame = pd.read_parquet(
        path,
        columns=list(EXPORT_COLUMNS),
    )

    return build_chronological_modeling_data(
        frame,
        expected_split_row_counts=expected_split_row_counts,
        expected_member_count=expected_member_count,
    )
