"""Tests for frozen validation-only model-selection decisions."""

from datetime import date

import pytest

from vitality_engagement.models.selection import (
    PYTHON_LOGISTIC_SELECTION,
    FrozenModelSelection,
    ModelSelectionError,
)


def test_python_logistic_threshold_is_frozen_from_validation() -> None:
    selection = PYTHON_LOGISTIC_SELECTION

    assert selection.model_name == "python_logistic_baseline"
    assert selection.selection_split == "validation"
    assert selection.selection_metric == "positive_f1"
    assert selection.threshold == pytest.approx(0.431)


def test_validation_selection_metadata_matches_stage_three_split() -> None:
    selection = PYTHON_LOGISTIC_SELECTION

    assert selection.validation_start_date == date(2025, 5, 1)
    assert selection.validation_end_date == date(2025, 5, 31)
    assert selection.validation_row_count == 15_500
    assert selection.validation_positive_count == 3_614
    assert selection.validation_positive_f1 == pytest.approx(0.7720781113378025)


def test_invalid_frozen_threshold_is_rejected() -> None:
    with pytest.raises(
        ModelSelectionError,
        match="between zero and one",
    ):
        FrozenModelSelection(
            model_name="invalid-model",
            selection_split="validation",
            selection_metric="positive_f1",
            threshold=1.1,
            validation_start_date=date(2025, 5, 1),
            validation_end_date=date(2025, 5, 31),
            validation_row_count=100,
            validation_positive_count=20,
            validation_positive_f1=0.7,
        )


def test_invalid_validation_counts_are_rejected() -> None:
    with pytest.raises(
        ModelSelectionError,
        match="positive count",
    ):
        FrozenModelSelection(
            model_name="invalid-model",
            selection_split="validation",
            selection_metric="positive_f1",
            threshold=0.5,
            validation_start_date=date(2025, 5, 1),
            validation_end_date=date(2025, 5, 31),
            validation_row_count=100,
            validation_positive_count=101,
            validation_positive_f1=0.7,
        )
