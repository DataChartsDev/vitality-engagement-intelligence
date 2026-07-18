"""Tests for untouched evaluation using a validation-frozen threshold."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from vitality_engagement.models.evaluation import ModelEvaluationError
from vitality_engagement.models.selection import FrozenModelSelection
from vitality_engagement.models.test_evaluation import (
    evaluate_frozen_threshold_probabilities,
)


def make_target(values: list[bool]) -> pd.Series[bool]:
    """Create a typed Boolean target series."""
    return pd.Series(values, dtype=bool)


def make_selection(threshold: float) -> FrozenModelSelection:
    """Create a compact validation-only selection record."""
    return FrozenModelSelection(
        model_name="test-logistic-model",
        selection_split="validation",
        selection_metric="positive_f1",
        threshold=threshold,
        validation_start_date=date(2025, 5, 1),
        validation_end_date=date(2025, 5, 31),
        validation_row_count=100,
        validation_positive_count=20,
        validation_positive_f1=0.75,
    )


def test_frozen_threshold_is_used_without_test_reselection() -> None:
    target = make_target([False, False, True, True])
    probabilities = np.array([0.1, 0.4, 0.35, 0.8])

    evaluation = evaluate_frozen_threshold_probabilities(
        target,
        probabilities,
        selection=make_selection(0.5),
    )

    assert evaluation.frozen_threshold.threshold == pytest.approx(0.5)
    assert evaluation.frozen_threshold.positive_f1 == pytest.approx(2.0 / 3.0)


def test_test_metrics_have_expected_confusion_counts() -> None:
    target = make_target([False, False, True, True])
    probabilities = np.array([0.1, 0.4, 0.35, 0.8])

    evaluation = evaluate_frozen_threshold_probabilities(
        target,
        probabilities,
        selection=make_selection(0.5),
    )
    metrics = evaluation.frozen_threshold

    assert metrics.true_negatives == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 1
    assert metrics.true_positives == 1


def test_test_evaluation_returns_probability_metrics() -> None:
    target = make_target([False, False, True, True])
    probabilities = np.array([0.1, 0.4, 0.35, 0.8])

    evaluation = evaluate_frozen_threshold_probabilities(
        target,
        probabilities,
        selection=make_selection(0.5),
    )

    assert evaluation.row_count == 4
    assert evaluation.positive_count == 2
    assert evaluation.base_positive_rate == pytest.approx(0.5)
    assert 0.0 <= evaluation.roc_auc <= 1.0
    assert 0.0 <= evaluation.pr_auc <= 1.0
    assert 0.0 <= evaluation.average_precision <= 1.0
    assert evaluation.logarithmic_loss >= 0.0
    assert 0.0 <= evaluation.brier_score <= 1.0
    assert len(evaluation.calibration_bins) >= 1


def test_invalid_test_probabilities_are_rejected() -> None:
    target = make_target([False, True])

    with pytest.raises(
        ModelEvaluationError,
        match="between zero and one",
    ):
        evaluate_frozen_threshold_probabilities(
            target,
            np.array([-0.1, 1.1]),
            selection=make_selection(0.5),
        )
