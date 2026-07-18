"""Tests for validation-only model evaluation."""

import numpy as np
import pandas as pd
import pytest

from vitality_engagement.models.evaluation import (
    ModelEvaluationError,
    build_threshold_grid,
    calculate_calibration_bins,
    calculate_threshold_metrics,
    calculate_top_decile_metrics,
    evaluate_validation_probabilities,
    select_validation_threshold,
)


def make_target(values: list[bool]) -> pd.Series[bool]:
    """Create a typed Boolean target series."""
    return pd.Series(values, dtype=bool)


def test_threshold_metrics_have_expected_confusion_values() -> None:
    target = make_target([False, False, True, True])
    probabilities = np.array([0.1, 0.4, 0.35, 0.8])

    metrics = calculate_threshold_metrics(
        target,
        probabilities,
        threshold=0.5,
    )

    assert metrics.true_negatives == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 1
    assert metrics.true_positives == 1
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.positive_f1 == pytest.approx(2.0 / 3.0)
    assert metrics.specificity == pytest.approx(1.0)
    assert metrics.accuracy == pytest.approx(0.75)


def test_threshold_selection_maximises_positive_f1() -> None:
    target = make_target([False, False, True, True])
    probabilities = np.array([0.1, 0.4, 0.35, 0.8])

    selected = select_validation_threshold(
        target,
        probabilities,
        thresholds=np.array([0.35, 0.4, 0.5]),
    )

    assert selected.threshold == pytest.approx(0.35)
    assert selected.positive_f1 == pytest.approx(0.8)


def test_threshold_tie_uses_lower_threshold() -> None:
    target = make_target([False, True])
    probabilities = np.array([0.2, 0.8])

    selected = select_validation_threshold(
        target,
        probabilities,
        thresholds=np.array([0.5, 0.7]),
    )

    assert selected.threshold == pytest.approx(0.5)
    assert selected.positive_f1 == pytest.approx(1.0)


def test_default_threshold_grid_has_expected_resolution() -> None:
    thresholds = build_threshold_grid()

    assert len(thresholds) == 1_001
    assert thresholds[0] == pytest.approx(0.0)
    assert thresholds[-1] == pytest.approx(1.0)
    assert thresholds[467] == pytest.approx(0.467)


def test_top_decile_metrics_have_expected_lift() -> None:
    target = make_target([True, True, True, True] + [False] * 16)
    probabilities = np.linspace(
        1.0,
        0.05,
        num=20,
    )

    metrics = calculate_top_decile_metrics(
        target,
        probabilities,
    )

    assert metrics.selected_row_count == 2
    assert metrics.selected_positive_count == 2
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.lift == pytest.approx(5.0)


def test_calibration_bins_calculate_ece_and_maximum_gap() -> None:
    target = make_target([False, False, True, True])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    bins, expected_calibration_error, maximum_gap = calculate_calibration_bins(
        target,
        probabilities,
        bin_count=2,
    )

    assert len(bins) == 2
    assert bins[0].row_count == 2
    assert bins[0].mean_predicted_probability == pytest.approx(0.15)
    assert bins[0].observed_positive_rate == pytest.approx(0.0)
    assert bins[1].mean_predicted_probability == pytest.approx(0.85)
    assert bins[1].observed_positive_rate == pytest.approx(1.0)
    assert expected_calibration_error == pytest.approx(0.15)
    assert maximum_gap == pytest.approx(0.15)


def test_validation_evaluation_returns_finite_probability_metrics() -> None:
    target = make_target([False, False, True, True])
    probabilities = np.array([0.1, 0.4, 0.35, 0.8])

    evaluation = evaluate_validation_probabilities(
        target,
        probabilities,
    )

    assert evaluation.row_count == 4
    assert evaluation.positive_count == 2
    assert evaluation.base_positive_rate == pytest.approx(0.5)
    assert 0.0 <= evaluation.roc_auc <= 1.0
    assert 0.0 <= evaluation.pr_auc <= 1.0
    assert 0.0 <= evaluation.average_precision <= 1.0
    assert evaluation.log_loss >= 0.0
    assert 0.0 <= evaluation.brier_score <= 1.0
    assert len(evaluation.calibration_bins) >= 1


def test_invalid_probabilities_are_rejected() -> None:
    target = make_target([False, True])

    with pytest.raises(
        ModelEvaluationError,
        match="between zero and one",
    ):
        evaluate_validation_probabilities(
            target,
            np.array([-0.1, 1.1]),
        )


def test_single_class_target_is_rejected() -> None:
    target = make_target([True, True])

    with pytest.raises(
        ModelEvaluationError,
        match="both Boolean classes",
    ):
        evaluate_validation_probabilities(
            target,
            np.array([0.7, 0.8]),
        )
