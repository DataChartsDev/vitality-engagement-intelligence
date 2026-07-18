"""Validation-only model evaluation and threshold selection."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

from vitality_engagement.models.baseline import fit_logistic_baseline
from vitality_engagement.models.load_data import ChronologicalModelingData

DEFAULT_THRESHOLD_STEP: Final = 0.001
CALIBRATION_BIN_COUNT: Final = 10
TOP_DECILE_FRACTION: Final = 0.10


class ModelEvaluationError(ValueError):
    """Raised when model-evaluation inputs or outputs are invalid."""


@dataclass(frozen=True)
class ThresholdMetrics:
    """Classification metrics calculated at one probability threshold."""

    threshold: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    precision: float
    recall: float
    positive_f1: float
    specificity: float
    accuracy: float


@dataclass(frozen=True)
class TopDecileMetrics:
    """Performance among the highest-risk ten percent of rows."""

    selected_row_count: int
    selected_positive_count: int
    recall: float
    precision: float
    lift: float


@dataclass(frozen=True)
class CalibrationBin:
    """Observed and predicted risk within one probability interval."""

    bin_index: int
    lower_bound: float
    upper_bound: float
    row_count: int
    mean_predicted_probability: float
    observed_positive_rate: float
    absolute_gap: float


@dataclass(frozen=True)
class ValidationEvaluation:
    """Complete validation-only evaluation for one fitted model."""

    row_count: int
    positive_count: int
    base_positive_rate: float
    roc_auc: float
    pr_auc: float
    average_precision: float
    log_loss: float
    brier_score: float
    selected_threshold: ThresholdMetrics
    top_decile: TopDecileMetrics
    calibration_bins: tuple[CalibrationBin, ...]
    expected_calibration_error: float
    maximum_calibration_gap: float


def _prepare_binary_inputs(
    target: pd.Series[bool],
    probabilities: npt.ArrayLike,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    """Validate and convert binary targets and positive probabilities."""
    target_array = target.to_numpy(dtype=np.bool_)
    probability_array = np.asarray(probabilities, dtype=np.float64)

    if target_array.ndim != 1 or probability_array.ndim != 1:
        raise ModelEvaluationError("Targets and probabilities must be one-dimensional.")

    if len(target_array) == 0:
        raise ModelEvaluationError("Evaluation data must not be empty.")

    if len(target_array) != len(probability_array):
        raise ModelEvaluationError("Target and probability row counts do not match.")

    if not bool(np.isfinite(probability_array).all()):
        raise ModelEvaluationError("Probabilities contain non-finite values.")

    if bool(((probability_array < 0.0) | (probability_array > 1.0)).any()):
        raise ModelEvaluationError("Probabilities must fall between zero and one.")

    observed_classes = set(np.unique(target_array).tolist())
    if observed_classes != {False, True}:
        raise ModelEvaluationError("Evaluation targets must contain both Boolean classes.")

    return target_array, probability_array


def _safe_divide(numerator: int, denominator: int) -> float:
    """Return a finite ratio, using zero when its denominator is zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _calculate_threshold_metrics_arrays(
    target: npt.NDArray[np.bool_],
    probabilities: npt.NDArray[np.float64],
    threshold: float,
) -> ThresholdMetrics:
    """Calculate classification metrics using prepared arrays."""
    if threshold < 0.0 or threshold > 1.0:
        raise ModelEvaluationError("Classification threshold must be between zero and one.")

    predictions = probabilities >= threshold

    true_positives = int(np.sum(predictions & target))
    false_positives = int(np.sum(predictions & ~target))
    false_negatives = int(np.sum(~predictions & target))
    true_negatives = int(np.sum(~predictions & ~target))

    precision = _safe_divide(
        true_positives,
        true_positives + false_positives,
    )
    recall = _safe_divide(
        true_positives,
        true_positives + false_negatives,
    )
    specificity = _safe_divide(
        true_negatives,
        true_negatives + false_positives,
    )
    accuracy = _safe_divide(
        true_positives + true_negatives,
        len(target),
    )

    positive_f1 = (
        0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    )

    return ThresholdMetrics(
        threshold=float(threshold),
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_positives=true_positives,
        precision=precision,
        recall=recall,
        positive_f1=positive_f1,
        specificity=specificity,
        accuracy=accuracy,
    )


def calculate_threshold_metrics(
    target: pd.Series[bool],
    probabilities: npt.ArrayLike,
    threshold: float,
) -> ThresholdMetrics:
    """Calculate classification metrics at a supplied threshold."""
    target_array, probability_array = _prepare_binary_inputs(
        target,
        probabilities,
    )
    return _calculate_threshold_metrics_arrays(
        target_array,
        probability_array,
        threshold,
    )


def build_threshold_grid(
    step: float = DEFAULT_THRESHOLD_STEP,
) -> npt.NDArray[np.float64]:
    """Create an inclusive zero-to-one threshold grid."""
    if step <= 0.0 or step > 1.0:
        raise ModelEvaluationError("Threshold step must be greater than zero and at most one.")

    threshold_count = int(round(1.0 / step)) + 1
    thresholds = np.linspace(
        0.0,
        1.0,
        threshold_count,
        dtype=np.float64,
    )
    return np.round(thresholds, decimals=12)


def select_validation_threshold(
    target: pd.Series[bool],
    probabilities: npt.ArrayLike,
    *,
    thresholds: npt.ArrayLike | None = None,
) -> ThresholdMetrics:
    """Select the validation threshold using positive-class F1."""
    target_array, probability_array = _prepare_binary_inputs(
        target,
        probabilities,
    )

    threshold_array = (
        build_threshold_grid() if thresholds is None else np.asarray(thresholds, dtype=np.float64)
    )

    if threshold_array.ndim != 1 or len(threshold_array) == 0:
        raise ModelEvaluationError(
            "Threshold candidates must be a non-empty one-dimensional array."
        )

    if not bool(np.isfinite(threshold_array).all()):
        raise ModelEvaluationError("Threshold candidates contain non-finite values.")

    if bool(((threshold_array < 0.0) | (threshold_array > 1.0)).any()):
        raise ModelEvaluationError("Threshold candidates must fall between zero and one.")

    candidates = tuple(
        _calculate_threshold_metrics_arrays(
            target_array,
            probability_array,
            float(threshold),
        )
        for threshold in threshold_array
    )

    return max(
        candidates,
        key=lambda metrics: (
            metrics.positive_f1,
            metrics.recall,
            -metrics.threshold,
        ),
    )


def calculate_top_decile_metrics(
    target: pd.Series[bool],
    probabilities: npt.ArrayLike,
) -> TopDecileMetrics:
    """Calculate recall, precision and lift in the highest-risk decile."""
    target_array, probability_array = _prepare_binary_inputs(
        target,
        probabilities,
    )

    selected_row_count = max(
        1,
        ceil(len(target_array) * TOP_DECILE_FRACTION),
    )
    descending_order = np.argsort(
        -probability_array,
        kind="stable",
    )
    selected_target = target_array[descending_order[:selected_row_count]]

    selected_positive_count = int(selected_target.sum())
    total_positive_count = int(target_array.sum())
    base_positive_rate = total_positive_count / len(target_array)

    recall = selected_positive_count / total_positive_count
    precision = selected_positive_count / selected_row_count
    lift = precision / base_positive_rate

    return TopDecileMetrics(
        selected_row_count=selected_row_count,
        selected_positive_count=selected_positive_count,
        recall=recall,
        precision=precision,
        lift=lift,
    )


def calculate_calibration_bins(
    target: pd.Series[bool],
    probabilities: npt.ArrayLike,
    *,
    bin_count: int = CALIBRATION_BIN_COUNT,
) -> tuple[tuple[CalibrationBin, ...], float, float]:
    """Calculate equal-width calibration bins, ECE and maximum gap."""
    target_array, probability_array = _prepare_binary_inputs(
        target,
        probabilities,
    )

    if bin_count < 2:
        raise ModelEvaluationError("Calibration requires at least two bins.")

    bin_indexes = np.minimum(
        (probability_array * bin_count).astype(np.int64),
        bin_count - 1,
    )

    bins: list[CalibrationBin] = []

    for bin_index in range(bin_count):
        bin_mask = bin_indexes == bin_index
        row_count = int(bin_mask.sum())

        if row_count == 0:
            continue

        bin_probabilities = probability_array[bin_mask]
        bin_target = target_array[bin_mask]

        mean_probability = float(bin_probabilities.mean())
        observed_rate = float(bin_target.mean())
        absolute_gap = abs(mean_probability - observed_rate)

        bins.append(
            CalibrationBin(
                bin_index=bin_index,
                lower_bound=bin_index / bin_count,
                upper_bound=(bin_index + 1) / bin_count,
                row_count=row_count,
                mean_predicted_probability=mean_probability,
                observed_positive_rate=observed_rate,
                absolute_gap=absolute_gap,
            )
        )

    expected_calibration_error = sum(
        calibration_bin.row_count / len(target_array) * calibration_bin.absolute_gap
        for calibration_bin in bins
    )
    maximum_calibration_gap = max(calibration_bin.absolute_gap for calibration_bin in bins)

    return (
        tuple(bins),
        expected_calibration_error,
        maximum_calibration_gap,
    )


def evaluate_validation_probabilities(
    target: pd.Series[bool],
    probabilities: npt.ArrayLike,
) -> ValidationEvaluation:
    """Calculate the complete validation-only evaluation."""
    target_array, probability_array = _prepare_binary_inputs(
        target,
        probabilities,
    )

    precision_values_raw, recall_values_raw, _ = precision_recall_curve(
        target_array,
        probability_array,
        pos_label=True,
    )
    precision_values = np.asarray(
        precision_values_raw,
        dtype=np.float64,
    )
    recall_values = np.asarray(
        recall_values_raw,
        dtype=np.float64,
    )

    selected_threshold = select_validation_threshold(
        target,
        probability_array,
    )
    top_decile = calculate_top_decile_metrics(
        target,
        probability_array,
    )
    (
        calibration_bins,
        expected_calibration_error,
        maximum_calibration_gap,
    ) = calculate_calibration_bins(
        target,
        probability_array,
    )

    probability_matrix = np.column_stack(
        (
            1.0 - probability_array,
            probability_array,
        )
    )

    return ValidationEvaluation(
        row_count=len(target_array),
        positive_count=int(target_array.sum()),
        base_positive_rate=float(target_array.mean()),
        roc_auc=float(
            roc_auc_score(
                target_array,
                probability_array,
            )
        ),
        pr_auc=float(
            auc(
                recall_values,
                precision_values,
            )
        ),
        average_precision=float(
            average_precision_score(
                target_array,
                probability_array,
                pos_label=True,
            )
        ),
        log_loss=float(
            log_loss(
                target_array,
                probability_matrix,
                labels=[False, True],
            )
        ),
        brier_score=float(
            brier_score_loss(
                target_array,
                probability_array,
                pos_label=True,
            )
        ),
        selected_threshold=selected_threshold,
        top_decile=top_decile,
        calibration_bins=calibration_bins,
        expected_calibration_error=expected_calibration_error,
        maximum_calibration_gap=maximum_calibration_gap,
    )


def evaluate_logistic_baseline_validation(
    data: ChronologicalModelingData,
) -> ValidationEvaluation:
    """Fit the logistic baseline and evaluate validation only."""
    baseline_result = fit_logistic_baseline(data)

    return evaluate_validation_probabilities(
        data.validation.target,
        baseline_result.validation_probabilities,
    )
