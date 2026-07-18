"""Untouched test evaluation using a validation-frozen threshold."""

from __future__ import annotations

from dataclasses import dataclass

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
from sklearn.pipeline import Pipeline

from vitality_engagement.models.baseline import (
    LogisticBaselineError,
    fit_logistic_baseline,
)
from vitality_engagement.models.evaluation import (
    CalibrationBin,
    ThresholdMetrics,
    TopDecileMetrics,
    calculate_calibration_bins,
    calculate_threshold_metrics,
    calculate_top_decile_metrics,
)
from vitality_engagement.models.load_data import ChronologicalModelingData
from vitality_engagement.models.selection import (
    PYTHON_LOGISTIC_SELECTION,
    FrozenModelSelection,
)


@dataclass(frozen=True)
class TestEvaluation:
    """Complete untouched-test evaluation at a frozen threshold."""

    row_count: int
    positive_count: int
    base_positive_rate: float
    roc_auc: float
    pr_auc: float
    average_precision: float
    logarithmic_loss: float
    brier_score: float
    frozen_threshold: ThresholdMetrics
    top_decile: TopDecileMetrics
    calibration_bins: tuple[CalibrationBin, ...]
    expected_calibration_error: float
    maximum_calibration_gap: float


def predict_positive_probabilities(
    pipeline: Pipeline,
    features: pd.DataFrame,
) -> npt.NDArray[np.float64]:
    """Return positive-class probabilities from a fitted binary pipeline."""
    probability_matrix = np.asarray(
        pipeline.predict_proba(features),
        dtype=np.float64,
    )

    if probability_matrix.ndim != 2:
        raise LogisticBaselineError("Classifier probability output must be two-dimensional.")

    classes = np.asarray(pipeline.classes_)
    positive_matches = np.flatnonzero(np.equal(classes, np.bool_(True)))

    if len(positive_matches) != 1:
        raise LogisticBaselineError(
            "Classifier does not expose exactly one positive Boolean class."
        )

    if probability_matrix.shape != (
        len(features),
        len(classes),
    ):
        raise LogisticBaselineError("Classifier probability dimensions are inconsistent.")

    positive_class_index = int(positive_matches[0])
    probabilities = probability_matrix[:, positive_class_index].copy()

    if not bool(np.isfinite(probabilities).all()):
        raise LogisticBaselineError("Test probabilities contain non-finite values.")

    if bool(((probabilities < 0.0) | (probabilities > 1.0)).any()):
        raise LogisticBaselineError("Test probabilities fall outside zero to one.")

    return probabilities


def evaluate_frozen_threshold_probabilities(
    target: pd.Series[bool],
    probabilities: npt.ArrayLike,
    *,
    selection: FrozenModelSelection = PYTHON_LOGISTIC_SELECTION,
) -> TestEvaluation:
    """Evaluate probabilities without selecting or changing the threshold."""
    probability_array = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    frozen_threshold = calculate_threshold_metrics(
        target,
        probability_array,
        threshold=selection.threshold,
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

    target_array = target.to_numpy(dtype=np.bool_)

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

    probability_matrix = np.column_stack(
        (
            1.0 - probability_array,
            probability_array,
        )
    )

    return TestEvaluation(
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
        logarithmic_loss=float(
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
        frozen_threshold=frozen_threshold,
        top_decile=top_decile,
        calibration_bins=calibration_bins,
        expected_calibration_error=expected_calibration_error,
        maximum_calibration_gap=maximum_calibration_gap,
    )


def evaluate_logistic_baseline_test(
    data: ChronologicalModelingData,
    *,
    selection: FrozenModelSelection = PYTHON_LOGISTIC_SELECTION,
) -> TestEvaluation:
    """Fit on training rows and evaluate the untouched test split once."""
    baseline_result = fit_logistic_baseline(data)
    test_probabilities = predict_positive_probabilities(
        baseline_result.pipeline,
        data.test.features,
    )

    return evaluate_frozen_threshold_probabilities(
        data.test.target,
        test_probabilities,
        selection=selection,
    )
