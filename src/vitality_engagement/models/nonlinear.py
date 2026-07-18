"""Validation-only histogram gradient-boosting model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from vitality_engagement.models.load_data import ChronologicalModelingData
from vitality_engagement.models.schema import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)

RANDOM_SEED: Final = 42
LEARNING_RATE: Final = 0.05
MAX_ITERATIONS: Final = 200
MAX_LEAF_NODES: Final = 15
MIN_SAMPLES_LEAF: Final = 50
L2_REGULARIZATION: Final = 1.0


class NonlinearModelError(RuntimeError):
    """Raised when the nonlinear model produces invalid output."""


@dataclass(frozen=True)
class NonlinearValidationResult:
    """Fitted nonlinear pipeline and its validation probabilities."""

    pipeline: Pipeline
    validation_probabilities: npt.NDArray[np.float64]
    validation_row_count: int

    def __post_init__(self) -> None:
        """Validate probability dimensions, completeness and range."""
        probabilities = self.validation_probabilities

        if probabilities.ndim != 1:
            raise NonlinearModelError("Validation probabilities must be one-dimensional.")

        if len(probabilities) != self.validation_row_count:
            raise NonlinearModelError(
                "Validation probability count does not match validation rows."
            )

        if not bool(np.isfinite(probabilities).all()):
            raise NonlinearModelError("Validation probabilities contain non-finite values.")

        if bool(((probabilities < 0.0) | (probabilities > 1.0)).any()):
            raise NonlinearModelError("Validation probabilities fall outside zero to one.")


def build_hist_gradient_boosting_pipeline() -> Pipeline:
    """Build the constrained nonlinear validation pipeline."""
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                list(CATEGORICAL_FEATURE_COLUMNS),
            ),
            (
                "numeric",
                numeric_pipeline,
                list(NUMERIC_FEATURE_COLUMNS),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    classifier = HistGradientBoostingClassifier(
        learning_rate=LEARNING_RATE,
        max_iter=MAX_ITERATIONS,
        max_leaf_nodes=MAX_LEAF_NODES,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        l2_regularization=L2_REGULARIZATION,
        early_stopping=False,
        random_state=RANDOM_SEED,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def fit_hist_gradient_boosting(
    data: ChronologicalModelingData,
) -> NonlinearValidationResult:
    """Fit on training rows and predict validation probabilities only."""
    pipeline = build_hist_gradient_boosting_pipeline()

    pipeline.fit(
        data.train.features,
        data.train.target,
    )

    probability_matrix = np.asarray(
        pipeline.predict_proba(data.validation.features),
        dtype=np.float64,
    )

    if probability_matrix.ndim != 2:
        raise NonlinearModelError("Classifier probability output must be two-dimensional.")

    classes = np.asarray(pipeline.classes_)
    positive_matches = np.flatnonzero(np.equal(classes, np.bool_(True)))

    if len(positive_matches) != 1:
        raise NonlinearModelError("Classifier does not expose exactly one positive Boolean class.")

    if probability_matrix.shape != (
        len(data.validation.features),
        len(classes),
    ):
        raise NonlinearModelError("Classifier probability dimensions are inconsistent.")

    positive_class_index = int(positive_matches[0])
    validation_probabilities = probability_matrix[
        :,
        positive_class_index,
    ].copy()

    return NonlinearValidationResult(
        pipeline=pipeline,
        validation_probabilities=validation_probabilities,
        validation_row_count=len(data.validation.features),
    )
