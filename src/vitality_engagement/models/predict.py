"""Typed prediction interface for the selected Stage 4 model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from vitality_engagement.models.load_data import ChronologicalModelingData
from vitality_engagement.models.persistence import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    load_selected_model,
)
from vitality_engagement.models.schema import (
    IDENTIFIER_COLUMNS,
    MODEL_FEATURE_COLUMNS,
)
from vitality_engagement.models.test_evaluation import (
    predict_positive_probabilities,
)

RISK_PROBABILITY_COLUMN: Final = "risk_probability"
HIGH_RISK_COLUMN: Final = "is_high_risk"
MODEL_NAME_COLUMN: Final = "model_name"
THRESHOLD_COLUMN: Final = "threshold"

PREDICTION_OUTPUT_COLUMNS: Final = (
    *IDENTIFIER_COLUMNS,
    RISK_PROBABILITY_COLUMN,
    HIGH_RISK_COLUMN,
    MODEL_NAME_COLUMN,
    THRESHOLD_COLUMN,
)


class PredictionInputError(ValueError):
    """Raised when prediction input violates the model contract."""


class PredictionOutputError(RuntimeError):
    """Raised when generated predictions violate their output contract."""


@dataclass(frozen=True)
class PredictionBatch:
    """Identifiers and approved features for one prediction batch."""

    identifiers: pd.DataFrame
    features: pd.DataFrame

    def __post_init__(self) -> None:
        """Validate identifier, predictor and row-count contracts."""
        identifier_columns = tuple(str(column) for column in self.identifiers.columns)
        feature_columns = tuple(str(column) for column in self.features.columns)

        if identifier_columns != IDENTIFIER_COLUMNS:
            raise PredictionInputError("Identifier columns do not match the approved contract.")

        if feature_columns != MODEL_FEATURE_COLUMNS:
            raise PredictionInputError("Feature columns do not match the approved model schema.")

        if len(self.identifiers) == 0:
            raise PredictionInputError("Prediction batch must contain at least one row.")

        if len(self.identifiers) != len(self.features):
            raise PredictionInputError("Identifier and feature row counts do not match.")

        if bool(self.identifiers.isna().any().any()):
            raise PredictionInputError("Prediction identifiers contain null values.")

        if bool(self.identifiers.duplicated(subset=list(IDENTIFIER_COLUMNS)).any()):
            raise PredictionInputError("Duplicate member and prediction-date identifiers detected.")


@dataclass(frozen=True)
class PredictionResult:
    """Validated model predictions and frozen decision metadata."""

    predictions: pd.DataFrame
    model_name: str
    threshold: float
    row_count: int

    def __post_init__(self) -> None:
        """Validate probabilities and frozen-threshold classifications."""
        output_columns = tuple(str(column) for column in self.predictions.columns)

        if output_columns != PREDICTION_OUTPUT_COLUMNS:
            raise PredictionOutputError(
                "Prediction output columns do not match the required contract."
            )

        if len(self.predictions) != self.row_count:
            raise PredictionOutputError("Prediction output row count is inconsistent.")

        if not self.model_name:
            raise PredictionOutputError("Prediction model name must not be empty.")

        if self.threshold < 0.0 or self.threshold > 1.0:
            raise PredictionOutputError("Prediction threshold must fall between zero and one.")

        probabilities = self.predictions[RISK_PROBABILITY_COLUMN].to_numpy(dtype=np.float64)

        if not bool(np.isfinite(probabilities).all()):
            raise PredictionOutputError("Prediction probabilities contain non-finite values.")

        if bool(((probabilities < 0.0) | (probabilities > 1.0)).any()):
            raise PredictionOutputError("Prediction probabilities fall outside zero to one.")

        actual_high_risk = self.predictions[HIGH_RISK_COLUMN].to_numpy(dtype=np.bool_)
        expected_high_risk = probabilities >= self.threshold

        if not bool(np.array_equal(actual_high_risk, expected_high_risk)):
            raise PredictionOutputError(
                "High-risk classifications do not use the frozen threshold."
            )

        output_model_names = set(self.predictions[MODEL_NAME_COLUMN].astype(str).tolist())
        if output_model_names != {self.model_name}:
            raise PredictionOutputError("Prediction model-name metadata is inconsistent.")

        output_thresholds = self.predictions[THRESHOLD_COLUMN].to_numpy(dtype=np.float64)
        if not bool(
            np.allclose(
                output_thresholds,
                self.threshold,
                rtol=0.0,
                atol=0.0,
            )
        ):
            raise PredictionOutputError("Prediction threshold metadata is inconsistent.")


def predict_with_pipeline(
    pipeline: Pipeline,
    batch: PredictionBatch,
    *,
    model_name: str,
    threshold: float,
) -> PredictionResult:
    """Generate validated probabilities and classifications."""
    probabilities = predict_positive_probabilities(
        pipeline,
        batch.features,
    )

    predictions = batch.identifiers.reset_index(drop=True).copy()
    predictions[RISK_PROBABILITY_COLUMN] = probabilities
    predictions[HIGH_RISK_COLUMN] = probabilities >= threshold
    predictions[MODEL_NAME_COLUMN] = model_name
    predictions[THRESHOLD_COLUMN] = threshold

    predictions = predictions.loc[
        :,
        list(PREDICTION_OUTPUT_COLUMNS),
    ]

    return PredictionResult(
        predictions=predictions,
        model_name=model_name,
        threshold=threshold,
        row_count=len(predictions),
    )


def score_with_persisted_model(
    batch: PredictionBatch,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> PredictionResult:
    """Load the trusted selected model and score one prediction batch."""
    pipeline, metadata = load_selected_model(
        model_path=model_path,
        metadata_path=metadata_path,
    )

    return predict_with_pipeline(
        pipeline,
        batch,
        model_name=metadata.model_name,
        threshold=metadata.selected_threshold,
    )


def score_operational_split(
    data: ChronologicalModelingData,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> PredictionResult:
    """Score the unlabelled chronological operational split."""
    batch = PredictionBatch(
        identifiers=data.scoring.identifiers,
        features=data.scoring.features,
    )

    return score_with_persisted_model(
        batch,
        model_path=model_path,
        metadata_path=metadata_path,
    )
