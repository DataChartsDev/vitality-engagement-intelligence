"""Global explainability for the selected Python logistic model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from vitality_engagement.models.baseline import fit_logistic_baseline
from vitality_engagement.models.load_data import ChronologicalModelingData
from vitality_engagement.models.schema import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)

RiskDirection = Literal[
    "increases_risk",
    "decreases_risk",
    "neutral",
]


class LogisticExplanationError(ValueError):
    """Raised when a fitted logistic pipeline cannot be explained."""


@dataclass(frozen=True)
class LogisticFeatureEffect:
    """One transformed feature's global logistic-regression effect."""

    rank: int
    transformed_feature: str
    source_feature: str
    coefficient: float
    absolute_coefficient: float
    odds_ratio: float
    direction: RiskDirection


@dataclass(frozen=True)
class LogisticModelExplanation:
    """Global explanation of a fitted binary logistic pipeline."""

    model_name: str
    positive_class: bool
    feature_count: int
    intercept: float
    intercept_odds: float
    effects: tuple[LogisticFeatureEffect, ...]


def _resolve_source_feature(transformed_feature: str) -> str:
    """Map a transformed feature name back to its source predictor."""
    if transformed_feature in NUMERIC_FEATURE_COLUMNS:
        return transformed_feature

    for source_feature in CATEGORICAL_FEATURE_COLUMNS:
        prefix = f"{source_feature}_"
        if transformed_feature.startswith(prefix):
            return source_feature

    raise LogisticExplanationError(
        f"Could not map transformed feature to an approved source predictor: {transformed_feature}"
    )


def _risk_direction(coefficient: float) -> RiskDirection:
    """Return the direction associated with a logistic coefficient."""
    if coefficient > 0.0:
        return "increases_risk"

    if coefficient < 0.0:
        return "decreases_risk"

    return "neutral"


def _calculate_odds_ratio(coefficient: float) -> float:
    """Convert a log-odds coefficient into an odds ratio."""
    try:
        odds_ratio = math.exp(coefficient)
    except OverflowError as error:
        raise LogisticExplanationError(
            "A logistic coefficient is too large to convert to an odds ratio."
        ) from error

    if not math.isfinite(odds_ratio):
        raise LogisticExplanationError("A logistic coefficient produced a non-finite odds ratio.")

    return odds_ratio


def explain_logistic_pipeline(
    pipeline: Pipeline,
) -> LogisticModelExplanation:
    """Extract coefficients and odds ratios from a fitted pipeline."""
    preprocessor = pipeline.named_steps.get("preprocessor")
    classifier = pipeline.named_steps.get("classifier")

    if not isinstance(preprocessor, ColumnTransformer):
        raise LogisticExplanationError(
            "Pipeline does not contain the expected column preprocessor."
        )

    if not isinstance(classifier, LogisticRegression):
        raise LogisticExplanationError(
            "Pipeline does not contain the expected logistic classifier."
        )

    if not hasattr(preprocessor, "transformers_"):
        raise LogisticExplanationError("Column preprocessor has not been fitted.")

    if not hasattr(classifier, "coef_"):
        raise LogisticExplanationError("Logistic classifier has not been fitted.")

    transformed_features = np.asarray(
        preprocessor.get_feature_names_out(),
        dtype=str,
    )
    coefficient_matrix = np.asarray(
        classifier.coef_,
        dtype=np.float64,
    )
    intercept_values = np.asarray(
        classifier.intercept_,
        dtype=np.float64,
    )
    classes = np.asarray(classifier.classes_)

    if coefficient_matrix.shape != (1, len(transformed_features)):
        raise LogisticExplanationError(
            "Logistic coefficient dimensions do not match transformed features."
        )

    if intercept_values.shape != (1,):
        raise LogisticExplanationError("Binary logistic model must expose exactly one intercept.")

    if classes.shape != (2,) or set(classes.tolist()) != {False, True}:
        raise LogisticExplanationError(
            "Logistic model must expose Boolean negative and positive classes."
        )

    coefficients = coefficient_matrix[0]

    if not bool(np.isfinite(coefficients).all()):
        raise LogisticExplanationError("Logistic coefficients contain non-finite values.")

    intercept = float(intercept_values[0])
    if not math.isfinite(intercept):
        raise LogisticExplanationError("Logistic intercept is not finite.")

    unsorted_effects: list[tuple[str, str, float, float, float, RiskDirection]] = []

    for transformed_feature, raw_coefficient in zip(
        transformed_features,
        coefficients,
        strict=True,
    ):
        coefficient = float(raw_coefficient)
        unsorted_effects.append(
            (
                str(transformed_feature),
                _resolve_source_feature(str(transformed_feature)),
                coefficient,
                abs(coefficient),
                _calculate_odds_ratio(coefficient),
                _risk_direction(coefficient),
            )
        )

    unsorted_effects.sort(
        key=lambda effect: (
            -effect[3],
            effect[0],
        )
    )

    effects = tuple(
        LogisticFeatureEffect(
            rank=rank,
            transformed_feature=transformed_feature,
            source_feature=source_feature,
            coefficient=coefficient,
            absolute_coefficient=absolute_coefficient,
            odds_ratio=odds_ratio,
            direction=direction,
        )
        for rank, (
            transformed_feature,
            source_feature,
            coefficient,
            absolute_coefficient,
            odds_ratio,
            direction,
        ) in enumerate(
            unsorted_effects,
            start=1,
        )
    )

    return LogisticModelExplanation(
        model_name="python_logistic_baseline",
        positive_class=True,
        feature_count=len(effects),
        intercept=intercept,
        intercept_odds=_calculate_odds_ratio(intercept),
        effects=effects,
    )


def explain_selected_logistic_model(
    data: ChronologicalModelingData,
) -> LogisticModelExplanation:
    """Fit the selected model on training data and explain it globally."""
    fitted_result = fit_logistic_baseline(data)
    return explain_logistic_pipeline(fitted_result.pipeline)
