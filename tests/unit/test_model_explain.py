"""Tests for selected logistic-model explainability."""

import math

import pandas as pd
import pytest

from vitality_engagement.models.baseline import (
    build_logistic_baseline_pipeline,
)
from vitality_engagement.models.explain import (
    LogisticExplanationError,
    explain_logistic_pipeline,
)
from vitality_engagement.models.schema import (
    CATEGORICAL_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)


def make_training_data() -> tuple[pd.DataFrame, pd.Series[bool]]:
    """Create compact training data using the complete predictor contract."""
    rows: list[dict[str, object]] = []
    targets: list[bool] = []

    for row_number in range(12):
        target = bool(row_number % 2)
        row: dict[str, object] = {}

        for feature_name in MODEL_FEATURE_COLUMNS:
            if feature_name == "age_band_as_of":
                value: object = "18-29" if target else "40-49"
            elif feature_name == "activity_level_as_of":
                value = "high" if target else "low"
            elif feature_name == "reward_profile_as_of":
                value = "points" if target else "discount"
            else:
                value = float(row_number + (3 if target else 0))

            row[feature_name] = value

        rows.append(row)
        targets.append(target)

    return (
        pd.DataFrame(rows, columns=list(MODEL_FEATURE_COLUMNS)),
        pd.Series(targets, dtype=bool),
    )


def make_fitted_pipeline() -> object:
    """Create a fitted logistic pipeline for explanation tests."""
    features, target = make_training_data()
    pipeline = build_logistic_baseline_pipeline()
    pipeline.fit(features, target)
    return pipeline


def test_explanation_contains_every_transformed_feature() -> None:
    pipeline = make_fitted_pipeline()

    explanation = explain_logistic_pipeline(pipeline)

    assert explanation.model_name == "python_logistic_baseline"
    assert explanation.positive_class is True
    assert explanation.feature_count == len(explanation.effects)
    assert explanation.feature_count > len(MODEL_FEATURE_COLUMNS)


def test_effects_are_sorted_by_absolute_coefficient() -> None:
    pipeline = make_fitted_pipeline()

    explanation = explain_logistic_pipeline(pipeline)
    magnitudes = [effect.absolute_coefficient for effect in explanation.effects]

    assert magnitudes == sorted(magnitudes, reverse=True)
    assert [effect.rank for effect in explanation.effects] == list(
        range(1, len(explanation.effects) + 1)
    )


def test_effects_map_to_approved_source_features() -> None:
    pipeline = make_fitted_pipeline()

    explanation = explain_logistic_pipeline(pipeline)
    approved_sources = set(MODEL_FEATURE_COLUMNS)

    assert {effect.source_feature for effect in explanation.effects}.issubset(approved_sources)

    assert set(NUMERIC_FEATURE_COLUMNS).issubset(
        {effect.transformed_feature for effect in explanation.effects}
    )

    assert set(CATEGORICAL_FEATURE_COLUMNS).issubset(
        {effect.source_feature for effect in explanation.effects}
    )


def test_odds_ratios_and_directions_match_coefficients() -> None:
    pipeline = make_fitted_pipeline()

    explanation = explain_logistic_pipeline(pipeline)

    for effect in explanation.effects:
        assert effect.odds_ratio == pytest.approx(math.exp(effect.coefficient))

        if effect.coefficient > 0.0:
            assert effect.direction == "increases_risk"
        elif effect.coefficient < 0.0:
            assert effect.direction == "decreases_risk"
        else:
            assert effect.direction == "neutral"


def test_unfitted_pipeline_is_rejected() -> None:
    pipeline = build_logistic_baseline_pipeline()

    with pytest.raises(
        LogisticExplanationError,
        match="has not been fitted",
    ):
        explain_logistic_pipeline(pipeline)
