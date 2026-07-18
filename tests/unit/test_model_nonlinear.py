"""Tests for the validation-only histogram gradient-boosting model."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier

from vitality_engagement.models.load_data import (
    ChronologicalModelingData,
    build_chronological_modeling_data,
)
from vitality_engagement.models.nonlinear import (
    L2_REGULARIZATION,
    LEARNING_RATE,
    MAX_ITERATIONS,
    MAX_LEAF_NODES,
    MIN_SAMPLES_LEAF,
    RANDOM_SEED,
    build_hist_gradient_boosting_pipeline,
    fit_hist_gradient_boosting,
)
from vitality_engagement.models.schema import (
    CATEGORICAL_FEATURE_COLUMNS,
    EXPORT_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    SPLIT_COLUMN,
    TARGET_COLUMN,
)


def make_modeling_data() -> ChronologicalModelingData:
    """Create compact chronological data for nonlinear model tests."""
    split_dates = {
        "train": "2025-04-30",
        "validation": "2025-05-31",
        "test": "2025-06-22",
        "scoring": "2025-06-29",
    }
    rows: list[dict[str, object]] = []

    for split_name, prediction_date in split_dates.items():
        for member_number in range(8):
            row: dict[str, object] = {
                "member_id": f"member-{member_number:03d}",
                "prediction_date": pd.Timestamp(prediction_date),
                SPLIT_COLUMN: split_name,
                TARGET_COLUMN: (None if split_name == "scoring" else bool(member_number % 2)),
            }

            for feature_name in MODEL_FEATURE_COLUMNS:
                if feature_name == "age_band_as_of":
                    value: object = (
                        "unseen-validation-age"
                        if split_name == "validation"
                        else ("18-29" if member_number % 2 == 0 else "30-39")
                    )
                elif feature_name == "activity_level_as_of":
                    value = "low" if member_number % 2 == 0 else "high"
                elif feature_name == "reward_profile_as_of":
                    value = "points" if member_number % 2 == 0 else "discount"
                else:
                    value = float(member_number + (2 if member_number % 2 else 0))

                row[feature_name] = value

            if split_name == "train" and member_number == 0:
                row["avg_sleep_hours_28d"] = None

            rows.append(row)

    frame = (
        pd.DataFrame(rows, columns=list(EXPORT_COLUMNS))
        .sort_values(
            ["prediction_date", "member_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return build_chronological_modeling_data(
        frame,
        expected_split_row_counts={
            "train": 8,
            "validation": 8,
            "test": 8,
            "scoring": 8,
        },
        expected_member_count=8,
    )


def test_pipeline_uses_approved_feature_groups() -> None:
    pipeline = build_hist_gradient_boosting_pipeline()

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    assert isinstance(preprocessor, ColumnTransformer)
    assert isinstance(classifier, HistGradientBoostingClassifier)

    transformers = {name: columns for name, _, columns in preprocessor.transformers}

    assert transformers["categorical"] == list(CATEGORICAL_FEATURE_COLUMNS)
    assert transformers["numeric"] == list(NUMERIC_FEATURE_COLUMNS)
    assert set(MODEL_FEATURE_COLUMNS) == (
        set(CATEGORICAL_FEATURE_COLUMNS) | set(NUMERIC_FEATURE_COLUMNS)
    )


def test_classifier_has_constrained_reproducible_configuration() -> None:
    pipeline = build_hist_gradient_boosting_pipeline()
    classifier = pipeline.named_steps["classifier"]

    assert isinstance(classifier, HistGradientBoostingClassifier)
    assert classifier.learning_rate == LEARNING_RATE
    assert classifier.max_iter == MAX_ITERATIONS
    assert classifier.max_leaf_nodes == MAX_LEAF_NODES
    assert classifier.min_samples_leaf == MIN_SAMPLES_LEAF
    assert classifier.l2_regularization == L2_REGULARIZATION
    assert classifier.early_stopping is False
    assert classifier.random_state == RANDOM_SEED


def test_fit_returns_valid_validation_probabilities() -> None:
    data = make_modeling_data()

    result = fit_hist_gradient_boosting(data)

    assert result.validation_probabilities.shape == (8,)
    assert np.isfinite(result.validation_probabilities).all()
    assert (result.validation_probabilities >= 0.0).all()
    assert (result.validation_probabilities <= 1.0).all()


def test_unseen_validation_categories_are_supported() -> None:
    data = make_modeling_data()

    assert set(data.validation.features["age_band_as_of"]) == {"unseen-validation-age"}

    result = fit_hist_gradient_boosting(data)

    assert len(result.validation_probabilities) == 8


def test_test_split_is_not_accessed_during_validation_fit() -> None:
    data = make_modeling_data()

    data.test.features["age_band_as_of"] = pd.Series(
        [["deliberately-invalid-test-value"]] * len(data.test.features),
        dtype=object,
    )

    result = fit_hist_gradient_boosting(data)

    assert len(result.validation_probabilities) == 8
