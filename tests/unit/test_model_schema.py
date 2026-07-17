"""Tests for the leakage-safe Python modelling schema."""

from vitality_engagement.models.schema import (
    CATEGORICAL_FEATURE_COLUMNS,
    EXPECTED_MEMBER_COUNT,
    EXPECTED_SPLIT_ROW_COUNTS,
    EXPECTED_TOTAL_ROW_COUNT,
    EXPORT_COLUMNS,
    IDENTIFIER_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    PROHIBITED_PREDICTOR_COLUMNS,
    SOURCE_TARGET_COLUMN,
    SPLIT_COLUMN,
    TARGET_COLUMN,
)


def test_model_schema_has_expected_feature_count() -> None:
    assert len(MODEL_FEATURE_COLUMNS) == 47
    assert len(CATEGORICAL_FEATURE_COLUMNS) == 3
    assert len(NUMERIC_FEATURE_COLUMNS) == 44


def test_model_feature_names_are_unique() -> None:
    assert len(MODEL_FEATURE_COLUMNS) == len(set(MODEL_FEATURE_COLUMNS))
    assert len(EXPORT_COLUMNS) == len(set(EXPORT_COLUMNS))


def test_categorical_and_numeric_features_partition_model_features() -> None:
    categorical = set(CATEGORICAL_FEATURE_COLUMNS)
    numeric = set(NUMERIC_FEATURE_COLUMNS)

    assert categorical.isdisjoint(numeric)
    assert categorical | numeric == set(MODEL_FEATURE_COLUMNS)


def test_prohibited_columns_are_not_predictors() -> None:
    assert set(MODEL_FEATURE_COLUMNS).isdisjoint(PROHIBITED_PREDICTOR_COLUMNS)


def test_export_schema_contains_audit_split_target_and_features() -> None:
    assert EXPORT_COLUMNS[:2] == IDENTIFIER_COLUMNS
    assert EXPORT_COLUMNS[2] == SPLIT_COLUMN
    assert EXPORT_COLUMNS[3] == TARGET_COLUMN
    assert EXPORT_COLUMNS[4:] == MODEL_FEATURE_COLUMNS
    assert len(EXPORT_COLUMNS) == 51


def test_target_names_are_explicit() -> None:
    assert SOURCE_TARGET_COLUMN == "label_will_miss_goal_next_7_days"
    assert TARGET_COLUMN == "will_miss_goal_next_7_days"


def test_expected_split_counts_match_stage_three() -> None:
    assert EXPECTED_SPLIT_ROW_COUNTS == {
        "train": 46_000,
        "validation": 15_500,
        "test": 11_000,
        "scoring": 3_500,
    }
    assert EXPECTED_TOTAL_ROW_COUNT == 76_000
    assert EXPECTED_MEMBER_COUNT == 500
