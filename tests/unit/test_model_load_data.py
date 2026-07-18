"""Tests for chronological Python modelling-data loading."""

from pathlib import Path

import pandas as pd
import pytest

from vitality_engagement.models.load_data import (
    ModelDataValidationError,
    build_chronological_modeling_data,
    load_chronological_modeling_data,
)
from vitality_engagement.models.schema import (
    CATEGORICAL_FEATURE_COLUMNS,
    EXPORT_COLUMNS,
    IDENTIFIER_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    SPLIT_COLUMN,
    TARGET_COLUMN,
)


def make_valid_modeling_frame() -> pd.DataFrame:
    """Create a small frame following the production modelling contract."""
    split_dates = {
        "train": "2025-04-30",
        "validation": "2025-05-31",
        "test": "2025-06-22",
        "scoring": "2025-06-29",
    }
    rows: list[dict[str, object]] = []

    for split_name, prediction_date in split_dates.items():
        for member_number in range(2):
            row: dict[str, object] = {
                "member_id": f"member-{member_number:03d}",
                "prediction_date": pd.Timestamp(prediction_date),
                SPLIT_COLUMN: split_name,
                TARGET_COLUMN: (None if split_name == "scoring" else bool(member_number)),
            }

            for feature_name in MODEL_FEATURE_COLUMNS:
                row[feature_name] = (
                    "category"
                    if feature_name in CATEGORICAL_FEATURE_COLUMNS
                    else float(member_number + 1)
                )

            rows.append(row)

    return (
        pd.DataFrame(rows, columns=list(EXPORT_COLUMNS))
        .sort_values(
            ["prediction_date", "member_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def expected_small_split_counts() -> dict[str, int]:
    """Return split counts used by the compact test dataset."""
    return {
        "train": 2,
        "validation": 2,
        "test": 2,
        "scoring": 2,
    }


def build_small_modeling_data() -> object:
    """Build chronological data using compact test expectations."""
    return build_chronological_modeling_data(
        make_valid_modeling_frame(),
        expected_split_row_counts=expected_small_split_counts(),
        expected_member_count=2,
    )


def test_build_chronological_data_has_expected_split_sizes() -> None:
    data = build_chronological_modeling_data(
        make_valid_modeling_frame(),
        expected_split_row_counts=expected_small_split_counts(),
        expected_member_count=2,
    )

    assert len(data.train.features) == 2
    assert len(data.validation.features) == 2
    assert len(data.test.features) == 2
    assert len(data.scoring.features) == 2


def test_loader_keeps_identifiers_and_target_out_of_features() -> None:
    data = build_chronological_modeling_data(
        make_valid_modeling_frame(),
        expected_split_row_counts=expected_small_split_counts(),
        expected_member_count=2,
    )

    assert list(data.train.identifiers.columns) == list(IDENTIFIER_COLUMNS)
    assert list(data.train.features.columns) == list(MODEL_FEATURE_COLUMNS)
    assert TARGET_COLUMN not in data.train.features.columns
    assert SPLIT_COLUMN not in data.train.features.columns
    assert set(IDENTIFIER_COLUMNS).isdisjoint(data.train.features.columns)


def test_labelled_targets_are_boolean() -> None:
    data = build_chronological_modeling_data(
        make_valid_modeling_frame(),
        expected_split_row_counts=expected_small_split_counts(),
        expected_member_count=2,
    )

    assert data.train.target.dtype == bool
    assert data.train.target.tolist() == [False, True]
    assert data.validation.target.tolist() == [False, True]
    assert data.test.target.tolist() == [False, True]


def test_scoring_split_has_no_target_attribute() -> None:
    data = build_chronological_modeling_data(
        make_valid_modeling_frame(),
        expected_split_row_counts=expected_small_split_counts(),
        expected_member_count=2,
    )

    assert not hasattr(data.scoring, "target")


def test_split_blocks_must_follow_chronological_order() -> None:
    frame = make_valid_modeling_frame()

    train_index = frame.index[frame[SPLIT_COLUMN] == "train"][0]
    validation_index = frame.index[frame[SPLIT_COLUMN] == "validation"][0]

    frame.loc[train_index, SPLIT_COLUMN] = "validation"
    frame.loc[validation_index, SPLIT_COLUMN] = "train"

    with pytest.raises(
        ModelDataValidationError,
        match="chronological order",
    ):
        build_chronological_modeling_data(
            frame,
            expected_split_row_counts=expected_small_split_counts(),
            expected_member_count=2,
        )


def test_load_chronological_data_from_parquet(tmp_path: Path) -> None:
    output_path = tmp_path / "modeling.parquet"
    make_valid_modeling_frame().to_parquet(output_path, index=False)

    data = load_chronological_modeling_data(
        output_path,
        expected_split_row_counts=expected_small_split_counts(),
        expected_member_count=2,
    )

    assert len(data.train.target) == 2
    assert len(data.scoring.features) == 2


def test_loader_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_chronological_modeling_data(missing_path)
