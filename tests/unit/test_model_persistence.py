"""Tests for selected-model persistence and loading."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from vitality_engagement.models.load_data import (
    ChronologicalModelingData,
    build_chronological_modeling_data,
)
from vitality_engagement.models.persistence import (
    ARTIFACT_VERSION,
    ModelPersistenceError,
    build_model_metadata,
    calculate_schema_fingerprint,
    load_selected_model,
    save_selected_model,
)
from vitality_engagement.models.schema import (
    CATEGORICAL_FEATURE_COLUMNS,
    EXPORT_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    SPLIT_COLUMN,
    TARGET_COLUMN,
)
from vitality_engagement.models.selection import (
    PYTHON_LOGISTIC_SELECTION,
)


def make_modeling_data() -> ChronologicalModelingData:
    """Create compact chronological data for persistence tests."""
    split_dates = {
        "train": "2025-04-30",
        "validation": "2025-05-31",
        "test": "2025-06-22",
        "scoring": "2025-06-29",
    }
    rows: list[dict[str, object]] = []

    for split_name, prediction_date in split_dates.items():
        for member_number in range(6):
            row: dict[str, object] = {
                "member_id": f"member-{member_number:03d}",
                "prediction_date": pd.Timestamp(prediction_date),
                SPLIT_COLUMN: split_name,
                TARGET_COLUMN: (None if split_name == "scoring" else bool(member_number % 2)),
            }

            for feature_name in MODEL_FEATURE_COLUMNS:
                if feature_name in CATEGORICAL_FEATURE_COLUMNS:
                    value: object = "category-a" if member_number % 2 == 0 else "category-b"
                else:
                    value = float(member_number + 1)

                row[feature_name] = value

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
            "train": 6,
            "validation": 6,
            "test": 6,
            "scoring": 6,
        },
        expected_member_count=6,
    )


def test_schema_fingerprint_is_stable() -> None:
    first_fingerprint = calculate_schema_fingerprint()
    second_fingerprint = calculate_schema_fingerprint()

    assert first_fingerprint == second_fingerprint
    assert len(first_fingerprint) == 64


def test_metadata_matches_frozen_model_choice() -> None:
    metadata = build_model_metadata()

    assert metadata.artifact_version == ARTIFACT_VERSION
    assert metadata.model_name == PYTHON_LOGISTIC_SELECTION.model_name
    assert metadata.selected_threshold == PYTHON_LOGISTIC_SELECTION.threshold
    assert metadata.schema_fingerprint == (calculate_schema_fingerprint())


def test_model_can_be_saved_and_loaded(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pkl"
    metadata_path = tmp_path / "model.metadata.json"

    save_selected_model(
        make_modeling_data(),
        model_path=model_path,
        metadata_path=metadata_path,
    )

    pipeline, metadata = load_selected_model(
        model_path=model_path,
        metadata_path=metadata_path,
    )

    assert isinstance(pipeline, Pipeline)
    assert metadata.model_name == "python_logistic_baseline"
    assert model_path.is_file()
    assert metadata_path.is_file()


def test_corrupted_metadata_is_rejected(tmp_path: Path) -> None:
    metadata_path = tmp_path / "model.metadata.json"
    metadata_path.write_text(
        '{"artifact_version": 999}',
        encoding="utf-8",
    )

    with pytest.raises(
        ModelPersistenceError,
        match="incomplete or invalid",
    ):
        load_selected_model(
            model_path=tmp_path / "model.pkl",
            metadata_path=metadata_path,
        )


def test_missing_artifact_is_rejected(tmp_path: Path) -> None:
    model_path = tmp_path / "missing.pkl"
    metadata_path = tmp_path / "model.metadata.json"

    save_selected_model(
        make_modeling_data(),
        model_path=tmp_path / "temporary.pkl",
        metadata_path=metadata_path,
    )

    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_selected_model(
            model_path=model_path,
            metadata_path=metadata_path,
        )
