"""Tests for operational scoring artifact persistence."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from vitality_engagement.models.predict import (
    HIGH_RISK_COLUMN,
    MODEL_NAME_COLUMN,
    PREDICTION_OUTPUT_COLUMNS,
    RISK_PROBABILITY_COLUMN,
    THRESHOLD_COLUMN,
    PredictionResult,
)
from vitality_engagement.models.scoring_artifact import (
    SCORING_ARTIFACT_VERSION,
    ScoringArtifactError,
    build_scoring_metadata,
    load_verified_scoring_artifact,
    verify_scoring_artifact,
    write_scoring_artifact,
)


def make_prediction_result() -> PredictionResult:
    """Create a compact valid operational prediction result."""
    threshold = 0.431
    probabilities = [0.10, 0.40, 0.50, 0.90]

    predictions = pd.DataFrame(
        {
            "member_id": [
                "member-001",
                "member-002",
                "member-003",
                "member-004",
            ],
            "prediction_date": pd.to_datetime(
                [
                    "2025-06-23",
                    "2025-06-23",
                    "2025-06-24",
                    "2025-06-24",
                ]
            ),
            RISK_PROBABILITY_COLUMN: probabilities,
            HIGH_RISK_COLUMN: [probability >= threshold for probability in probabilities],
            MODEL_NAME_COLUMN: ["python_logistic_baseline"] * 4,
            THRESHOLD_COLUMN: [threshold] * 4,
        },
        columns=list(PREDICTION_OUTPUT_COLUMNS),
    )

    return PredictionResult(
        predictions=predictions,
        model_name="python_logistic_baseline",
        threshold=threshold,
        row_count=4,
    )


def test_scoring_metadata_summarises_predictions() -> None:
    metadata = build_scoring_metadata(make_prediction_result())

    assert metadata.artifact_version == SCORING_ARTIFACT_VERSION
    assert metadata.model_name == "python_logistic_baseline"
    assert metadata.threshold == pytest.approx(0.431)
    assert metadata.row_count == 4
    assert metadata.member_count == 4
    assert metadata.minimum_prediction_date == "2025-06-23"
    assert metadata.maximum_prediction_date == "2025-06-24"
    assert metadata.high_risk_count == 2
    assert metadata.high_risk_rate == pytest.approx(0.5)
    assert metadata.minimum_probability == pytest.approx(0.1)
    assert metadata.maximum_probability == pytest.approx(0.9)
    assert metadata.mean_probability == pytest.approx(0.475)


def test_scoring_artifacts_are_written_and_verified(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "predictions.parquet"
    metadata_path = tmp_path / "predictions.metadata.json"
    result = make_prediction_result()

    write_scoring_artifact(
        result,
        prediction_path=prediction_path,
        metadata_path=metadata_path,
    )

    assert prediction_path.is_file()
    assert metadata_path.is_file()

    restored = pd.read_parquet(prediction_path)
    assert len(restored) == 4
    assert list(restored.columns) == list(PREDICTION_OUTPUT_COLUMNS)

    verify_scoring_artifact(
        prediction_path,
        expected_result=result,
    )


def test_scoring_artifact_rejects_shared_output_path(
    tmp_path: Path,
) -> None:
    shared_path = tmp_path / "shared-output"

    with pytest.raises(
        ScoringArtifactError,
        match="must differ",
    ):
        write_scoring_artifact(
            make_prediction_result(),
            prediction_path=shared_path,
            metadata_path=shared_path,
        )


def test_verification_rejects_missing_prediction_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        verify_scoring_artifact(
            tmp_path / "missing.parquet",
            expected_result=make_prediction_result(),
        )


def test_persisted_scoring_artifacts_load_independently(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "predictions.parquet"
    metadata_path = tmp_path / "predictions.metadata.json"

    write_scoring_artifact(
        make_prediction_result(),
        prediction_path=prediction_path,
        metadata_path=metadata_path,
    )

    artifact = load_verified_scoring_artifact(
        prediction_path,
        metadata_path,
    )

    assert artifact.result.row_count == 4
    assert artifact.metadata.model_name == ("python_logistic_baseline")
    assert artifact.metadata.threshold == pytest.approx(0.431)
    assert [prediction.member_id for prediction in artifact.predictions] == [
        "member-001",
        "member-002",
        "member-003",
        "member-004",
    ]
    assert artifact.prediction_artifact_sha256 == (
        hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    )


def test_persisted_scoring_metadata_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "predictions.parquet"
    metadata_path = tmp_path / "predictions.metadata.json"

    write_scoring_artifact(
        make_prediction_result(),
        prediction_path=prediction_path,
        metadata_path=metadata_path,
    )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["high_risk_count"] = 3
    payload["high_risk_rate"] = 0.75
    metadata_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ScoringArtifactError,
        match="metadata does not match",
    ):
        load_verified_scoring_artifact(
            prediction_path,
            metadata_path,
        )


def test_persisted_scoring_prediction_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "predictions.parquet"
    metadata_path = tmp_path / "predictions.metadata.json"

    write_scoring_artifact(
        make_prediction_result(),
        prediction_path=prediction_path,
        metadata_path=metadata_path,
    )

    restored = pd.read_parquet(prediction_path)
    restored.loc[0, RISK_PROBABILITY_COLUMN] = 0.8
    restored.to_parquet(
        prediction_path,
        index=False,
    )

    with pytest.raises(
        ScoringArtifactError,
        match="violate the output contract",
    ):
        load_verified_scoring_artifact(
            prediction_path,
            metadata_path,
        )


def test_persisted_scoring_metadata_rejects_extra_fields(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "predictions.parquet"
    metadata_path = tmp_path / "predictions.metadata.json"

    write_scoring_artifact(
        make_prediction_result(),
        prediction_path=prediction_path,
        metadata_path=metadata_path,
    )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["unapproved_field"] = "not allowed"
    metadata_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ScoringArtifactError,
        match="fields do not match",
    ):
        load_verified_scoring_artifact(
            prediction_path,
            metadata_path,
        )


def test_persisted_scoring_requires_boolean_classifications(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "predictions.parquet"
    metadata_path = tmp_path / "predictions.metadata.json"

    write_scoring_artifact(
        make_prediction_result(),
        prediction_path=prediction_path,
        metadata_path=metadata_path,
    )

    restored = pd.read_parquet(prediction_path)
    restored[HIGH_RISK_COLUMN] = restored[HIGH_RISK_COLUMN].astype(str)
    restored.to_parquet(
        prediction_path,
        index=False,
    )

    with pytest.raises(
        ScoringArtifactError,
        match="must be Boolean",
    ):
        load_verified_scoring_artifact(
            prediction_path,
            metadata_path,
        )
