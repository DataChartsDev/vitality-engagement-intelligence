"""Tests for fail-closed offline activation orchestration."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from vitality_engagement.activation.artifact import (
    verify_activation_artifact,
)
from vitality_engagement.activation.context_artifact import (
    CONTACT_CONTEXT_ARTIFACT_VERSION,
    CONTACT_CONTEXT_COLUMNS,
    ContactContextArtifactError,
)
from vitality_engagement.activation.orchestrator import (
    ActivationOrchestrationError,
    orchestrate_offline_activation,
)
from vitality_engagement.activation.policy import ActivationPolicy
from vitality_engagement.activation.review_queue import (
    verify_review_queue_artifact,
)
from vitality_engagement.models.predict import (
    HIGH_RISK_COLUMN,
    MODEL_NAME_COLUMN,
    PREDICTION_OUTPUT_COLUMNS,
    RISK_PROBABILITY_COLUMN,
    THRESHOLD_COLUMN,
    PredictionResult,
)
from vitality_engagement.models.scoring_artifact import (
    ScoringArtifactError,
    write_scoring_artifact,
)

DECISION_TIMESTAMP = datetime(
    2025,
    6,
    30,
    8,
    0,
    tzinfo=UTC,
)
SNAPSHOT_TIMESTAMP = datetime(
    2025,
    6,
    30,
    7,
    30,
    tzinfo=UTC,
)


def _prediction_result() -> PredictionResult:
    threshold = 0.431
    probabilities = [0.90, 0.80, 0.70, 0.20]

    predictions = pd.DataFrame(
        {
            "member_id": [
                "member-001",
                "member-002",
                "member-003",
                "member-004",
            ],
            "prediction_date": pd.to_datetime(["2025-06-29"] * 4),
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


def _write_scoring_artifacts(
    tmp_path: Path,
) -> tuple[Path, Path]:
    prediction_path = tmp_path / "predictions.parquet"
    metadata_path = tmp_path / "predictions.metadata.json"

    write_scoring_artifact(
        _prediction_result(),
        prediction_path=prediction_path,
        metadata_path=metadata_path,
    )

    return prediction_path, metadata_path


def _context_frame(
    snapshot_timestamp: datetime,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "member_id": "member-001",
                "contact_allowed": True,
                "opted_out": False,
                "active_case_open": False,
                "last_contacted_at": None,
                "interventions_last_28d": 0,
                "context_as_of": snapshot_timestamp,
            },
            {
                "member_id": "member-002",
                "contact_allowed": False,
                "opted_out": True,
                "active_case_open": False,
                "last_contacted_at": None,
                "interventions_last_28d": 0,
                "context_as_of": snapshot_timestamp,
            },
            {
                "member_id": "member-003",
                "contact_allowed": True,
                "opted_out": False,
                "active_case_open": True,
                "last_contacted_at": None,
                "interventions_last_28d": 0,
                "context_as_of": snapshot_timestamp,
            },
        ],
        columns=list(CONTACT_CONTEXT_COLUMNS),
    )


def _write_context_artifacts(
    tmp_path: Path,
    *,
    snapshot_timestamp: datetime = SNAPSHOT_TIMESTAMP,
) -> tuple[Path, Path]:
    context_path = tmp_path / "contact_context.parquet"
    metadata_path = tmp_path / "contact_context.metadata.json"

    frame = _context_frame(snapshot_timestamp)
    frame.to_parquet(
        context_path,
        index=False,
    )

    metadata = {
        "artifact_version": CONTACT_CONTEXT_ARTIFACT_VERSION,
        "source_name": "approved_contact_governance_view",
        "source_snapshot_reference": ("snapshot-2025-06-30T07:30:00Z"),
        "source_query_sha256": "a" * 64,
        "context_artifact_sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
        "snapshot_timestamp": snapshot_timestamp.isoformat(),
        "row_count": len(frame),
        "member_count": int(frame["member_id"].nunique()),
        "output_columns": list(CONTACT_CONTEXT_COLUMNS),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return context_path, metadata_path


def test_orchestrator_verifies_decides_and_writes_locally(
    tmp_path: Path,
) -> None:
    prediction_path, scoring_metadata_path = _write_scoring_artifacts(tmp_path)
    context_path, context_metadata_path = _write_context_artifacts(tmp_path)
    decision_path = tmp_path / "activation.parquet"
    activation_metadata_path = tmp_path / "activation.metadata.json"
    review_queue_path = tmp_path / "human_review_queue.parquet"
    review_metadata_path = tmp_path / "human_review_queue.metadata.json"

    result = orchestrate_offline_activation(
        context_path=context_path,
        context_metadata_path=context_metadata_path,
        decision_timestamp=DECISION_TIMESTAMP,
        policy=ActivationPolicy(maximum_activations_per_run=1),
        scoring_prediction_path=prediction_path,
        scoring_metadata_path=scoring_metadata_path,
        activation_decision_path=decision_path,
        activation_metadata_path=activation_metadata_path,
        review_queue_path=review_queue_path,
        review_metadata_path=review_metadata_path,
    )

    assert result.decision_path == decision_path
    assert result.metadata_path == activation_metadata_path
    assert result.review_queue_path == review_queue_path
    assert result.review_metadata_path == review_metadata_path
    assert decision_path.is_file()
    assert activation_metadata_path.is_file()
    assert review_queue_path.is_file()
    assert review_metadata_path.is_file()
    assert result.decision_result.metadata.source_row_count == 4
    assert result.decision_result.metadata.selected_count == 1
    assert result.decision_result.metadata.contact_context_lineage.artifact_path == str(
        context_path
    )
    assert (
        result.decision_result.metadata.contact_context_lineage.artifact_sha256
        == result.contact_context_artifact.metadata.context_artifact_sha256
    )

    verify_activation_artifact(
        decision_path,
        activation_metadata_path,
        expected_result=result.decision_result,
    )
    verify_review_queue_artifact(
        review_queue_path,
        review_metadata_path,
        expected_result=result.decision_result,
        activation_decision_path=decision_path,
        activation_metadata_path=activation_metadata_path,
    )


def test_scoring_failure_writes_no_activation_output(
    tmp_path: Path,
) -> None:
    prediction_path, scoring_metadata_path = _write_scoring_artifacts(tmp_path)
    context_path, context_metadata_path = _write_context_artifacts(tmp_path)
    decision_path = tmp_path / "activation.parquet"
    activation_metadata_path = tmp_path / "activation.metadata.json"
    review_queue_path = tmp_path / "human_review_queue.parquet"
    review_metadata_path = tmp_path / "human_review_queue.metadata.json"

    frame = pd.read_parquet(prediction_path)
    frame.loc[0, RISK_PROBABILITY_COLUMN] = 0.75
    frame.to_parquet(
        prediction_path,
        index=False,
    )

    with pytest.raises(ScoringArtifactError):
        orchestrate_offline_activation(
            context_path=context_path,
            context_metadata_path=context_metadata_path,
            decision_timestamp=DECISION_TIMESTAMP,
            scoring_prediction_path=prediction_path,
            scoring_metadata_path=scoring_metadata_path,
            activation_decision_path=decision_path,
            activation_metadata_path=activation_metadata_path,
            review_queue_path=review_queue_path,
            review_metadata_path=review_metadata_path,
        )

    assert not decision_path.exists()
    assert not activation_metadata_path.exists()
    assert not review_queue_path.exists()
    assert not review_metadata_path.exists()


def test_future_context_writes_no_activation_output(
    tmp_path: Path,
) -> None:
    prediction_path, scoring_metadata_path = _write_scoring_artifacts(tmp_path)
    context_path, context_metadata_path = _write_context_artifacts(
        tmp_path,
        snapshot_timestamp=datetime(
            2025,
            6,
            30,
            9,
            0,
            tzinfo=UTC,
        ),
    )
    decision_path = tmp_path / "activation.parquet"
    activation_metadata_path = tmp_path / "activation.metadata.json"
    review_queue_path = tmp_path / "human_review_queue.parquet"
    review_metadata_path = tmp_path / "human_review_queue.metadata.json"

    with pytest.raises(
        ContactContextArtifactError,
        match="must not be from the future",
    ):
        orchestrate_offline_activation(
            context_path=context_path,
            context_metadata_path=context_metadata_path,
            decision_timestamp=DECISION_TIMESTAMP,
            scoring_prediction_path=prediction_path,
            scoring_metadata_path=scoring_metadata_path,
            activation_decision_path=decision_path,
            activation_metadata_path=activation_metadata_path,
            review_queue_path=review_queue_path,
            review_metadata_path=review_metadata_path,
        )

    assert not decision_path.exists()
    assert not activation_metadata_path.exists()
    assert not review_queue_path.exists()
    assert not review_metadata_path.exists()


def test_input_output_path_collision_is_rejected(
    tmp_path: Path,
) -> None:
    prediction_path, scoring_metadata_path = _write_scoring_artifacts(tmp_path)
    context_path, context_metadata_path = _write_context_artifacts(tmp_path)
    original_digest = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    activation_metadata_path = tmp_path / "activation.metadata.json"
    review_metadata_path = tmp_path / "human_review_queue.metadata.json"

    with pytest.raises(
        ActivationOrchestrationError,
        match="must use distinct paths",
    ):
        orchestrate_offline_activation(
            context_path=context_path,
            context_metadata_path=context_metadata_path,
            decision_timestamp=DECISION_TIMESTAMP,
            scoring_prediction_path=prediction_path,
            scoring_metadata_path=scoring_metadata_path,
            activation_decision_path=prediction_path,
            activation_metadata_path=activation_metadata_path,
            review_queue_path=tmp_path / "human_review_queue.parquet",
            review_metadata_path=review_metadata_path,
        )

    assert hashlib.sha256(prediction_path.read_bytes()).hexdigest() == original_digest
    assert not activation_metadata_path.exists()
    assert not review_metadata_path.exists()


def test_review_queue_path_collision_is_rejected_before_outputs(
    tmp_path: Path,
) -> None:
    prediction_path, scoring_metadata_path = _write_scoring_artifacts(tmp_path)
    context_path, context_metadata_path = _write_context_artifacts(tmp_path)
    original_digest = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    decision_path = tmp_path / "activation.parquet"
    activation_metadata_path = tmp_path / "activation.metadata.json"
    review_metadata_path = tmp_path / "human_review_queue.metadata.json"

    with pytest.raises(
        ActivationOrchestrationError,
        match="must use distinct paths",
    ):
        orchestrate_offline_activation(
            context_path=context_path,
            context_metadata_path=context_metadata_path,
            decision_timestamp=DECISION_TIMESTAMP,
            scoring_prediction_path=prediction_path,
            scoring_metadata_path=scoring_metadata_path,
            activation_decision_path=decision_path,
            activation_metadata_path=activation_metadata_path,
            review_queue_path=prediction_path,
            review_metadata_path=review_metadata_path,
        )

    assert hashlib.sha256(prediction_path.read_bytes()).hexdigest() == original_digest
    assert not decision_path.exists()
    assert not activation_metadata_path.exists()
    assert not review_metadata_path.exists()
