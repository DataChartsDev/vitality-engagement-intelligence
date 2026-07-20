"""Tests for verified activation decision artifacts."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from vitality_engagement.activation.artifact import (
    ACTIVATION_DECISION_COLUMNS,
    ActivationArtifactError,
    build_activation_artifact_metadata,
    build_activation_decision_frame,
    verify_activation_artifact,
    write_activation_artifact,
)
from vitality_engagement.activation.engine import (
    ActivationDecisionResult,
    decide_activations,
)
from vitality_engagement.activation.policy import ActivationPolicy
from vitality_engagement.activation.schema import (
    ContactContextLineage,
    DecisionOutcome,
    MemberActivationContext,
    ScoredPrediction,
)

DECISION_TIMESTAMP = datetime(2025, 6, 30, 8, 0, tzinfo=UTC)
SCORING_DIGEST = "a" * 64
SCORING_PATH = "artifacts/scoring/python_logistic_scoring_predictions.parquet"
CONTACT_CONTEXT_LINEAGE = ContactContextLineage(
    artifact_path="artifacts/activation/contact_context.parquet",
    artifact_sha256="b" * 64,
    source_name="approved_contact_context_snapshot",
    source_snapshot_reference=("snapshot-2025-06-30T07:30:00Z"),
    source_query_sha256="c" * 64,
    snapshot_timestamp=datetime(
        2025,
        6,
        30,
        7,
        30,
        tzinfo=UTC,
    ),
)


def _prediction(
    member_id: str,
    *,
    prediction_date: date = date(2025, 6, 29),
    probability: float = 0.8,
) -> ScoredPrediction:
    return ScoredPrediction(
        member_id=member_id,
        prediction_date=prediction_date,
        risk_probability=probability,
        is_high_risk=probability >= 0.431,
        model_name="python_logistic_baseline",
        threshold=0.431,
    )


def _context(
    member_id: str,
    *,
    contact_allowed: bool = True,
) -> MemberActivationContext:
    return MemberActivationContext(
        member_id=member_id,
        contact_allowed=contact_allowed,
        opted_out=False,
        active_case_open=False,
        last_contacted_at=None,
        interventions_last_28d=0,
    )


def _result() -> ActivationDecisionResult:
    return decide_activations(
        predictions=[
            _prediction(
                "member-a",
                prediction_date=date(2025, 6, 28),
                probability=0.95,
            ),
            _prediction(
                "member-a",
                prediction_date=date(2025, 6, 29),
                probability=0.90,
            ),
            _prediction("member-b", probability=0.80),
            _prediction("member-c", probability=0.70),
            _prediction("member-d", probability=0.20),
        ],
        contexts=[
            _context("member-a"),
            _context("member-b", contact_allowed=False),
            _context("member-c"),
        ],
        policy=ActivationPolicy(maximum_activations_per_run=1),
        decision_timestamp=DECISION_TIMESTAMP,
        scoring_artifact_path=SCORING_PATH,
        scoring_artifact_sha256=SCORING_DIGEST,
        contact_context_lineage=CONTACT_CONTEXT_LINEAGE,
    )


def test_activation_decision_frame_matches_audit_contract() -> None:
    result = _result()
    frame = build_activation_decision_frame(result)

    assert tuple(str(column) for column in frame.columns) == (ACTIVATION_DECISION_COLUMNS)
    assert len(frame) == result.metadata.source_row_count
    assert not frame.duplicated(subset=["member_id", "prediction_date"]).any()

    outcome_counts = frame["outcome"].value_counts().to_dict()

    assert outcome_counts[DecisionOutcome.NO_CONTACT_SUPERSEDED.value] == 1
    assert outcome_counts[DecisionOutcome.NO_CONTACT_BELOW_THRESHOLD.value] == 1
    assert outcome_counts[DecisionOutcome.NO_CONTACT_EXCLUDED.value] == 1
    assert outcome_counts[DecisionOutcome.NO_CONTACT_CAPACITY.value] == 1
    assert outcome_counts[DecisionOutcome.SELECTED_FOR_REVIEW.value] == 1


def test_selected_and_no_contact_fields_remain_separated() -> None:
    frame = build_activation_decision_frame(_result())
    selected_mask = frame["outcome"] == DecisionOutcome.SELECTED_FOR_REVIEW.value

    assert frame.loc[selected_mask, "intervention_category"].notna().all()
    assert frame.loc[selected_mask, "priority_rank"].notna().all()
    assert frame.loc[~selected_mask, "intervention_category"].isna().all()
    assert frame.loc[~selected_mask, "priority_rank"].isna().all()


def test_activation_metadata_preserves_run_lineage() -> None:
    result = _result()
    metadata = build_activation_artifact_metadata(result)

    assert metadata.run_id == result.metadata.run_id
    assert metadata.policy_fingerprint == result.metadata.policy_fingerprint
    assert metadata.artifact_version == 2
    assert metadata.scoring_artifact_sha256 == SCORING_DIGEST
    assert metadata.contact_context_artifact_path == CONTACT_CONTEXT_LINEAGE.artifact_path
    assert metadata.contact_context_artifact_sha256 == CONTACT_CONTEXT_LINEAGE.artifact_sha256
    assert metadata.contact_context_source_name == CONTACT_CONTEXT_LINEAGE.source_name
    assert (
        metadata.contact_context_source_snapshot_reference
        == CONTACT_CONTEXT_LINEAGE.source_snapshot_reference
    )
    assert (
        metadata.contact_context_source_query_sha256 == CONTACT_CONTEXT_LINEAGE.source_query_sha256
    )
    assert metadata.contact_context_snapshot_timestamp == (
        CONTACT_CONTEXT_LINEAGE.snapshot_timestamp.isoformat()
    )
    assert metadata.source_row_count == 5
    assert metadata.source_member_count == 4
    assert metadata.selected_count == 1
    assert metadata.output_columns == ACTIVATION_DECISION_COLUMNS


def test_write_and_verify_activation_artifact(tmp_path: Path) -> None:
    result = _result()
    decision_path = tmp_path / "activation_decisions.parquet"
    metadata_path = tmp_path / "activation_decisions.metadata.json"

    written_decisions, written_metadata = write_activation_artifact(
        result,
        decision_path=decision_path,
        metadata_path=metadata_path,
    )

    assert written_decisions == decision_path
    assert written_metadata == metadata_path
    assert decision_path.is_file()
    assert metadata_path.is_file()

    verify_activation_artifact(
        decision_path,
        metadata_path,
        expected_result=result,
    )

    assert not (decision_path.parent / f".{decision_path.name}.tmp").exists()
    assert not (metadata_path.parent / f".{metadata_path.name}.tmp").exists()


def test_writer_rejects_identical_output_paths(tmp_path: Path) -> None:
    output_path = tmp_path / "activation.parquet"

    with pytest.raises(ActivationArtifactError, match="paths must differ"):
        write_activation_artifact(
            _result(),
            decision_path=output_path,
            metadata_path=output_path,
        )


def test_verifier_detects_duplicate_identifiers(tmp_path: Path) -> None:
    result = _result()
    decision_path = tmp_path / "activation_decisions.parquet"
    metadata_path = tmp_path / "activation_decisions.metadata.json"

    write_activation_artifact(
        result,
        decision_path=decision_path,
        metadata_path=metadata_path,
    )

    restored = pd.read_parquet(decision_path)
    restored.at[1, "member_id"] = restored.at[0, "member_id"]
    restored.at[1, "prediction_date"] = restored.at[0, "prediction_date"]
    restored.to_parquet(decision_path, index=False)

    with pytest.raises(
        ActivationArtifactError,
        match="duplicate identifiers",
    ):
        verify_activation_artifact(
            decision_path,
            metadata_path,
            expected_result=result,
        )


def test_verifier_detects_metadata_tampering(tmp_path: Path) -> None:
    result = _result()
    decision_path = tmp_path / "activation_decisions.parquet"
    metadata_path = tmp_path / "activation_decisions.metadata.json"

    write_activation_artifact(
        result,
        decision_path=decision_path,
        metadata_path=metadata_path,
    )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["selected_count"] = 99
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ActivationArtifactError,
        match="metadata does not match",
    ):
        verify_activation_artifact(
            decision_path,
            metadata_path,
            expected_result=result,
        )
