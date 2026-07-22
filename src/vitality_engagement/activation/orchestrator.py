"""Fail-closed offline orchestration for governed activation decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from vitality_engagement.activation.artifact import (
    DEFAULT_ACTIVATION_DECISION_PATH,
    DEFAULT_ACTIVATION_METADATA_PATH,
    write_activation_artifact,
)
from vitality_engagement.activation.context_artifact import (
    VerifiedContactContextArtifact,
    load_verified_contact_context_artifact,
)
from vitality_engagement.activation.engine import (
    ActivationDecisionResult,
    decide_activations,
)
from vitality_engagement.activation.policy import ActivationPolicy
from vitality_engagement.activation.review_queue import (
    DEFAULT_REVIEW_QUEUE_METADATA_PATH,
    DEFAULT_REVIEW_QUEUE_PATH,
    write_review_queue_artifact,
)
from vitality_engagement.activation.schema import ContactContextLineage
from vitality_engagement.models.scoring_artifact import (
    DEFAULT_PREDICTION_PATH,
    DEFAULT_SCORING_METADATA_PATH,
    VerifiedScoringArtifact,
    load_verified_scoring_artifact,
)


class ActivationOrchestrationError(RuntimeError):
    """Raised when offline orchestration inputs are unsafe or ambiguous."""


@dataclass(frozen=True)
class ActivationOrchestrationResult:
    """Verified inputs, deterministic decisions, and local output paths."""

    scoring_artifact: VerifiedScoringArtifact
    contact_context_artifact: VerifiedContactContextArtifact
    decision_result: ActivationDecisionResult
    decision_path: Path
    metadata_path: Path
    review_queue_path: Path
    review_metadata_path: Path


def _validate_distinct_paths(
    *,
    scoring_prediction_path: Path,
    scoring_metadata_path: Path,
    context_path: Path,
    context_metadata_path: Path,
    activation_decision_path: Path,
    activation_metadata_path: Path,
    review_queue_path: Path,
    review_metadata_path: Path,
) -> None:
    """Prevent any output path from overlapping governed inputs."""
    named_paths = {
        "scoring_prediction_path": scoring_prediction_path,
        "scoring_metadata_path": scoring_metadata_path,
        "context_path": context_path,
        "context_metadata_path": context_metadata_path,
        "activation_decision_path": activation_decision_path,
        "activation_metadata_path": activation_metadata_path,
        "review_queue_path": review_queue_path,
        "review_metadata_path": review_metadata_path,
    }

    resolved: dict[Path, str] = {}

    for field_name, path in named_paths.items():
        normalised = path.resolve()

        if normalised in resolved:
            raise ActivationOrchestrationError(
                f"{field_name} and {resolved[normalised]} must use distinct paths."
            )

        resolved[normalised] = field_name


def _contact_context_lineage(
    artifact: VerifiedContactContextArtifact,
    *,
    context_path: Path,
) -> ContactContextLineage:
    """Build validated runtime lineage from verified context metadata."""
    metadata = artifact.metadata

    try:
        snapshot_timestamp = datetime.fromisoformat(metadata.snapshot_timestamp)
    except ValueError as error:
        raise ActivationOrchestrationError(
            "Verified contact-context metadata contains an invalid snapshot timestamp."
        ) from error

    if snapshot_timestamp.tzinfo is None or snapshot_timestamp.utcoffset() is None:
        raise ActivationOrchestrationError(
            "Verified contact-context snapshot timestamp must be timezone-aware."
        )

    return ContactContextLineage(
        artifact_path=str(context_path),
        artifact_sha256=metadata.context_artifact_sha256,
        source_name=metadata.source_name,
        source_snapshot_reference=(metadata.source_snapshot_reference),
        source_query_sha256=metadata.source_query_sha256,
        snapshot_timestamp=snapshot_timestamp.astimezone(UTC),
    )


def orchestrate_offline_activation(
    *,
    context_path: Path,
    context_metadata_path: Path,
    decision_timestamp: datetime,
    policy: ActivationPolicy | None = None,
    scoring_prediction_path: Path = DEFAULT_PREDICTION_PATH,
    scoring_metadata_path: Path = DEFAULT_SCORING_METADATA_PATH,
    activation_decision_path: Path = DEFAULT_ACTIVATION_DECISION_PATH,
    activation_metadata_path: Path = DEFAULT_ACTIVATION_METADATA_PATH,
    review_queue_path: Path = DEFAULT_REVIEW_QUEUE_PATH,
    review_metadata_path: Path = DEFAULT_REVIEW_QUEUE_METADATA_PATH,
) -> ActivationOrchestrationResult:
    """Verify inputs, decide activations, and write local artifacts only."""
    _validate_distinct_paths(
        scoring_prediction_path=scoring_prediction_path,
        scoring_metadata_path=scoring_metadata_path,
        context_path=context_path,
        context_metadata_path=context_metadata_path,
        activation_decision_path=activation_decision_path,
        activation_metadata_path=activation_metadata_path,
        review_queue_path=review_queue_path,
        review_metadata_path=review_metadata_path,
    )

    scoring_artifact = load_verified_scoring_artifact(
        scoring_prediction_path,
        scoring_metadata_path,
    )
    context_artifact = load_verified_contact_context_artifact(
        context_path,
        context_metadata_path,
        decision_timestamp=decision_timestamp,
    )

    lineage = _contact_context_lineage(
        context_artifact,
        context_path=context_path,
    )

    decision_result = decide_activations(
        predictions=scoring_artifact.predictions,
        contexts=context_artifact.contexts,
        policy=policy or ActivationPolicy(),
        decision_timestamp=decision_timestamp,
        scoring_artifact_path=str(scoring_prediction_path),
        scoring_artifact_sha256=(scoring_artifact.prediction_artifact_sha256),
        contact_context_lineage=lineage,
    )

    written_decision_path, written_metadata_path = write_activation_artifact(
        decision_result,
        decision_path=activation_decision_path,
        metadata_path=activation_metadata_path,
    )
    written_review_queue_path, written_review_metadata_path = write_review_queue_artifact(
        decision_result,
        activation_decision_path=written_decision_path,
        activation_metadata_path=written_metadata_path,
        review_queue_path=review_queue_path,
        review_metadata_path=review_metadata_path,
    )

    return ActivationOrchestrationResult(
        scoring_artifact=scoring_artifact,
        contact_context_artifact=context_artifact,
        decision_result=decision_result,
        decision_path=written_decision_path,
        metadata_path=written_metadata_path,
        review_queue_path=written_review_queue_path,
        review_metadata_path=written_review_metadata_path,
    )
