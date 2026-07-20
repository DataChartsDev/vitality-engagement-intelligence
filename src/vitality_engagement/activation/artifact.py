"""Persist and verify deterministic activation decision artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
import pandas as pd

from vitality_engagement.activation.engine import ActivationDecisionResult
from vitality_engagement.activation.schema import (
    DecisionOutcome,
    InterventionCategory,
    ReasonCode,
)

DEFAULT_ACTIVATION_DECISION_PATH: Final = Path("artifacts/activation/activation_decisions.parquet")
DEFAULT_ACTIVATION_METADATA_PATH: Final = Path(
    "artifacts/activation/activation_decisions.metadata.json"
)
ACTIVATION_ARTIFACT_VERSION: Final = 2

RUN_ID_COLUMN: Final = "run_id"
POLICY_VERSION_COLUMN: Final = "policy_version"
MEMBER_ID_COLUMN: Final = "member_id"
PREDICTION_DATE_COLUMN: Final = "prediction_date"
DECISION_TIMESTAMP_COLUMN: Final = "decision_timestamp"
OUTCOME_COLUMN: Final = "outcome"
REASON_CODE_COLUMN: Final = "reason_code"
RISK_PROBABILITY_COLUMN: Final = "risk_probability"
MODEL_NAME_COLUMN: Final = "model_name"
THRESHOLD_COLUMN: Final = "threshold"
INTERVENTION_CATEGORY_COLUMN: Final = "intervention_category"
PRIORITY_RANK_COLUMN: Final = "priority_rank"

ACTIVATION_DECISION_COLUMNS: Final = (
    RUN_ID_COLUMN,
    POLICY_VERSION_COLUMN,
    MEMBER_ID_COLUMN,
    PREDICTION_DATE_COLUMN,
    DECISION_TIMESTAMP_COLUMN,
    OUTCOME_COLUMN,
    REASON_CODE_COLUMN,
    RISK_PROBABILITY_COLUMN,
    MODEL_NAME_COLUMN,
    THRESHOLD_COLUMN,
    INTERVENTION_CATEGORY_COLUMN,
    PRIORITY_RANK_COLUMN,
)


class ActivationArtifactError(RuntimeError):
    """Raised when an activation artifact violates its contract."""


@dataclass(frozen=True)
class ActivationArtifactMetadata:
    """Persisted lineage and counts for one activation decision artifact."""

    artifact_version: int
    run_id: str
    policy_version: str
    policy_fingerprint: str
    model_name: str
    threshold: float
    scoring_artifact_path: str
    scoring_artifact_sha256: str
    contact_context_artifact_path: str
    contact_context_artifact_sha256: str
    contact_context_source_name: str
    contact_context_source_snapshot_reference: str
    contact_context_source_query_sha256: str
    contact_context_snapshot_timestamp: str
    decision_timestamp: str
    capacity_limit: int
    source_row_count: int
    source_member_count: int
    superseded_count: int
    below_threshold_count: int
    excluded_count: int
    suppressed_count: int
    eligible_count: int
    capacity_not_selected_count: int
    selected_count: int
    output_columns: tuple[str, ...]


def build_activation_decision_frame(
    result: ActivationDecisionResult,
) -> pd.DataFrame:
    """Flatten validated audit records into the persisted table contract."""
    rows: list[dict[str, object]] = []

    for record in result.audit_records:
        rows.append(
            {
                RUN_ID_COLUMN: record.run_id,
                POLICY_VERSION_COLUMN: record.policy_version,
                MEMBER_ID_COLUMN: record.member_id,
                PREDICTION_DATE_COLUMN: record.prediction_date,
                DECISION_TIMESTAMP_COLUMN: record.decision_timestamp,
                OUTCOME_COLUMN: record.outcome.value,
                REASON_CODE_COLUMN: record.reason_code.value,
                RISK_PROBABILITY_COLUMN: record.risk_probability,
                MODEL_NAME_COLUMN: record.model_name,
                THRESHOLD_COLUMN: record.threshold,
                INTERVENTION_CATEGORY_COLUMN: (
                    None
                    if record.intervention_category is None
                    else record.intervention_category.value
                ),
                PRIORITY_RANK_COLUMN: record.priority_rank,
            }
        )

    frame = pd.DataFrame.from_records(
        rows,
        columns=list(ACTIVATION_DECISION_COLUMNS),
    )

    if len(frame) != result.metadata.source_row_count:
        raise ActivationArtifactError(
            "Activation decision frame does not match the source-row count."
        )

    return frame


def build_activation_artifact_metadata(
    result: ActivationDecisionResult,
) -> ActivationArtifactMetadata:
    """Build persisted metadata from validated activation run metadata."""
    metadata = result.metadata

    return ActivationArtifactMetadata(
        artifact_version=ACTIVATION_ARTIFACT_VERSION,
        run_id=metadata.run_id,
        policy_version=metadata.policy_version,
        policy_fingerprint=metadata.policy_fingerprint,
        model_name=metadata.model_name,
        threshold=metadata.threshold,
        scoring_artifact_path=metadata.scoring_artifact_path,
        scoring_artifact_sha256=metadata.scoring_artifact_sha256,
        contact_context_artifact_path=(metadata.contact_context_lineage.artifact_path),
        contact_context_artifact_sha256=(metadata.contact_context_lineage.artifact_sha256),
        contact_context_source_name=(metadata.contact_context_lineage.source_name),
        contact_context_source_snapshot_reference=(
            metadata.contact_context_lineage.source_snapshot_reference
        ),
        contact_context_source_query_sha256=(metadata.contact_context_lineage.source_query_sha256),
        contact_context_snapshot_timestamp=(
            metadata.contact_context_lineage.snapshot_timestamp.isoformat()
        ),
        decision_timestamp=metadata.decision_timestamp.isoformat(),
        capacity_limit=metadata.capacity_limit,
        source_row_count=metadata.source_row_count,
        source_member_count=metadata.source_member_count,
        superseded_count=metadata.superseded_count,
        below_threshold_count=metadata.below_threshold_count,
        excluded_count=metadata.excluded_count,
        suppressed_count=metadata.suppressed_count,
        eligible_count=metadata.eligible_count,
        capacity_not_selected_count=metadata.capacity_not_selected_count,
        selected_count=metadata.selected_count,
        output_columns=ACTIVATION_DECISION_COLUMNS,
    )


def _metadata_payload(
    metadata: ActivationArtifactMetadata,
) -> dict[str, object]:
    """Return the JSON-compatible metadata representation."""
    serialized = json.loads(
        json.dumps(
            asdict(metadata),
            sort_keys=True,
        )
    )

    if not isinstance(serialized, dict):
        raise ActivationArtifactError("Activation metadata must serialize to an object.")

    return cast(dict[str, object], serialized)


def _temporary_path(output_path: Path) -> Path:
    """Return a temporary path beside the requested final output."""
    return output_path.parent / f".{output_path.name}.tmp"


def _write_pending_decisions(
    frame: pd.DataFrame,
    temporary_path: Path,
) -> None:
    """Write a pending Parquet artifact."""
    temporary_path.parent.mkdir(parents=True, exist_ok=True)

    if temporary_path.exists():
        temporary_path.unlink()

    frame.to_parquet(
        temporary_path,
        index=False,
    )


def _write_pending_metadata(
    metadata: ActivationArtifactMetadata,
    temporary_path: Path,
) -> None:
    """Write pending JSON metadata."""
    temporary_path.parent.mkdir(parents=True, exist_ok=True)

    if temporary_path.exists():
        temporary_path.unlink()

    temporary_path.write_text(
        json.dumps(
            _metadata_payload(metadata),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _count_outcome(
    frame: pd.DataFrame,
    outcome: DecisionOutcome,
) -> int:
    """Count one persisted decision outcome."""
    return int((frame[OUTCOME_COLUMN].astype(str) == outcome.value).sum())


def verify_activation_artifact(
    decision_path: Path,
    metadata_path: Path,
    *,
    expected_result: ActivationDecisionResult,
) -> None:
    """Verify persisted activation decisions and metadata."""
    if not decision_path.is_file():
        raise FileNotFoundError(f"Activation decision artifact does not exist: {decision_path}")

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Activation metadata artifact does not exist: {metadata_path}")

    restored = pd.read_parquet(decision_path)

    if tuple(str(column) for column in restored.columns) != ACTIVATION_DECISION_COLUMNS:
        raise ActivationArtifactError(
            "Persisted activation columns do not match the output contract."
        )

    expected_metadata = build_activation_artifact_metadata(expected_result)

    if len(restored) != expected_metadata.source_row_count:
        raise ActivationArtifactError("Persisted activation row count is inconsistent.")

    if bool(restored.duplicated(subset=[MEMBER_ID_COLUMN, PREDICTION_DATE_COLUMN]).any()):
        raise ActivationArtifactError(
            "Persisted activation decisions contain duplicate identifiers."
        )

    restored_dates = tuple(
        pd.Timestamp(value).date() for value in restored[PREDICTION_DATE_COLUMN].tolist()
    )
    restored_identifiers = set(
        zip(
            restored[MEMBER_ID_COLUMN].astype(str).tolist(),
            restored_dates,
            strict=True,
        )
    )
    expected_identifiers = {
        (record.member_id, record.prediction_date) for record in expected_result.audit_records
    }

    if restored_identifiers != expected_identifiers:
        raise ActivationArtifactError(
            "Persisted activation identifiers do not match the source decisions."
        )

    if set(restored[RUN_ID_COLUMN].astype(str).tolist()) != {expected_metadata.run_id}:
        raise ActivationArtifactError("Persisted activation run IDs are inconsistent.")

    if set(restored[POLICY_VERSION_COLUMN].astype(str).tolist()) != {
        expected_metadata.policy_version
    }:
        raise ActivationArtifactError("Persisted activation policy versions are inconsistent.")

    if set(restored[MODEL_NAME_COLUMN].astype(str).tolist()) != {expected_metadata.model_name}:
        raise ActivationArtifactError("Persisted activation model names are inconsistent.")

    probabilities = restored[RISK_PROBABILITY_COLUMN].to_numpy(dtype=np.float64)

    if not bool(np.isfinite(probabilities).all()):
        raise ActivationArtifactError(
            "Persisted activation probabilities contain non-finite values."
        )

    if bool(((probabilities < 0.0) | (probabilities > 1.0)).any()):
        raise ActivationArtifactError(
            "Persisted activation probabilities fall outside zero to one."
        )

    thresholds = restored[THRESHOLD_COLUMN].to_numpy(dtype=np.float64)

    if not bool(
        np.allclose(
            thresholds,
            expected_metadata.threshold,
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise ActivationArtifactError("Persisted activation thresholds are inconsistent.")

    expected_timestamp = pd.Timestamp(expected_result.metadata.decision_timestamp).tz_convert("UTC")
    restored_timestamps = pd.to_datetime(
        restored[DECISION_TIMESTAMP_COLUMN],
        errors="raise",
        utc=True,
    )

    if bool((restored_timestamps != expected_timestamp).any()):
        raise ActivationArtifactError("Persisted activation decision timestamps are inconsistent.")

    persisted_outcomes = set(restored[OUTCOME_COLUMN].astype(str).tolist())
    allowed_outcomes = {outcome.value for outcome in DecisionOutcome}

    if not persisted_outcomes.issubset(allowed_outcomes):
        raise ActivationArtifactError("Persisted activation outcomes contain unsupported values.")

    persisted_reasons = set(restored[REASON_CODE_COLUMN].astype(str).tolist())
    allowed_reasons = {reason.value for reason in ReasonCode}

    if not persisted_reasons.issubset(allowed_reasons):
        raise ActivationArtifactError(
            "Persisted activation reason codes contain unsupported values."
        )

    selected_mask = (
        restored[OUTCOME_COLUMN].astype(str) == DecisionOutcome.SELECTED_FOR_REVIEW.value
    )
    no_contact_mask = ~selected_mask

    if bool(
        restored.loc[
            selected_mask,
            INTERVENTION_CATEGORY_COLUMN,
        ]
        .isna()
        .any()
    ):
        raise ActivationArtifactError("Selected activation rows require an intervention category.")

    if bool(restored.loc[selected_mask, PRIORITY_RANK_COLUMN].isna().any()):
        raise ActivationArtifactError("Selected activation rows require a priority rank.")

    if bool(
        restored.loc[
            no_contact_mask,
            INTERVENTION_CATEGORY_COLUMN,
        ]
        .notna()
        .any()
    ):
        raise ActivationArtifactError("No-contact rows must not contain an intervention category.")

    if bool(
        restored.loc[
            no_contact_mask,
            PRIORITY_RANK_COLUMN,
        ]
        .notna()
        .any()
    ):
        raise ActivationArtifactError("No-contact rows must not contain a priority rank.")

    selected_categories = set(
        restored.loc[
            selected_mask,
            INTERVENTION_CATEGORY_COLUMN,
        ]
        .astype(str)
        .tolist()
    )
    allowed_categories = {category.value for category in InterventionCategory}

    if not selected_categories.issubset(allowed_categories):
        raise ActivationArtifactError(
            "Persisted intervention categories contain unsupported values."
        )

    selected_ranks = np.sort(
        restored.loc[selected_mask, PRIORITY_RANK_COLUMN].to_numpy(dtype=np.float64)
    )
    expected_ranks = np.arange(
        1,
        expected_metadata.selected_count + 1,
        dtype=np.float64,
    )

    if not bool(np.array_equal(selected_ranks, expected_ranks)):
        raise ActivationArtifactError("Persisted activation priority ranks are inconsistent.")

    outcome_counts = {
        DecisionOutcome.NO_CONTACT_SUPERSEDED: (expected_metadata.superseded_count),
        DecisionOutcome.NO_CONTACT_BELOW_THRESHOLD: (expected_metadata.below_threshold_count),
        DecisionOutcome.NO_CONTACT_EXCLUDED: (expected_metadata.excluded_count),
        DecisionOutcome.NO_CONTACT_SUPPRESSED: (expected_metadata.suppressed_count),
        DecisionOutcome.NO_CONTACT_CAPACITY: (expected_metadata.capacity_not_selected_count),
        DecisionOutcome.SELECTED_FOR_REVIEW: (expected_metadata.selected_count),
    }

    for outcome, expected_count in outcome_counts.items():
        if _count_outcome(restored, outcome) != expected_count:
            raise ActivationArtifactError(
                f"Persisted count is inconsistent for outcome: {outcome.value}"
            )

    raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if not isinstance(raw_metadata, dict):
        raise ActivationArtifactError("Persisted activation metadata must contain a JSON object.")

    if raw_metadata != _metadata_payload(expected_metadata):
        raise ActivationArtifactError(
            "Persisted activation metadata does not match the source result."
        )


def write_activation_artifact(
    result: ActivationDecisionResult,
    *,
    decision_path: Path = DEFAULT_ACTIVATION_DECISION_PATH,
    metadata_path: Path = DEFAULT_ACTIVATION_METADATA_PATH,
) -> tuple[Path, Path]:
    """Persist and verify activation decisions plus run metadata."""
    if decision_path.resolve() == metadata_path.resolve():
        raise ActivationArtifactError("Activation decision and metadata paths must differ.")

    frame = build_activation_decision_frame(result)
    metadata = build_activation_artifact_metadata(result)

    temporary_decision_path = _temporary_path(decision_path)
    temporary_metadata_path = _temporary_path(metadata_path)

    try:
        _write_pending_decisions(
            frame,
            temporary_decision_path,
        )
        _write_pending_metadata(
            metadata,
            temporary_metadata_path,
        )
        verify_activation_artifact(
            temporary_decision_path,
            temporary_metadata_path,
            expected_result=result,
        )

        decision_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        os.replace(temporary_decision_path, decision_path)
        os.replace(temporary_metadata_path, metadata_path)
    finally:
        if temporary_decision_path.exists():
            temporary_decision_path.unlink()

        if temporary_metadata_path.exists():
            temporary_metadata_path.unlink()

    return decision_path, metadata_path
