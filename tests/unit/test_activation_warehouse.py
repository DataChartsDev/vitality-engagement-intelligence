"""Tests for the verified activation BigQuery uploader."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from google.cloud import bigquery

from vitality_engagement.activation.artifact import (
    ActivationArtifactError,
    write_activation_artifact,
)
from vitality_engagement.activation.bigquery import (
    ACTIVATION_DECISION_SCHEMA,
    ACTIVATION_RUN_SCHEMA,
    ActivationWarehouseConfig,
    ActivationWarehouseError,
)
from vitality_engagement.activation.engine import (
    ActivationDecisionResult,
    decide_activations,
)
from vitality_engagement.activation.policy import ActivationPolicy
from vitality_engagement.activation.schema import (
    ContactContextLineage,
    MemberActivationContext,
    ScoredPrediction,
)
from vitality_engagement.activation.warehouse import (
    ActivationWarehouseClient,
    upload_activation_artifact_to_bigquery,
)

DECISION_TIMESTAMP = datetime(
    2025,
    6,
    30,
    8,
    0,
    tzinfo=UTC,
)
INGESTED_AT = datetime(
    2025,
    6,
    30,
    8,
    5,
    tzinfo=UTC,
)
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


class _FakeJob:
    """Synchronous fake BigQuery job."""

    def __init__(self) -> None:
        self.result_call_count = 0

    def result(self) -> object:
        """Record that the caller awaited completion."""
        self.result_call_count += 1
        return None


class _FakeClient:
    """Recording BigQuery client for isolated unit tests."""

    def __init__(
        self,
        *,
        fail_on_load_number: int | None = None,
    ) -> None:
        self.fail_on_load_number = fail_on_load_number
        self.query_calls: list[tuple[str, str, bigquery.QueryJobConfig]] = []
        self.load_calls: list[
            tuple[
                pd.DataFrame,
                str,
                str,
                bigquery.LoadJobConfig,
            ]
        ] = []
        self.deleted_tables: list[str] = []

    def query(
        self,
        query: str,
        *,
        location: str,
        job_config: bigquery.QueryJobConfig,
    ) -> _FakeJob:
        """Record a query invocation."""
        self.query_calls.append(
            (
                query,
                location,
                job_config,
            )
        )
        return _FakeJob()

    def load_table_from_dataframe(
        self,
        dataframe: pd.DataFrame,
        destination: str,
        *,
        location: str,
        job_config: bigquery.LoadJobConfig,
    ) -> _FakeJob:
        """Record a staging-table load."""
        self.load_calls.append(
            (
                dataframe.copy(),
                destination,
                location,
                job_config,
            )
        )

        if self.fail_on_load_number == len(self.load_calls):
            raise RuntimeError("simulated staging load failure")

        return _FakeJob()

    def delete_table(
        self,
        table: str,
        *,
        not_found_ok: bool,
    ) -> None:
        """Record staging-table cleanup."""
        assert not_found_ok is True
        self.deleted_tables.append(table)


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
                probability=0.90,
            ),
            _prediction(
                "member-b",
                probability=0.80,
            ),
            _prediction(
                "member-c",
                probability=0.70,
            ),
            _prediction(
                "member-d",
                probability=0.20,
            ),
        ],
        contexts=[
            _context("member-a"),
            _context(
                "member-b",
                contact_allowed=False,
            ),
            _context("member-c"),
        ],
        policy=ActivationPolicy(
            maximum_activations_per_run=1,
        ),
        decision_timestamp=DECISION_TIMESTAMP,
        scoring_artifact_path=SCORING_PATH,
        scoring_artifact_sha256=SCORING_DIGEST,
        contact_context_lineage=CONTACT_CONTEXT_LINEAGE,
    )


def _write_artifact(
    tmp_path: Path,
) -> tuple[
    ActivationDecisionResult,
    Path,
    Path,
]:
    result = _result()
    decision_path = tmp_path / "activation_decisions.parquet"
    metadata_path = tmp_path / "activation_decisions.metadata.json"

    write_activation_artifact(
        result,
        decision_path=decision_path,
        metadata_path=metadata_path,
    )

    return result, decision_path, metadata_path


def test_successful_upload_is_explicit_atomic_and_cleaned(
    tmp_path: Path,
) -> None:
    result, decision_path, metadata_path = _write_artifact(tmp_path)
    fake_client = _FakeClient()
    typed_client: ActivationWarehouseClient = fake_client

    upload = upload_activation_artifact_to_bigquery(
        result,
        decision_path=decision_path,
        metadata_path=metadata_path,
        client=typed_client,
        ingested_at=INGESTED_AT,
    )

    assert upload.run_id == result.metadata.run_id
    assert upload.decision_row_count == 5
    assert upload.run_table.endswith(".activation_runs")
    assert upload.decision_table.endswith(".activation_decisions")

    assert len(fake_client.query_calls) == 2
    create_query, create_location, create_config = fake_client.query_calls[0]
    merge_query, merge_location, merge_config = fake_client.query_calls[1]

    assert "CREATE TABLE IF NOT EXISTS" in create_query
    assert "BEGIN TRANSACTION;" in merge_query
    assert merge_query.count("MERGE ") == 2
    assert "COMMIT TRANSACTION;" in merge_query
    assert create_location == "asia-southeast1"
    assert merge_location == "asia-southeast1"
    assert create_config.use_legacy_sql is False
    assert merge_config.use_legacy_sql is False

    assert len(fake_client.load_calls) == 2

    run_frame, run_stage, run_location, run_config = fake_client.load_calls[0]
    decision_frame, decision_stage, decision_location, decision_config = fake_client.load_calls[1]

    assert len(run_frame) == 1
    assert len(decision_frame) == 5
    assert run_stage.endswith(f"_activation_runs_stage_{result.metadata.run_id[4:]}")
    assert decision_stage.endswith(f"_activation_decisions_stage_{result.metadata.run_id[4:]}")
    assert run_location == "asia-southeast1"
    assert decision_location == "asia-southeast1"

    assert run_config.write_disposition == (bigquery.WriteDisposition.WRITE_TRUNCATE)
    assert decision_config.write_disposition == (bigquery.WriteDisposition.WRITE_TRUNCATE)

    assert [field.name for field in run_config.schema] == [
        definition[0] for definition in ACTIVATION_RUN_SCHEMA
    ]
    assert [field.name for field in decision_config.schema] == [
        definition[0] for definition in ACTIVATION_DECISION_SCHEMA
    ]

    assert set(run_frame["ingested_at"]) == {pd.Timestamp(INGESTED_AT)}
    assert set(run_frame["contact_context_artifact_path"]) == {
        CONTACT_CONTEXT_LINEAGE.artifact_path
    }
    assert set(run_frame["contact_context_artifact_sha256"]) == {
        CONTACT_CONTEXT_LINEAGE.artifact_sha256
    }
    assert set(run_frame["contact_context_source_name"]) == {CONTACT_CONTEXT_LINEAGE.source_name}
    assert set(run_frame["contact_context_source_snapshot_reference"]) == {
        CONTACT_CONTEXT_LINEAGE.source_snapshot_reference
    }
    assert set(run_frame["contact_context_source_query_sha256"]) == {
        CONTACT_CONTEXT_LINEAGE.source_query_sha256
    }
    assert set(run_frame["contact_context_snapshot_timestamp"]) == {
        pd.Timestamp(CONTACT_CONTEXT_LINEAGE.snapshot_timestamp)
    }
    assert set(decision_frame["ingested_at"]) == {pd.Timestamp(INGESTED_AT)}

    assert fake_client.deleted_tables == [
        decision_stage,
        run_stage,
    ]


def test_verification_failure_prevents_all_bigquery_calls(
    tmp_path: Path,
) -> None:
    result, decision_path, metadata_path = _write_artifact(tmp_path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["selected_count"] = 99
    metadata_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    fake_client = _FakeClient()

    with pytest.raises(
        ActivationArtifactError,
        match="metadata does not match",
    ):
        upload_activation_artifact_to_bigquery(
            result,
            decision_path=decision_path,
            metadata_path=metadata_path,
            client=fake_client,
            ingested_at=INGESTED_AT,
        )

    assert fake_client.query_calls == []
    assert fake_client.load_calls == []
    assert fake_client.deleted_tables == []


def test_staging_tables_are_cleaned_after_load_failure(
    tmp_path: Path,
) -> None:
    result, decision_path, metadata_path = _write_artifact(tmp_path)
    fake_client = _FakeClient(
        fail_on_load_number=2,
    )
    config = ActivationWarehouseConfig()

    with pytest.raises(
        RuntimeError,
        match="simulated staging load failure",
    ):
        upload_activation_artifact_to_bigquery(
            result,
            decision_path=decision_path,
            metadata_path=metadata_path,
            config=config,
            client=fake_client,
            ingested_at=INGESTED_AT,
        )

    suffix = result.metadata.run_id[4:]

    assert fake_client.deleted_tables == [
        (f"{config.project_id}.{config.dataset_id}._activation_decisions_stage_{suffix}"),
        (f"{config.project_id}.{config.dataset_id}._activation_runs_stage_{suffix}"),
    ]


def test_naive_ingestion_timestamp_is_rejected(
    tmp_path: Path,
) -> None:
    result, decision_path, metadata_path = _write_artifact(tmp_path)
    fake_client = _FakeClient()

    with pytest.raises(
        ActivationWarehouseError,
        match="timezone-aware",
    ):
        upload_activation_artifact_to_bigquery(
            result,
            decision_path=decision_path,
            metadata_path=metadata_path,
            client=fake_client,
            ingested_at=datetime(
                2025,
                6,
                30,
                8,
                5,
            ),
        )

    assert fake_client.query_calls == []
    assert fake_client.load_calls == []
    assert fake_client.deleted_tables == []
