"""Verify and upload activation artifacts to governed BigQuery tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import pandas as pd
from google.cloud import bigquery

from vitality_engagement.activation.artifact import (
    ACTIVATION_DECISION_COLUMNS,
    DEFAULT_ACTIVATION_DECISION_PATH,
    DEFAULT_ACTIVATION_METADATA_PATH,
    ActivationArtifactMetadata,
    build_activation_artifact_metadata,
    verify_activation_artifact,
)
from vitality_engagement.activation.bigquery import (
    ACTIVATION_DECISION_SCHEMA,
    ACTIVATION_DECISION_WAREHOUSE_COLUMNS,
    ACTIVATION_RUN_COLUMNS,
    ACTIVATION_RUN_SCHEMA,
    ACTIVATION_RUN_WAREHOUSE_COLUMNS,
    ActivationStagingTables,
    ActivationWarehouseConfig,
    ActivationWarehouseError,
    SchemaDefinition,
    build_activation_merge_query,
    build_bigquery_schema,
    build_create_activation_tables_query,
    build_staging_tables,
)
from vitality_engagement.activation.engine import ActivationDecisionResult


class _BigQueryJobLike(Protocol):
    """Minimal asynchronous BigQuery job interface."""

    def result(self) -> object:
        """Wait for the BigQuery job to complete."""


class ActivationWarehouseClient(Protocol):
    """Minimal BigQuery client interface used by the uploader."""

    def query(
        self,
        query: str,
        *,
        location: str,
        job_config: bigquery.QueryJobConfig,
    ) -> _BigQueryJobLike:
        """Start a GoogleSQL query job."""

    def load_table_from_dataframe(
        self,
        dataframe: pd.DataFrame,
        destination: str,
        *,
        location: str,
        job_config: bigquery.LoadJobConfig,
    ) -> _BigQueryJobLike:
        """Load a DataFrame into a BigQuery table."""

    def delete_table(
        self,
        table: str,
        *,
        not_found_ok: bool,
    ) -> None:
        """Delete a BigQuery table."""


@dataclass(frozen=True)
class ActivationWarehouseUploadResult:
    """Summary of one successful activation warehouse upload."""

    run_id: str
    decision_row_count: int
    run_table: str
    decision_table: str


def _normalise_ingested_at(
    ingested_at: datetime | None,
) -> datetime:
    """Return a timezone-aware UTC ingestion timestamp."""
    timestamp = ingested_at or datetime.now(UTC)

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ActivationWarehouseError("ingested_at must be timezone-aware.")

    return timestamp.astimezone(UTC)


def _build_decision_warehouse_frame(
    decision_path: Path,
    *,
    ingested_at: datetime,
) -> pd.DataFrame:
    """Read verified decisions and apply warehouse-compatible types."""
    frame = pd.read_parquet(
        decision_path,
        columns=list(ACTIVATION_DECISION_COLUMNS),
    ).copy()

    frame["prediction_date"] = pd.to_datetime(
        frame["prediction_date"],
        errors="raise",
    ).dt.date

    frame["decision_timestamp"] = pd.to_datetime(
        frame["decision_timestamp"],
        errors="raise",
        utc=True,
    )

    frame["risk_probability"] = pd.to_numeric(
        frame["risk_probability"],
        errors="raise",
    ).astype("float64")

    frame["threshold"] = pd.to_numeric(
        frame["threshold"],
        errors="raise",
    ).astype("float64")

    frame["priority_rank"] = pd.to_numeric(
        frame["priority_rank"],
        errors="raise",
    ).astype("Int64")

    frame["ingested_at"] = pd.Timestamp(ingested_at)

    return frame.loc[
        :,
        list(ACTIVATION_DECISION_WAREHOUSE_COLUMNS),
    ]


def _build_run_warehouse_frame(
    metadata: ActivationArtifactMetadata,
    *,
    ingested_at: datetime,
) -> pd.DataFrame:
    """Build the single governed activation-run warehouse row."""
    payload = asdict(metadata)
    payload.pop("output_columns")

    frame = pd.DataFrame.from_records(
        [payload],
        columns=list(ACTIVATION_RUN_COLUMNS),
    )

    frame["decision_timestamp"] = pd.to_datetime(
        frame["decision_timestamp"],
        errors="raise",
        utc=True,
    )
    frame["contact_context_snapshot_timestamp"] = pd.to_datetime(
        frame["contact_context_snapshot_timestamp"],
        errors="raise",
        utc=True,
    )
    frame["ingested_at"] = pd.Timestamp(ingested_at)

    return frame.loc[
        :,
        list(ACTIVATION_RUN_WAREHOUSE_COLUMNS),
    ]


def _execute_query(
    client: ActivationWarehouseClient,
    query: str,
    *,
    location: str,
) -> None:
    """Execute a location-bound GoogleSQL query."""
    job = client.query(
        query,
        location=location,
        job_config=bigquery.QueryJobConfig(
            use_legacy_sql=False,
        ),
    )
    job.result()


def _load_staging_frame(
    client: ActivationWarehouseClient,
    frame: pd.DataFrame,
    destination: str,
    schema_definition: SchemaDefinition,
    *,
    location: str,
) -> None:
    """Overwrite one run-specific staging table using an explicit schema."""
    job = client.load_table_from_dataframe(
        frame,
        destination,
        location=location,
        job_config=bigquery.LoadJobConfig(
            schema=list(build_bigquery_schema(schema_definition)),
            create_disposition=(bigquery.CreateDisposition.CREATE_IF_NEEDED),
            write_disposition=(bigquery.WriteDisposition.WRITE_TRUNCATE),
        ),
    )
    job.result()


def _cleanup_staging_tables(
    client: ActivationWarehouseClient,
    staging: ActivationStagingTables,
) -> tuple[str, ...]:
    """Delete run-specific staging tables and return cleanup failures."""
    errors: list[str] = []

    for table in (
        staging.decision_table,
        staging.run_table,
    ):
        try:
            client.delete_table(
                table,
                not_found_ok=True,
            )
        except Exception as error:
            errors.append(f"{table}: {type(error).__name__}: {error}")

    return tuple(errors)


def upload_activation_artifact_to_bigquery(
    expected_result: ActivationDecisionResult,
    *,
    decision_path: Path = DEFAULT_ACTIVATION_DECISION_PATH,
    metadata_path: Path = DEFAULT_ACTIVATION_METADATA_PATH,
    config: ActivationWarehouseConfig | None = None,
    client: ActivationWarehouseClient | None = None,
    ingested_at: datetime | None = None,
) -> ActivationWarehouseUploadResult:
    """Verify and idempotently upload one activation artifact."""
    active_config = config or ActivationWarehouseConfig()

    verify_activation_artifact(
        decision_path,
        metadata_path,
        expected_result=expected_result,
    )

    ingestion_timestamp = _normalise_ingested_at(ingested_at)
    metadata = build_activation_artifact_metadata(expected_result)

    decision_frame = _build_decision_warehouse_frame(
        decision_path,
        ingested_at=ingestion_timestamp,
    )
    run_frame = _build_run_warehouse_frame(
        metadata,
        ingested_at=ingestion_timestamp,
    )

    staging = build_staging_tables(
        active_config,
        metadata.run_id,
    )

    active_client = client

    if active_client is None:
        active_client = cast(
            ActivationWarehouseClient,
            bigquery.Client(
                project=active_config.project_id,
            ),
        )

    operation_succeeded = False

    try:
        _execute_query(
            active_client,
            build_create_activation_tables_query(active_config),
            location=active_config.location,
        )

        _load_staging_frame(
            active_client,
            run_frame,
            staging.run_table,
            ACTIVATION_RUN_SCHEMA,
            location=active_config.location,
        )

        _load_staging_frame(
            active_client,
            decision_frame,
            staging.decision_table,
            ACTIVATION_DECISION_SCHEMA,
            location=active_config.location,
        )

        _execute_query(
            active_client,
            build_activation_merge_query(
                active_config,
                staging,
            ),
            location=active_config.location,
        )

        operation_succeeded = True
    finally:
        cleanup_errors = _cleanup_staging_tables(
            active_client,
            staging,
        )

        if operation_succeeded and cleanup_errors:
            joined_errors = "; ".join(cleanup_errors)
            raise ActivationWarehouseError(
                f"Activation records were merged, but staging cleanup failed: {joined_errors}"
            )

    return ActivationWarehouseUploadResult(
        run_id=metadata.run_id,
        decision_row_count=len(decision_frame),
        run_table=active_config.run_table,
        decision_table=active_config.decision_table,
    )
