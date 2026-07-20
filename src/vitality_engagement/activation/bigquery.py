"""Governed BigQuery contracts for Stage 5 activation records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from google.cloud import bigquery

from vitality_engagement.activation.artifact import (
    ACTIVATION_DECISION_COLUMNS,
)

DEFAULT_PROJECT_ID: Final = "vitality-engagement-43999"
DEFAULT_DATASET_ID: Final = "vitality_engagement_dev"
DEFAULT_LOCATION: Final = "asia-southeast1"
DEFAULT_DECISION_TABLE_ID: Final = "activation_decisions"
DEFAULT_RUN_TABLE_ID: Final = "activation_runs"

INGESTED_AT_COLUMN: Final = "ingested_at"

ACTIVATION_RUN_COLUMNS: Final = (
    "artifact_version",
    "run_id",
    "policy_version",
    "policy_fingerprint",
    "model_name",
    "threshold",
    "scoring_artifact_path",
    "scoring_artifact_sha256",
    "contact_context_artifact_path",
    "contact_context_artifact_sha256",
    "contact_context_source_name",
    "contact_context_source_snapshot_reference",
    "contact_context_source_query_sha256",
    "contact_context_snapshot_timestamp",
    "decision_timestamp",
    "capacity_limit",
    "source_row_count",
    "source_member_count",
    "superseded_count",
    "below_threshold_count",
    "excluded_count",
    "suppressed_count",
    "eligible_count",
    "capacity_not_selected_count",
    "selected_count",
)

ACTIVATION_DECISION_WAREHOUSE_COLUMNS: Final = (
    *ACTIVATION_DECISION_COLUMNS,
    INGESTED_AT_COLUMN,
)
ACTIVATION_RUN_WAREHOUSE_COLUMNS: Final = (
    *ACTIVATION_RUN_COLUMNS,
    INGESTED_AT_COLUMN,
)

type SchemaDefinition = tuple[tuple[str, str, str], ...]

ACTIVATION_DECISION_SCHEMA: Final[SchemaDefinition] = (
    ("run_id", "STRING", "REQUIRED"),
    ("policy_version", "STRING", "REQUIRED"),
    ("member_id", "STRING", "REQUIRED"),
    ("prediction_date", "DATE", "REQUIRED"),
    ("decision_timestamp", "TIMESTAMP", "REQUIRED"),
    ("outcome", "STRING", "REQUIRED"),
    ("reason_code", "STRING", "REQUIRED"),
    ("risk_probability", "FLOAT", "REQUIRED"),
    ("model_name", "STRING", "REQUIRED"),
    ("threshold", "FLOAT", "REQUIRED"),
    ("intervention_category", "STRING", "NULLABLE"),
    ("priority_rank", "INTEGER", "NULLABLE"),
    ("ingested_at", "TIMESTAMP", "REQUIRED"),
)

ACTIVATION_RUN_SCHEMA: Final[SchemaDefinition] = (
    ("artifact_version", "INTEGER", "REQUIRED"),
    ("run_id", "STRING", "REQUIRED"),
    ("policy_version", "STRING", "REQUIRED"),
    ("policy_fingerprint", "STRING", "REQUIRED"),
    ("model_name", "STRING", "REQUIRED"),
    ("threshold", "FLOAT", "REQUIRED"),
    ("scoring_artifact_path", "STRING", "REQUIRED"),
    ("scoring_artifact_sha256", "STRING", "REQUIRED"),
    ("contact_context_artifact_path", "STRING", "REQUIRED"),
    ("contact_context_artifact_sha256", "STRING", "REQUIRED"),
    ("contact_context_source_name", "STRING", "REQUIRED"),
    (
        "contact_context_source_snapshot_reference",
        "STRING",
        "REQUIRED",
    ),
    (
        "contact_context_source_query_sha256",
        "STRING",
        "REQUIRED",
    ),
    (
        "contact_context_snapshot_timestamp",
        "TIMESTAMP",
        "REQUIRED",
    ),
    ("decision_timestamp", "TIMESTAMP", "REQUIRED"),
    ("capacity_limit", "INTEGER", "REQUIRED"),
    ("source_row_count", "INTEGER", "REQUIRED"),
    ("source_member_count", "INTEGER", "REQUIRED"),
    ("superseded_count", "INTEGER", "REQUIRED"),
    ("below_threshold_count", "INTEGER", "REQUIRED"),
    ("excluded_count", "INTEGER", "REQUIRED"),
    ("suppressed_count", "INTEGER", "REQUIRED"),
    ("eligible_count", "INTEGER", "REQUIRED"),
    ("capacity_not_selected_count", "INTEGER", "REQUIRED"),
    ("selected_count", "INTEGER", "REQUIRED"),
    ("ingested_at", "TIMESTAMP", "REQUIRED"),
)

_PROJECT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LOCATION_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_RUN_ID_PATTERN = re.compile(r"^act_[0-9a-f]{24}$")


class ActivationWarehouseError(ValueError):
    """Raised when a BigQuery activation contract is invalid."""


@dataclass(frozen=True)
class ActivationWarehouseConfig:
    """BigQuery destination configuration for activation records."""

    project_id: str = DEFAULT_PROJECT_ID
    dataset_id: str = DEFAULT_DATASET_ID
    decision_table_id: str = DEFAULT_DECISION_TABLE_ID
    run_table_id: str = DEFAULT_RUN_TABLE_ID
    location: str = DEFAULT_LOCATION

    def __post_init__(self) -> None:
        """Validate identifiers before interpolating them into GoogleSQL."""
        if not _PROJECT_PATTERN.fullmatch(self.project_id):
            raise ActivationWarehouseError("project_id contains unsupported characters.")

        for field_name, value in (
            ("dataset_id", self.dataset_id),
            ("decision_table_id", self.decision_table_id),
            ("run_table_id", self.run_table_id),
        ):
            if not _IDENTIFIER_PATTERN.fullmatch(value):
                raise ActivationWarehouseError(f"{field_name} contains unsupported characters.")

        if not _LOCATION_PATTERN.fullmatch(self.location):
            raise ActivationWarehouseError("location contains unsupported characters.")

        if self.decision_table_id == self.run_table_id:
            raise ActivationWarehouseError("Decision and run table IDs must differ.")

    @property
    def decision_table(self) -> str:
        """Return the fully qualified activation-decision table."""
        return f"{self.project_id}.{self.dataset_id}.{self.decision_table_id}"

    @property
    def run_table(self) -> str:
        """Return the fully qualified activation-run table."""
        return f"{self.project_id}.{self.dataset_id}.{self.run_table_id}"


@dataclass(frozen=True)
class ActivationStagingTables:
    """Unique staging tables for one immutable activation run."""

    decision_table: str
    run_table: str


def build_bigquery_schema(
    definition: SchemaDefinition,
) -> tuple[bigquery.SchemaField, ...]:
    """Build an explicit BigQuery schema from the governed definition."""
    return tuple(
        bigquery.SchemaField(
            name=name,
            field_type=field_type,
            mode=mode,
        )
        for name, field_type, mode in definition
    )


def build_staging_tables(
    config: ActivationWarehouseConfig,
    run_id: str,
) -> ActivationStagingTables:
    """Build deterministic, run-specific staging-table names."""
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ActivationWarehouseError("run_id must use the deterministic act_<24 hex> format.")

    suffix = run_id.removeprefix("act_")

    return ActivationStagingTables(
        decision_table=(
            f"{config.project_id}.{config.dataset_id}._activation_decisions_stage_{suffix}"
        ),
        run_table=(f"{config.project_id}.{config.dataset_id}._activation_runs_stage_{suffix}"),
    )


def build_create_activation_tables_query(
    config: ActivationWarehouseConfig,
) -> str:
    """Build idempotent DDL for governed activation destination tables."""
    return f"""
CREATE TABLE IF NOT EXISTS `{config.run_table}` (
    artifact_version INT64 NOT NULL,
    run_id STRING NOT NULL,
    policy_version STRING NOT NULL,
    policy_fingerprint STRING NOT NULL,
    model_name STRING NOT NULL,
    threshold FLOAT64 NOT NULL,
    scoring_artifact_path STRING NOT NULL,
    scoring_artifact_sha256 STRING NOT NULL,
    contact_context_artifact_path STRING NOT NULL,
    contact_context_artifact_sha256 STRING NOT NULL,
    contact_context_source_name STRING NOT NULL,
    contact_context_source_snapshot_reference STRING NOT NULL,
    contact_context_source_query_sha256 STRING NOT NULL,
    contact_context_snapshot_timestamp TIMESTAMP NOT NULL,
    decision_timestamp TIMESTAMP NOT NULL,
    capacity_limit INT64 NOT NULL,
    source_row_count INT64 NOT NULL,
    source_member_count INT64 NOT NULL,
    superseded_count INT64 NOT NULL,
    below_threshold_count INT64 NOT NULL,
    excluded_count INT64 NOT NULL,
    suppressed_count INT64 NOT NULL,
    eligible_count INT64 NOT NULL,
    capacity_not_selected_count INT64 NOT NULL,
    selected_count INT64 NOT NULL,
    ingested_at TIMESTAMP NOT NULL
)
PARTITION BY DATE(decision_timestamp)
CLUSTER BY policy_version, model_name
OPTIONS (
    description = 'Immutable lineage and decision counts for activation runs.'
);

ALTER TABLE `{config.run_table}`
ADD COLUMN IF NOT EXISTS contact_context_artifact_path STRING;

ALTER TABLE `{config.run_table}`
ADD COLUMN IF NOT EXISTS contact_context_artifact_sha256 STRING;

ALTER TABLE `{config.run_table}`
ADD COLUMN IF NOT EXISTS contact_context_source_name STRING;

ALTER TABLE `{config.run_table}`
ADD COLUMN IF NOT EXISTS contact_context_source_snapshot_reference STRING;

ALTER TABLE `{config.run_table}`
ADD COLUMN IF NOT EXISTS contact_context_source_query_sha256 STRING;

ALTER TABLE `{config.run_table}`
ADD COLUMN IF NOT EXISTS contact_context_snapshot_timestamp TIMESTAMP;

CREATE TABLE IF NOT EXISTS `{config.decision_table}` (
    run_id STRING NOT NULL,
    policy_version STRING NOT NULL,
    member_id STRING NOT NULL,
    prediction_date DATE NOT NULL,
    decision_timestamp TIMESTAMP NOT NULL,
    outcome STRING NOT NULL,
    reason_code STRING NOT NULL,
    risk_probability FLOAT64 NOT NULL,
    model_name STRING NOT NULL,
    threshold FLOAT64 NOT NULL,
    intervention_category STRING,
    priority_rank INT64,
    ingested_at TIMESTAMP NOT NULL
)
PARTITION BY prediction_date
CLUSTER BY run_id, outcome, member_id
OPTIONS (
    description = 'Auditable supportive activation decisions from verified Python scores.'
);
""".strip()


def _duplicate_assertion(
    table: str,
    key_columns: tuple[str, ...],
    error_message: str,
) -> str:
    """Build a duplicate-key assertion for one table."""
    struct_fields = ", ".join(key_columns)

    return f"""
ASSERT (
    SELECT
        COUNT(*)
        - COUNT(
            DISTINCT TO_JSON_STRING(
                STRUCT({struct_fields})
            )
        )
    FROM `{table}`
) = 0
AS '{error_message}';
""".strip()


def _conflict_expression(
    columns: tuple[str, ...],
) -> str:
    """Build null-safe comparisons for immutable matched records."""
    return "\n        OR ".join(
        f"target.{column} IS DISTINCT FROM source.{column}" for column in columns
    )


def build_run_merge_query(
    config: ActivationWarehouseConfig,
    staging: ActivationStagingTables,
) -> str:
    """Build an immutable, idempotent activation-run merge."""
    comparison_columns = tuple(column for column in ACTIVATION_RUN_COLUMNS if column != "run_id")
    insert_columns = ",\n        ".join(ACTIVATION_RUN_WAREHOUSE_COLUMNS)
    insert_values = ",\n        ".join(
        f"source.{column}" for column in ACTIVATION_RUN_WAREHOUSE_COLUMNS
    )

    return f"""
{
        _duplicate_assertion(
            staging.run_table,
            ("run_id",),
            "Duplicate run IDs detected in activation-run staging",
        )
    }

{
        _duplicate_assertion(
            config.run_table,
            ("run_id",),
            "Duplicate run IDs detected in the activation-run destination",
        )
    }

ASSERT (
    SELECT COUNT(*)
    FROM `{staging.run_table}` AS source
    INNER JOIN `{config.run_table}` AS target
        ON target.run_id = source.run_id
    WHERE
        {_conflict_expression(comparison_columns)}
) = 0
AS 'An existing activation run conflicts with immutable staging metadata';

MERGE `{config.run_table}` AS target
USING `{staging.run_table}` AS source
ON target.run_id = source.run_id
WHEN NOT MATCHED THEN
    INSERT (
        {insert_columns}
    )
    VALUES (
        {insert_values}
    );
""".strip()


def build_decision_merge_query(
    config: ActivationWarehouseConfig,
    staging: ActivationStagingTables,
) -> str:
    """Build an immutable, idempotent activation-decision merge."""
    key_columns = (
        "run_id",
        "member_id",
        "prediction_date",
    )
    comparison_columns = tuple(
        column for column in ACTIVATION_DECISION_COLUMNS if column not in key_columns
    )
    insert_columns = ",\n        ".join(ACTIVATION_DECISION_WAREHOUSE_COLUMNS)
    insert_values = ",\n        ".join(
        f"source.{column}" for column in ACTIVATION_DECISION_WAREHOUSE_COLUMNS
    )

    return f"""
{
        _duplicate_assertion(
            staging.decision_table,
            key_columns,
            "Duplicate activation decision keys detected in staging",
        )
    }

{
        _duplicate_assertion(
            config.decision_table,
            key_columns,
            "Duplicate activation decision keys detected in the destination",
        )
    }

ASSERT (
    SELECT COUNT(*)
    FROM `{staging.decision_table}` AS source
    INNER JOIN `{config.decision_table}` AS target
        ON target.run_id = source.run_id
        AND target.member_id = source.member_id
        AND target.prediction_date = source.prediction_date
    WHERE
        {_conflict_expression(comparison_columns)}
) = 0
AS 'An existing activation decision conflicts with immutable staging data';

MERGE `{config.decision_table}` AS target
USING `{staging.decision_table}` AS source
ON target.run_id = source.run_id
    AND target.member_id = source.member_id
    AND target.prediction_date = source.prediction_date
WHEN NOT MATCHED THEN
    INSERT (
        {insert_columns}
    )
    VALUES (
        {insert_values}
    );
""".strip()


def _split_assertions_and_merge(
    query: str,
) -> tuple[str, str]:
    """Split a governed merge script into assertions and its MERGE."""
    assertions, separator, merge_body = query.partition("MERGE ")

    if not separator:
        raise ActivationWarehouseError("Governed activation merge query does not contain MERGE.")

    return assertions.strip(), f"MERGE {merge_body.strip()}"


def build_activation_merge_query(
    config: ActivationWarehouseConfig,
    staging: ActivationStagingTables,
) -> str:
    """Build an atomic merge for activation-run and decision records."""
    run_assertions, run_merge = _split_assertions_and_merge(
        build_run_merge_query(
            config,
            staging,
        )
    )
    decision_assertions, decision_merge = _split_assertions_and_merge(
        build_decision_merge_query(
            config,
            staging,
        )
    )

    return f"""
{run_assertions}

{decision_assertions}

BEGIN TRANSACTION;

{run_merge}

{decision_merge}

COMMIT TRANSACTION;
""".strip()
