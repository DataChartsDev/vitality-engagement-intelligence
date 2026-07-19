"""Tests for governed BigQuery activation contracts."""

from __future__ import annotations

import pytest

from vitality_engagement.activation.bigquery import (
    ACTIVATION_DECISION_SCHEMA,
    ACTIVATION_RUN_SCHEMA,
    ActivationWarehouseConfig,
    ActivationWarehouseError,
    build_activation_merge_query,
    build_bigquery_schema,
    build_create_activation_tables_query,
    build_decision_merge_query,
    build_run_merge_query,
    build_staging_tables,
)


def test_default_warehouse_config_is_explicit_and_region_bound() -> None:
    config = ActivationWarehouseConfig()

    assert config.project_id == "vitality-engagement-43999"
    assert config.dataset_id == "vitality_engagement_dev"
    assert config.location == "asia-southeast1"
    assert (
        config.decision_table == "vitality-engagement-43999."
        "vitality_engagement_dev."
        "activation_decisions"
    )
    assert config.run_table == "vitality-engagement-43999.vitality_engagement_dev.activation_runs"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("project_id", "unsafe`project"),
        ("dataset_id", "unsafe-dataset"),
        ("decision_table_id", "unsafe table"),
        ("run_table_id", "unsafe.table"),
        ("location", "Asia Southeast 1"),
    ],
)
def test_warehouse_config_rejects_unsafe_identifiers(
    field_name: str,
    value: str,
) -> None:
    arguments = {field_name: value}

    with pytest.raises(
        ActivationWarehouseError,
        match="unsupported characters",
    ):
        ActivationWarehouseConfig(**arguments)


def test_warehouse_config_requires_distinct_destination_tables() -> None:
    with pytest.raises(
        ActivationWarehouseError,
        match="must differ",
    ):
        ActivationWarehouseConfig(
            decision_table_id="activation_records",
            run_table_id="activation_records",
        )


def test_bigquery_schemas_match_governed_columns() -> None:
    decision_schema = build_bigquery_schema(ACTIVATION_DECISION_SCHEMA)
    run_schema = build_bigquery_schema(ACTIVATION_RUN_SCHEMA)

    assert decision_schema[0].name == "run_id"
    assert decision_schema[-1].name == "ingested_at"
    assert decision_schema[-1].field_type == "TIMESTAMP"
    assert decision_schema[-1].mode == "REQUIRED"

    assert run_schema[0].name == "artifact_version"
    assert run_schema[1].name == "run_id"
    assert run_schema[-1].name == "ingested_at"


def test_staging_tables_are_deterministic_and_run_specific() -> None:
    config = ActivationWarehouseConfig()
    run_id = "act_0123456789abcdef01234567"

    first = build_staging_tables(config, run_id)
    second = build_staging_tables(config, run_id)

    assert first == second
    assert first.decision_table.endswith("_activation_decisions_stage_0123456789abcdef01234567")
    assert first.run_table.endswith("_activation_runs_stage_0123456789abcdef01234567")


def test_staging_tables_reject_invalid_run_id() -> None:
    with pytest.raises(
        ActivationWarehouseError,
        match="act_<24 hex>",
    ):
        build_staging_tables(
            ActivationWarehouseConfig(),
            "not-a-valid-run-id",
        )


def test_create_tables_query_uses_governed_partitioning_and_clustering() -> None:
    query = build_create_activation_tables_query(ActivationWarehouseConfig())

    assert query.count("CREATE TABLE IF NOT EXISTS") == 2
    assert "`vitality-engagement-43999.vitality_engagement_dev.activation_runs`" in query
    assert "`vitality-engagement-43999.vitality_engagement_dev.activation_decisions`" in query
    assert "PARTITION BY DATE(decision_timestamp)" in query
    assert "PARTITION BY prediction_date" in query
    assert "CLUSTER BY policy_version, model_name" in query
    assert "CLUSTER BY run_id, outcome, member_id" in query
    assert "0.467" not in query


def test_run_merge_is_insert_only_and_conflict_checked() -> None:
    config = ActivationWarehouseConfig()
    staging = build_staging_tables(
        config,
        "act_0123456789abcdef01234567",
    )
    query = build_run_merge_query(config, staging)

    assert "MERGE" in query
    assert "ON target.run_id = source.run_id" in query
    assert "WHEN NOT MATCHED THEN" in query
    assert "WHEN MATCHED THEN UPDATE" not in query
    assert "IS DISTINCT FROM" in query
    assert "Duplicate run IDs" in query


def test_decision_merge_uses_complete_immutable_key() -> None:
    config = ActivationWarehouseConfig()
    staging = build_staging_tables(
        config,
        "act_0123456789abcdef01234567",
    )
    query = build_decision_merge_query(config, staging)

    assert "target.run_id = source.run_id" in query
    assert "target.member_id = source.member_id" in query
    assert "target.prediction_date = source.prediction_date" in query
    assert "WHEN NOT MATCHED THEN" in query
    assert "WHEN MATCHED THEN UPDATE" not in query
    assert "target.threshold IS DISTINCT FROM source.threshold" in query
    assert "target.outcome IS DISTINCT FROM source.outcome" in query
    assert "0.467" not in query


def test_combined_merge_is_atomic_and_asserts_before_transaction() -> None:
    config = ActivationWarehouseConfig()
    staging = build_staging_tables(
        config,
        "act_0123456789abcdef01234567",
    )
    query = build_activation_merge_query(config, staging)

    assert query.count("BEGIN TRANSACTION;") == 1
    assert query.count("COMMIT TRANSACTION;") == 1
    assert query.count("MERGE ") == 2
    assert query.index("ASSERT") < query.index("BEGIN TRANSACTION;")
    assert query.index(f"MERGE `{config.run_table}`") > query.index("BEGIN TRANSACTION;")
    assert query.index(f"MERGE `{config.decision_table}`") > query.index(
        f"MERGE `{config.run_table}`"
    )
    assert query.index("COMMIT TRANSACTION;") > query.index(f"MERGE `{config.decision_table}`")
