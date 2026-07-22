"""Tests for the local-only activation command-line entry point."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from vitality_engagement.activation import cli
from vitality_engagement.activation.policy import ActivationPolicy


@dataclass(frozen=True)
class _FakeMetadata:
    run_id: str
    source_row_count: int
    selected_count: int


@dataclass(frozen=True)
class _FakeDecisionResult:
    metadata: _FakeMetadata


@dataclass(frozen=True)
class _FakeOrchestrationResult:
    decision_result: _FakeDecisionResult
    decision_path: Path
    metadata_path: Path
    review_queue_path: Path
    review_metadata_path: Path


def test_parser_requires_governed_contact_inputs() -> None:
    parser = cli.build_argument_parser()

    with pytest.raises(SystemExit) as error:
        parser.parse_args([])

    assert error.value.code == 2


def test_parser_rejects_naive_decision_timestamp() -> None:
    parser = cli.build_argument_parser()

    with pytest.raises(SystemExit) as error:
        parser.parse_args(
            [
                "--context-path",
                "context.parquet",
                "--context-metadata-path",
                "context.metadata.json",
                "--decision-timestamp",
                "2025-06-30T08:00:00",
            ]
        )

    assert error.value.code == 2


def test_parser_normalises_aware_timestamp_to_utc() -> None:
    arguments = cli.build_argument_parser().parse_args(
        [
            "--context-path",
            "context.parquet",
            "--context-metadata-path",
            "context.metadata.json",
            "--decision-timestamp",
            "2025-06-30T10:00:00+02:00",
        ]
    )

    assert arguments.decision_timestamp == datetime(
        2025,
        6,
        30,
        8,
        0,
        tzinfo=UTC,
    )


def test_help_exposes_no_remote_or_delivery_options() -> None:
    help_text = cli.build_argument_parser().format_help().lower()

    for prohibited_term in (
        "bigquery",
        "warehouse",
        "upload",
        "dispatch",
        "send",
        "outreach",
    ):
        assert prohibited_term not in help_text


def test_main_runs_local_orchestrator_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    decision_path = tmp_path / "activation.parquet"
    metadata_path = tmp_path / "activation.metadata.json"
    review_queue_path = tmp_path / "human_review_queue.parquet"
    review_metadata_path = tmp_path / "human_review_queue.metadata.json"

    def fake_orchestrate_offline_activation(
        **arguments: object,
    ) -> _FakeOrchestrationResult:
        captured.update(arguments)
        return _FakeOrchestrationResult(
            decision_result=_FakeDecisionResult(
                metadata=_FakeMetadata(
                    run_id="act-test-run",
                    source_row_count=4,
                    selected_count=1,
                )
            ),
            decision_path=decision_path,
            metadata_path=metadata_path,
            review_queue_path=review_queue_path,
            review_metadata_path=review_metadata_path,
        )

    monkeypatch.setattr(
        cli,
        "orchestrate_offline_activation",
        fake_orchestrate_offline_activation,
    )

    exit_code = cli.main(
        [
            "--context-path",
            str(tmp_path / "context.parquet"),
            "--context-metadata-path",
            str(tmp_path / "context.metadata.json"),
            "--decision-timestamp",
            "2025-06-30T08:00:00Z",
            "--activation-decision-path",
            str(decision_path),
            "--activation-metadata-path",
            str(metadata_path),
            "--review-queue-path",
            str(review_queue_path),
            "--review-metadata-path",
            str(review_metadata_path),
        ]
    )

    assert exit_code == 0
    assert captured["context_path"] == (tmp_path / "context.parquet")
    assert captured["context_metadata_path"] == (tmp_path / "context.metadata.json")
    assert captured["decision_timestamp"] == datetime(
        2025,
        6,
        30,
        8,
        0,
        tzinfo=UTC,
    )
    assert captured["policy"] == ActivationPolicy()
    assert captured["activation_decision_path"] == decision_path
    assert captured["activation_metadata_path"] == metadata_path
    assert captured["review_queue_path"] == review_queue_path
    assert captured["review_metadata_path"] == review_metadata_path

    output = capsys.readouterr().out

    assert "Run ID: act-test-run" in output
    assert f"Review queue artifact: {review_queue_path}" in output
    assert f"Review queue metadata: {review_metadata_path}" in output
    assert "Source rows audited: 4" in output
    assert "Selected for human review: 1" in output
    assert "Mode: local artifacts only" in output
