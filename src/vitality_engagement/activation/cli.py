"""Command-line entry point for local governed activation decisions."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from vitality_engagement.activation.artifact import (
    DEFAULT_ACTIVATION_DECISION_PATH,
    DEFAULT_ACTIVATION_METADATA_PATH,
)
from vitality_engagement.activation.orchestrator import (
    orchestrate_offline_activation,
)
from vitality_engagement.activation.policy import ActivationPolicy
from vitality_engagement.activation.review_queue import (
    DEFAULT_REVIEW_QUEUE_METADATA_PATH,
    DEFAULT_REVIEW_QUEUE_PATH,
)
from vitality_engagement.models.scoring_artifact import (
    DEFAULT_PREDICTION_PATH,
    DEFAULT_SCORING_METADATA_PATH,
)


def _aware_timestamp(value: str) -> datetime:
    """Parse an ISO timestamp and require an explicit UTC offset."""
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "decision timestamp must be a valid ISO-8601 timestamp"
        ) from error

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise argparse.ArgumentTypeError("decision timestamp must include a timezone offset")

    return timestamp.astimezone(UTC)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the local-only activation command parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Verify governed scoring and contact-context artifacts, "
            "make deterministic activation decisions, and write "
            "local review artifacts."
        )
    )
    parser.add_argument(
        "--context-path",
        type=Path,
        required=True,
        help="Verified member contact-context Parquet artifact.",
    )
    parser.add_argument(
        "--context-metadata-path",
        type=Path,
        required=True,
        help="Metadata JSON for the contact-context artifact.",
    )
    parser.add_argument(
        "--decision-timestamp",
        type=_aware_timestamp,
        required=True,
        help=("Timezone-aware ISO-8601 timestamp for the decision run."),
    )
    parser.add_argument(
        "--scoring-prediction-path",
        type=Path,
        default=DEFAULT_PREDICTION_PATH,
        help="Verified persisted scoring prediction artifact.",
    )
    parser.add_argument(
        "--scoring-metadata-path",
        type=Path,
        default=DEFAULT_SCORING_METADATA_PATH,
        help="Metadata JSON for the scoring prediction artifact.",
    )
    parser.add_argument(
        "--activation-decision-path",
        type=Path,
        default=DEFAULT_ACTIVATION_DECISION_PATH,
        help="Local Parquet path for activation decisions.",
    )
    parser.add_argument(
        "--activation-metadata-path",
        type=Path,
        default=DEFAULT_ACTIVATION_METADATA_PATH,
        help="Local JSON path for activation run metadata.",
    )
    parser.add_argument(
        "--review-queue-path",
        type=Path,
        default=DEFAULT_REVIEW_QUEUE_PATH,
        help="Local Parquet path for the pending human-review queue.",
    )
    parser.add_argument(
        "--review-metadata-path",
        type=Path,
        default=DEFAULT_REVIEW_QUEUE_METADATA_PATH,
        help="Local JSON path for human-review queue metadata.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run verified offline activation orchestration."""
    arguments = build_argument_parser().parse_args(argv)

    result = orchestrate_offline_activation(
        context_path=arguments.context_path,
        context_metadata_path=(arguments.context_metadata_path),
        decision_timestamp=arguments.decision_timestamp,
        policy=ActivationPolicy(),
        scoring_prediction_path=(arguments.scoring_prediction_path),
        scoring_metadata_path=arguments.scoring_metadata_path,
        activation_decision_path=arguments.activation_decision_path,
        activation_metadata_path=arguments.activation_metadata_path,
        review_queue_path=arguments.review_queue_path,
        review_metadata_path=arguments.review_metadata_path,
    )

    metadata = result.decision_result.metadata

    print(f"Run ID: {metadata.run_id}")
    print(f"Decision artifact: {result.decision_path}")
    print(f"Metadata artifact: {result.metadata_path}")
    print(f"Review queue artifact: {result.review_queue_path}")
    print(f"Review queue metadata: {result.review_metadata_path}")
    print(f"Source rows audited: {metadata.source_row_count}")
    print(f"Selected for human review: {metadata.selected_count}")
    print("Mode: local artifacts only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
