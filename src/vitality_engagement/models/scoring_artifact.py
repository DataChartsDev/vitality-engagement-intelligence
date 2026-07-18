"""Persist and verify operational scoring predictions and metadata."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from vitality_engagement.models.load_data import (
    DEFAULT_MODELING_DATA_PATH,
    load_chronological_modeling_data,
)
from vitality_engagement.models.persistence import (
    DEFAULT_METADATA_PATH as DEFAULT_MODEL_METADATA_PATH,
)
from vitality_engagement.models.persistence import DEFAULT_MODEL_PATH
from vitality_engagement.models.predict import (
    HIGH_RISK_COLUMN,
    MODEL_NAME_COLUMN,
    PREDICTION_OUTPUT_COLUMNS,
    RISK_PROBABILITY_COLUMN,
    THRESHOLD_COLUMN,
    PredictionResult,
    score_operational_split,
)

DEFAULT_PREDICTION_PATH: Final = Path(
    "artifacts/scoring/python_logistic_scoring_predictions.parquet"
)
DEFAULT_SCORING_METADATA_PATH: Final = Path(
    "artifacts/scoring/python_logistic_scoring_predictions.metadata.json"
)
SCORING_ARTIFACT_VERSION: Final = 1


class ScoringArtifactError(RuntimeError):
    """Raised when an operational scoring artifact is invalid."""


@dataclass(frozen=True)
class ScoringArtifactMetadata:
    """Summary metadata for one operational prediction artifact."""

    artifact_version: int
    model_name: str
    threshold: float
    row_count: int
    member_count: int
    minimum_prediction_date: str
    maximum_prediction_date: str
    high_risk_count: int
    high_risk_rate: float
    minimum_probability: float
    maximum_probability: float
    mean_probability: float
    output_columns: tuple[str, ...]


def build_scoring_metadata(
    result: PredictionResult,
) -> ScoringArtifactMetadata:
    """Build summary metadata from validated operational predictions."""
    predictions = result.predictions
    prediction_dates = pd.to_datetime(
        predictions["prediction_date"],
        errors="raise",
    )
    probabilities = predictions[RISK_PROBABILITY_COLUMN].to_numpy(dtype=np.float64)
    high_risk = predictions[HIGH_RISK_COLUMN].to_numpy(dtype=np.bool_)

    return ScoringArtifactMetadata(
        artifact_version=SCORING_ARTIFACT_VERSION,
        model_name=result.model_name,
        threshold=result.threshold,
        row_count=result.row_count,
        member_count=int(predictions["member_id"].nunique(dropna=True)),
        minimum_prediction_date=(pd.Timestamp(prediction_dates.min()).date().isoformat()),
        maximum_prediction_date=(pd.Timestamp(prediction_dates.max()).date().isoformat()),
        high_risk_count=int(high_risk.sum()),
        high_risk_rate=float(high_risk.mean()),
        minimum_probability=float(probabilities.min()),
        maximum_probability=float(probabilities.max()),
        mean_probability=float(probabilities.mean()),
        output_columns=tuple(str(column) for column in predictions.columns),
    )


def _temporary_path(output_path: Path) -> Path:
    """Return a temporary path beside the final output."""
    return output_path.parent / f".{output_path.name}.tmp"


def _write_predictions_atomically(
    result: PredictionResult,
    output_path: Path,
) -> None:
    """Write prediction rows atomically to Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_path(output_path)

    if temporary_path.exists():
        temporary_path.unlink()

    result.predictions.to_parquet(
        temporary_path,
        index=False,
    )
    os.replace(temporary_path, output_path)


def _write_metadata_atomically(
    metadata: ScoringArtifactMetadata,
    output_path: Path,
) -> None:
    """Write scoring metadata atomically as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_path(output_path)

    if temporary_path.exists():
        temporary_path.unlink()

    temporary_path.write_text(
        json.dumps(
            asdict(metadata),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)


def verify_scoring_artifact(
    prediction_path: Path,
    *,
    expected_result: PredictionResult,
) -> None:
    """Verify a persisted scoring artifact against its source result."""
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Scoring prediction artifact does not exist: {prediction_path}")

    restored = pd.read_parquet(prediction_path)

    if tuple(str(column) for column in restored.columns) != (PREDICTION_OUTPUT_COLUMNS):
        raise ScoringArtifactError("Persisted prediction columns do not match the output contract.")

    if len(restored) != expected_result.row_count:
        raise ScoringArtifactError("Persisted prediction row count is inconsistent.")

    if bool(restored.duplicated(subset=["member_id", "prediction_date"]).any()):
        raise ScoringArtifactError("Persisted predictions contain duplicate identifiers.")

    probabilities = restored[RISK_PROBABILITY_COLUMN].to_numpy(dtype=np.float64)

    if not bool(np.isfinite(probabilities).all()):
        raise ScoringArtifactError("Persisted predictions contain non-finite probabilities.")

    if bool(((probabilities < 0.0) | (probabilities > 1.0)).any()):
        raise ScoringArtifactError("Persisted probabilities fall outside zero to one.")

    thresholds = restored[THRESHOLD_COLUMN].to_numpy(dtype=np.float64)
    high_risk = restored[HIGH_RISK_COLUMN].to_numpy(dtype=np.bool_)

    if not bool(
        np.allclose(
            thresholds,
            expected_result.threshold,
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise ScoringArtifactError("Persisted thresholds do not match the frozen threshold.")

    if not bool(
        np.array_equal(
            high_risk,
            probabilities >= expected_result.threshold,
        )
    ):
        raise ScoringArtifactError("Persisted high-risk classifications are inconsistent.")

    model_names = set(restored[MODEL_NAME_COLUMN].astype(str).tolist())
    if model_names != {expected_result.model_name}:
        raise ScoringArtifactError("Persisted model names are inconsistent.")


def write_scoring_artifact(
    result: PredictionResult,
    *,
    prediction_path: Path = DEFAULT_PREDICTION_PATH,
    metadata_path: Path = DEFAULT_SCORING_METADATA_PATH,
) -> tuple[Path, Path]:
    """Persist and verify operational predictions plus metadata."""
    if prediction_path.resolve() == metadata_path.resolve():
        raise ScoringArtifactError("Prediction and metadata paths must differ.")

    metadata = build_scoring_metadata(result)

    _write_predictions_atomically(
        result,
        prediction_path,
    )
    _write_metadata_atomically(
        metadata,
        metadata_path,
    )
    verify_scoring_artifact(
        prediction_path,
        expected_result=result,
    )

    return prediction_path, metadata_path


def score_and_write_operational_artifact(
    *,
    data_path: Path = DEFAULT_MODELING_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    model_metadata_path: Path = DEFAULT_MODEL_METADATA_PATH,
    prediction_path: Path = DEFAULT_PREDICTION_PATH,
    scoring_metadata_path: Path = DEFAULT_SCORING_METADATA_PATH,
) -> tuple[Path, Path]:
    """Score the operational split and persist its verified artifacts."""
    data = load_chronological_modeling_data(data_path)
    result = score_operational_split(
        data,
        model_path=model_path,
        metadata_path=model_metadata_path,
    )

    return write_scoring_artifact(
        result,
        prediction_path=prediction_path,
        metadata_path=scoring_metadata_path,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the operational-scoring command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Score the unlabelled operational split and persist verified prediction artifacts."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_MODELING_DATA_PATH,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument(
        "--model-metadata-path",
        type=Path,
        default=DEFAULT_MODEL_METADATA_PATH,
    )
    parser.add_argument(
        "--prediction-path",
        type=Path,
        default=DEFAULT_PREDICTION_PATH,
    )
    parser.add_argument(
        "--scoring-metadata-path",
        type=Path,
        default=DEFAULT_SCORING_METADATA_PATH,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run operational scoring and artifact persistence."""
    arguments = build_argument_parser().parse_args(argv)

    prediction_path, metadata_path = score_and_write_operational_artifact(
        data_path=arguments.data_path,
        model_path=arguments.model_path,
        model_metadata_path=arguments.model_metadata_path,
        prediction_path=arguments.prediction_path,
        scoring_metadata_path=arguments.scoring_metadata_path,
    )

    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    print(f"Predictions: {prediction_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Rows: {metadata_payload['row_count']}")
    print(f"High-risk rows: {metadata_payload['high_risk_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
