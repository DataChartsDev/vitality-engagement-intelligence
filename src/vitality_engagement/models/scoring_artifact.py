"""Persist and verify operational scoring predictions and metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from datetime import date
from math import isclose, isfinite
from pathlib import Path
from typing import Final, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from vitality_engagement.activation.schema import ScoredPrediction
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
    PredictionOutputError,
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

    def __post_init__(self) -> None:
        """Validate persisted scoring metadata."""
        if (
            not isinstance(self.artifact_version, int)
            or isinstance(self.artifact_version, bool)
            or self.artifact_version != SCORING_ARTIFACT_VERSION
        ):
            raise ScoringArtifactError("Unsupported scoring artifact version.")

        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ScoringArtifactError("Scoring model name must be a non-empty string.")

        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (int, float))
            or not isfinite(float(self.threshold))
            or float(self.threshold) < 0.0
            or float(self.threshold) > 1.0
        ):
            raise ScoringArtifactError("Scoring threshold must fall between zero and one.")

        if (
            not isinstance(self.row_count, int)
            or isinstance(self.row_count, bool)
            or self.row_count < 1
        ):
            raise ScoringArtifactError("Scoring row_count must be a positive integer.")

        if (
            not isinstance(self.member_count, int)
            or isinstance(self.member_count, bool)
            or self.member_count < 1
            or self.member_count > self.row_count
        ):
            raise ScoringArtifactError("Scoring member_count is invalid.")

        if (
            not isinstance(self.high_risk_count, int)
            or isinstance(self.high_risk_count, bool)
            or self.high_risk_count < 0
            or self.high_risk_count > self.row_count
        ):
            raise ScoringArtifactError("Scoring high_risk_count is invalid.")

        numeric_summaries = (
            self.high_risk_rate,
            self.minimum_probability,
            self.maximum_probability,
            self.mean_probability,
        )

        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            for value in numeric_summaries
        ):
            raise ScoringArtifactError("Scoring probability summaries must be finite.")

        if float(self.high_risk_rate) < 0.0 or float(self.high_risk_rate) > 1.0:
            raise ScoringArtifactError("Scoring high_risk_rate is invalid.")

        expected_high_risk_rate = self.high_risk_count / self.row_count

        if not isclose(
            float(self.high_risk_rate),
            expected_high_risk_rate,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ScoringArtifactError("Scoring high-risk count and rate are inconsistent.")

        minimum_probability = float(self.minimum_probability)
        maximum_probability = float(self.maximum_probability)
        mean_probability = float(self.mean_probability)

        if (
            minimum_probability < 0.0
            or maximum_probability > 1.0
            or minimum_probability > maximum_probability
            or mean_probability < minimum_probability
            or mean_probability > maximum_probability
        ):
            raise ScoringArtifactError("Scoring probability summaries are inconsistent.")

        try:
            minimum_date = date.fromisoformat(self.minimum_prediction_date)
            maximum_date = date.fromisoformat(self.maximum_prediction_date)
        except (TypeError, ValueError) as error:
            raise ScoringArtifactError(
                "Scoring prediction dates must use ISO date format."
            ) from error

        if minimum_date > maximum_date:
            raise ScoringArtifactError("Scoring prediction-date range is invalid.")

        if self.output_columns != PREDICTION_OUTPUT_COLUMNS:
            raise ScoringArtifactError(
                "Scoring metadata columns do not match the prediction contract."
            )


@dataclass(frozen=True)
class VerifiedScoringArtifact:
    """Independently verified persisted scoring inputs."""

    metadata: ScoringArtifactMetadata
    result: PredictionResult
    predictions: tuple[ScoredPrediction, ...]
    prediction_artifact_sha256: str


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


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a persisted artifact."""
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for block in iter(
            lambda: file_handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _load_scoring_metadata(
    metadata_path: Path,
) -> ScoringArtifactMetadata:
    """Load and strictly validate scoring metadata JSON."""
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Scoring metadata artifact does not exist: {metadata_path}")

    raw_payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    if not isinstance(raw_payload, dict):
        raise ScoringArtifactError("Scoring metadata must contain a JSON object.")

    raw_metadata = cast(dict[str, object], raw_payload)
    expected_keys = {field.name for field in fields(ScoringArtifactMetadata)}

    if set(raw_metadata) != expected_keys:
        raise ScoringArtifactError("Scoring metadata fields do not match the governed contract.")

    raw_columns = raw_metadata["output_columns"]

    if not isinstance(raw_columns, list) or not all(
        isinstance(column, str) for column in raw_columns
    ):
        raise ScoringArtifactError("Scoring output_columns must be a string array.")

    return ScoringArtifactMetadata(
        artifact_version=cast(
            int,
            raw_metadata["artifact_version"],
        ),
        model_name=cast(
            str,
            raw_metadata["model_name"],
        ),
        threshold=cast(
            float,
            raw_metadata["threshold"],
        ),
        row_count=cast(
            int,
            raw_metadata["row_count"],
        ),
        member_count=cast(
            int,
            raw_metadata["member_count"],
        ),
        minimum_prediction_date=cast(
            str,
            raw_metadata["minimum_prediction_date"],
        ),
        maximum_prediction_date=cast(
            str,
            raw_metadata["maximum_prediction_date"],
        ),
        high_risk_count=cast(
            int,
            raw_metadata["high_risk_count"],
        ),
        high_risk_rate=cast(
            float,
            raw_metadata["high_risk_rate"],
        ),
        minimum_probability=cast(
            float,
            raw_metadata["minimum_probability"],
        ),
        maximum_probability=cast(
            float,
            raw_metadata["maximum_probability"],
        ),
        mean_probability=cast(
            float,
            raw_metadata["mean_probability"],
        ),
        output_columns=tuple(cast(list[str], raw_columns)),
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


def load_verified_scoring_artifact(
    prediction_path: Path = DEFAULT_PREDICTION_PATH,
    metadata_path: Path = DEFAULT_SCORING_METADATA_PATH,
) -> VerifiedScoringArtifact:
    """Independently verify and load persisted scoring artifacts."""
    if prediction_path.resolve() == metadata_path.resolve():
        raise ScoringArtifactError("Prediction and metadata paths must differ.")

    if not prediction_path.is_file():
        raise FileNotFoundError(f"Scoring prediction artifact does not exist: {prediction_path}")

    metadata = _load_scoring_metadata(metadata_path)
    restored = pd.read_parquet(prediction_path)

    if tuple(str(column) for column in restored.columns) != PREDICTION_OUTPUT_COLUMNS:
        raise ScoringArtifactError("Persisted prediction columns do not match the output contract.")

    if bool(restored["member_id"].isna().any()):
        raise ScoringArtifactError("Persisted prediction member IDs contain null values.")

    if bool(restored["member_id"].astype(str).str.strip().eq("").any()):
        raise ScoringArtifactError("Persisted prediction member IDs contain empty values.")

    if bool(restored["prediction_date"].isna().any()):
        raise ScoringArtifactError("Persisted prediction dates contain null values.")

    if bool(restored[MODEL_NAME_COLUMN].isna().any()):
        raise ScoringArtifactError("Persisted prediction model names contain null values.")

    if not is_bool_dtype(restored[HIGH_RISK_COLUMN].dtype):
        raise ScoringArtifactError("Persisted high-risk classifications must be Boolean.")

    try:
        result = PredictionResult(
            predictions=restored,
            model_name=metadata.model_name,
            threshold=metadata.threshold,
            row_count=metadata.row_count,
        )
    except (
        PredictionOutputError,
        TypeError,
        ValueError,
    ) as error:
        raise ScoringArtifactError(
            "Persisted scoring predictions violate the output contract."
        ) from error

    verify_scoring_artifact(
        prediction_path,
        expected_result=result,
    )

    actual_metadata = build_scoring_metadata(result)

    if actual_metadata != metadata:
        raise ScoringArtifactError(
            "Persisted scoring metadata does not match the prediction artifact."
        )

    ordered = restored.sort_values(
        [
            "prediction_date",
            "member_id",
        ],
        kind="stable",
    )

    predictions = tuple(
        ScoredPrediction(
            member_id=str(row["member_id"]),
            prediction_date=pd.Timestamp(row["prediction_date"]).date(),
            risk_probability=float(row[RISK_PROBABILITY_COLUMN]),
            is_high_risk=bool(row[HIGH_RISK_COLUMN]),
            model_name=str(row[MODEL_NAME_COLUMN]),
            threshold=float(row[THRESHOLD_COLUMN]),
        )
        for row in ordered.to_dict(orient="records")
    )

    return VerifiedScoringArtifact(
        metadata=metadata,
        result=result,
        predictions=predictions,
        prediction_artifact_sha256=_sha256_file(prediction_path),
    )


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
