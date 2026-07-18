"""Persist and load the selected Stage 4 logistic model safely."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import asdict, dataclass
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Final, cast

from sklearn.pipeline import Pipeline

from vitality_engagement.models.baseline import fit_logistic_baseline
from vitality_engagement.models.load_data import ChronologicalModelingData
from vitality_engagement.models.schema import (
    CATEGORICAL_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)
from vitality_engagement.models.selection import PYTHON_LOGISTIC_SELECTION

DEFAULT_MODEL_PATH: Final = Path("models/python_logistic_baseline.pkl")
DEFAULT_METADATA_PATH: Final = Path("models/python_logistic_baseline.metadata.json")
ARTIFACT_VERSION: Final = 1


class ModelPersistenceError(RuntimeError):
    """Raised when a model artifact cannot be validated or persisted."""


@dataclass(frozen=True)
class ModelArtifactMetadata:
    """Metadata required to validate a persisted model artifact."""

    artifact_version: int
    model_name: str
    selected_threshold: float
    selection_split: str
    selection_metric: str
    training_start_date: str
    training_end_date: str
    validation_start_date: str
    validation_end_date: str
    model_feature_columns: tuple[str, ...]
    categorical_feature_columns: tuple[str, ...]
    numeric_feature_columns: tuple[str, ...]
    schema_fingerprint: str
    python_version: str
    scikit_learn_version: str
    pandas_version: str
    numpy_version: str


def calculate_schema_fingerprint() -> str:
    """Return a stable SHA-256 fingerprint of the predictor contract."""
    schema_payload = {
        "model_features": list(MODEL_FEATURE_COLUMNS),
        "categorical_features": list(CATEGORICAL_FEATURE_COLUMNS),
        "numeric_features": list(NUMERIC_FEATURE_COLUMNS),
    }
    encoded_payload = json.dumps(
        schema_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded_payload).hexdigest()


def build_model_metadata() -> ModelArtifactMetadata:
    """Build metadata for the selected Stage 4 model."""
    import platform

    return ModelArtifactMetadata(
        artifact_version=ARTIFACT_VERSION,
        model_name=PYTHON_LOGISTIC_SELECTION.model_name,
        selected_threshold=PYTHON_LOGISTIC_SELECTION.threshold,
        selection_split=PYTHON_LOGISTIC_SELECTION.selection_split,
        selection_metric=PYTHON_LOGISTIC_SELECTION.selection_metric,
        training_start_date=date(2025, 1, 29).isoformat(),
        training_end_date=date(2025, 4, 30).isoformat(),
        validation_start_date=(PYTHON_LOGISTIC_SELECTION.validation_start_date.isoformat()),
        validation_end_date=(PYTHON_LOGISTIC_SELECTION.validation_end_date.isoformat()),
        model_feature_columns=MODEL_FEATURE_COLUMNS,
        categorical_feature_columns=CATEGORICAL_FEATURE_COLUMNS,
        numeric_feature_columns=NUMERIC_FEATURE_COLUMNS,
        schema_fingerprint=calculate_schema_fingerprint(),
        python_version=platform.python_version(),
        scikit_learn_version=version("scikit-learn"),
        pandas_version=version("pandas"),
        numpy_version=version("numpy"),
    )


def _write_pickle_atomically(
    pipeline: Pipeline,
    output_path: Path,
) -> None:
    """Write the fitted pipeline atomically using the pickle protocol."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")

    if temporary_path.exists():
        temporary_path.unlink()

    with temporary_path.open("wb") as artifact_file:
        pickle.dump(
            pipeline,
            artifact_file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    os.replace(temporary_path, output_path)


def _write_metadata_atomically(
    metadata: ModelArtifactMetadata,
    output_path: Path,
) -> None:
    """Write model metadata atomically as formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")

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


def save_selected_model(
    data: ChronologicalModelingData,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> tuple[Path, Path]:
    """Fit and persist the selected training-only logistic pipeline."""
    fitted_result = fit_logistic_baseline(data)
    metadata = build_model_metadata()

    _write_pickle_atomically(
        fitted_result.pipeline,
        model_path,
    )
    _write_metadata_atomically(
        metadata,
        metadata_path,
    )

    return model_path, metadata_path


def load_model_metadata(
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> ModelArtifactMetadata:
    """Load and validate persisted model metadata."""
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model metadata file does not exist: {metadata_path}")

    raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if not isinstance(raw_metadata, dict):
        raise ModelPersistenceError("Model metadata must contain a JSON object.")

    try:
        metadata = ModelArtifactMetadata(
            artifact_version=int(raw_metadata["artifact_version"]),
            model_name=str(raw_metadata["model_name"]),
            selected_threshold=float(raw_metadata["selected_threshold"]),
            selection_split=str(raw_metadata["selection_split"]),
            selection_metric=str(raw_metadata["selection_metric"]),
            training_start_date=str(raw_metadata["training_start_date"]),
            training_end_date=str(raw_metadata["training_end_date"]),
            validation_start_date=str(raw_metadata["validation_start_date"]),
            validation_end_date=str(raw_metadata["validation_end_date"]),
            model_feature_columns=tuple(
                str(value) for value in raw_metadata["model_feature_columns"]
            ),
            categorical_feature_columns=tuple(
                str(value) for value in raw_metadata["categorical_feature_columns"]
            ),
            numeric_feature_columns=tuple(
                str(value) for value in raw_metadata["numeric_feature_columns"]
            ),
            schema_fingerprint=str(raw_metadata["schema_fingerprint"]),
            python_version=str(raw_metadata["python_version"]),
            scikit_learn_version=str(raw_metadata["scikit_learn_version"]),
            pandas_version=str(raw_metadata["pandas_version"]),
            numpy_version=str(raw_metadata["numpy_version"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ModelPersistenceError("Model metadata is incomplete or invalid.") from error

    validate_model_metadata(metadata)
    return metadata


def validate_model_metadata(
    metadata: ModelArtifactMetadata,
) -> None:
    """Validate persisted metadata against the current code contract."""
    if metadata.artifact_version != ARTIFACT_VERSION:
        raise ModelPersistenceError("Model artifact version is not supported.")

    if metadata.model_name != PYTHON_LOGISTIC_SELECTION.model_name:
        raise ModelPersistenceError("Persisted model name does not match the selected model.")

    if metadata.selected_threshold != PYTHON_LOGISTIC_SELECTION.threshold:
        raise ModelPersistenceError("Persisted threshold does not match the frozen threshold.")

    if metadata.model_feature_columns != MODEL_FEATURE_COLUMNS:
        raise ModelPersistenceError("Persisted predictor columns do not match the current schema.")

    if metadata.categorical_feature_columns != CATEGORICAL_FEATURE_COLUMNS:
        raise ModelPersistenceError("Persisted categorical predictors do not match the schema.")

    if metadata.numeric_feature_columns != NUMERIC_FEATURE_COLUMNS:
        raise ModelPersistenceError("Persisted numeric predictors do not match the schema.")

    if metadata.schema_fingerprint != calculate_schema_fingerprint():
        raise ModelPersistenceError("Persisted schema fingerprint is invalid.")


def load_selected_model(
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> tuple[Pipeline, ModelArtifactMetadata]:
    """Load a trusted persisted model and validate its metadata.

    Pickle files can execute arbitrary code during loading. Only load artifacts
    created by this project from trusted local or controlled storage.
    """
    metadata = load_model_metadata(metadata_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact file does not exist: {model_path}")

    with model_path.open("rb") as artifact_file:
        loaded_object = pickle.load(artifact_file)  # noqa: S301

    if not isinstance(loaded_object, Pipeline):
        raise ModelPersistenceError("Persisted artifact is not a scikit-learn Pipeline.")

    pipeline = cast(Pipeline, loaded_object)

    if not hasattr(pipeline, "classes_"):
        raise ModelPersistenceError("Persisted pipeline does not appear to be fitted.")

    return pipeline, metadata
