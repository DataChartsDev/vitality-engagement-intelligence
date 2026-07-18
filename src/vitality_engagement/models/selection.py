"""Frozen model-selection decisions made using validation data only."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final, Literal

SelectionSplit = Literal["validation"]


class ModelSelectionError(ValueError):
    """Raised when a frozen model-selection record is invalid."""


@dataclass(frozen=True)
class FrozenModelSelection:
    """A model and threshold selected exclusively on validation data."""

    model_name: str
    selection_split: SelectionSplit
    selection_metric: str
    threshold: float
    validation_start_date: date
    validation_end_date: date
    validation_row_count: int
    validation_positive_count: int
    validation_positive_f1: float

    def __post_init__(self) -> None:
        """Validate the frozen selection record."""
        if not self.model_name:
            raise ModelSelectionError("Model name must not be empty.")

        if self.selection_split != "validation":
            raise ModelSelectionError("Model selection must use the validation split.")

        if not self.selection_metric:
            raise ModelSelectionError("Selection metric must not be empty.")

        if self.threshold < 0.0 or self.threshold > 1.0:
            raise ModelSelectionError("Frozen threshold must fall between zero and one.")

        if self.validation_start_date > self.validation_end_date:
            raise ModelSelectionError("Validation date range is invalid.")

        if self.validation_row_count <= 0:
            raise ModelSelectionError("Validation row count must be positive.")

        if (
            self.validation_positive_count <= 0
            or self.validation_positive_count > self.validation_row_count
        ):
            raise ModelSelectionError("Validation positive count is invalid.")

        if self.validation_positive_f1 < 0.0 or self.validation_positive_f1 > 1.0:
            raise ModelSelectionError("Validation F1 must fall between zero and one.")


PYTHON_LOGISTIC_SELECTION: Final = FrozenModelSelection(
    model_name="python_logistic_baseline",
    selection_split="validation",
    selection_metric="positive_f1",
    threshold=0.431,
    validation_start_date=date(2025, 5, 1),
    validation_end_date=date(2025, 5, 31),
    validation_row_count=15_500,
    validation_positive_count=3_614,
    validation_positive_f1=0.7720781113378025,
)
