"""Generate, validate, and export synthetic modelling data."""

import argparse
from pathlib import Path

from vitality_engagement.data.generate_engagement import (
    generate_modeling_dataset,
)
from vitality_engagement.data.schema import GenerationConfig
from vitality_engagement.data.validation import (
    validate_modeling_dataset,
)


def export_modeling_dataset(
    config: GenerationConfig,
    output_directory: Path,
) -> tuple[Path, Path]:
    """Generate, validate, and export CSV and Parquet files.

    Args:
        config: Synthetic-data generation configuration.
        output_directory: Directory that will receive the files.

    Returns:
        Paths to the exported CSV and Parquet files.
    """
    data = generate_modeling_dataset(config)
    validate_modeling_dataset(data, config)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = output_directory / "engagement_development.csv"
    parquet_path = output_directory / "engagement_development.parquet"

    data.to_csv(
        csv_path,
        index=False,
    )

    data.to_parquet(
        parquet_path,
        index=False,
    )

    return csv_path, parquet_path


def main() -> None:
    """Run dataset generation and export from the command line."""
    parser = argparse.ArgumentParser(
        description=("Generate the synthetic Vitality engagement dataset.")
    )

    parser.add_argument(
        "--member-count",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--day-count",
        type=int,
        default=180,
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/sample/generated"),
    )

    arguments = parser.parse_args()

    config = GenerationConfig(
        member_count=arguments.member_count,
        day_count=arguments.day_count,
        random_seed=arguments.random_seed,
    )

    csv_path, parquet_path = export_modeling_dataset(
        config,
        arguments.output_directory,
    )

    print(f"CSV exported to: {csv_path}")
    print(f"Parquet exported to: {parquet_path}")


if __name__ == "__main__":
    main()
