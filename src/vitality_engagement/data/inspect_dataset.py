"""Inspect the generated synthetic development dataset."""

from pathlib import Path

import pandas as pd

DATA_DIRECTORY = Path("data/sample/generated")
CSV_PATH = DATA_DIRECTORY / "engagement_development.csv"
PARQUET_PATH = DATA_DIRECTORY / "engagement_development.parquet"

EXPECTED_MEMBER_COUNT = 500
EXPECTED_DAY_COUNT = 180
EXPECTED_ROW_COUNT = EXPECTED_MEMBER_COUNT * EXPECTED_DAY_COUNT


def print_heading(title: str) -> None:
    """Print a clearly separated report heading."""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_rate(name: str, values: pd.Series) -> None:
    """Print the percentage of truthy values in a series."""
    rate = values.fillna(False).astype(bool).mean()
    print(f"{name:<38} {rate:>8.2%}")


def inspect_dataset() -> None:
    """Load and inspect the exported development dataset."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet file not found: {PARQUET_PATH}")

    parquet_data = pd.read_parquet(PARQUET_PATH)
    csv_data = pd.read_csv(
        CSV_PATH,
        low_memory=False,
    )

    print_heading("1. Export consistency")

    print(f"CSV shape:                 {csv_data.shape}")
    print(f"Parquet shape:             {parquet_data.shape}")
    print(f"Shapes match:              {csv_data.shape == parquet_data.shape}")
    print(f"Column names match:        {list(csv_data.columns) == list(parquet_data.columns)}")
    print(f"Expected rows:             {EXPECTED_ROW_COUNT:,}")
    print(f"Actual rows:               {len(parquet_data):,}")
    print(f"Column count:              {len(parquet_data.columns)}")

    print_heading("2. Dataset structure")

    member_count = parquet_data["member_id"].nunique()
    duplicate_count = int(parquet_data.duplicated(subset=["member_id", "date"]).sum())

    rows_per_member = parquet_data.groupby("member_id").size()

    print(f"Unique members:            {member_count:,}")
    print(f"Expected members:          {EXPECTED_MEMBER_COUNT:,}")
    print(f"Duplicate member-dates:    {duplicate_count:,}")
    print(f"Minimum rows per member:   {rows_per_member.min():,}")
    print(f"Maximum rows per member:   {rows_per_member.max():,}")
    print(f"Expected days per member:  {EXPECTED_DAY_COUNT:,}")
    print(f"First date:                {parquet_data['date'].min()}")
    print(f"Last date:                 {parquet_data['date'].max()}")

    print_heading("3. Target availability and balance")

    target_column = "will_miss_goal_next_7_days"

    labelled = parquet_data[parquet_data[target_column].notna()].copy()

    unlabelled = parquet_data[parquet_data[target_column].isna()]

    labelled_target = labelled[target_column].astype(bool)

    expected_unlabelled_rows = member_count * 7

    print(f"Labelled rows:             {len(labelled):,}")
    print(f"Unlabelled rows:           {len(unlabelled):,}")
    print(f"Expected unlabelled rows:  {expected_unlabelled_rows:,}")
    print(f"Target positive rate:     {labelled_target.mean():.2%}")
    print(f"Target negative rate:     {(~labelled_target).mean():.2%}")

    print_heading("4. Data-quality rates")

    print_rate(
        "Missing sleep rate",
        parquet_data["sleep_hours_missing"],
    )
    print_rate(
        "Missing app-session rate",
        parquet_data["app_sessions_missing"],
    )
    print_rate(
        "Step outlier rate",
        parquet_data["is_step_outlier"],
    )
    print_rate(
        "Late-record rate",
        parquet_data["is_late_record"],
    )
    print_rate(
        "Activity-category change rate",
        parquet_data["activity_level_changed"],
    )

    print_heading("5. Interaction rates")

    print_rate(
        "Intervention received rate",
        parquet_data["intervention_received"],
    )

    sent = parquet_data[parquet_data["intervention_received"].astype(bool)]

    print(f"Interventions sent:        {len(sent):,}")

    if not sent.empty:
        print_rate(
            "Open rate among sent",
            sent["intervention_opened"],
        )
        print_rate(
            "Click rate among sent",
            sent["intervention_clicked"],
        )

    print_heading("6. Missing values")

    missing_rates = parquet_data.isna().mean().sort_values(ascending=False)

    missing_rates = missing_rates[missing_rates > 0]

    if missing_rates.empty:
        print("No missing values found.")
    else:
        for column, rate in missing_rates.items():
            print(f"{column:<38} {rate:>8.2%}")

    print_heading("7. Key numeric distributions")

    numeric_columns = [
        "daily_steps",
        "active_minutes",
        "sleep_hours",
        "weekly_goal",
        "app_sessions",
        "goal_completion_percentage",
        "future_7_day_active_minutes",
        "record_delay_days",
    ]

    available_numeric_columns = [
        column for column in numeric_columns if column in parquet_data.columns
    ]

    summary = (
        parquet_data[available_numeric_columns].describe(percentiles=[0.01, 0.50, 0.99]).transpose()
    )

    print(
        summary[
            [
                "count",
                "mean",
                "std",
                "min",
                "1%",
                "50%",
                "99%",
                "max",
            ]
        ].round(2)
    )

    print_heading("8. Member-category distributions")

    for column in [
        "age_band",
        "activity_level",
        "reward_profile",
        "intervention_profile",
    ]:
        print()
        print(f"{column}:")
        distribution = (
            parquet_data[["member_id", column]]
            .drop_duplicates("member_id")[column]
            .value_counts(
                normalize=True,
                dropna=False,
            )
        )

        for category, rate in distribution.items():
            print(f"  {str(category):<25} {rate:>8.2%}")

    print_heading("9. Target rate by subgroup")

    subgroup_data = labelled.copy()
    subgroup_data["target"] = labelled_target

    for column in [
        "age_band",
        "activity_level",
        "reward_profile",
    ]:
        print()
        print(f"Target by {column}:")

        subgroup_summary = (
            subgroup_data.groupby(
                column,
                observed=True,
            )["target"]
            .agg(
                labelled_rows="count",
                miss_rate="mean",
            )
            .sort_index()
        )

        subgroup_summary["miss_rate"] = subgroup_summary["miss_rate"].map(
            lambda value: f"{value:.2%}"
        )

        print(subgroup_summary)

    print_heading("10. Inspection result")

    structural_checks = {
        "Expected row count": (len(parquet_data) == EXPECTED_ROW_COUNT),
        "Expected member count": (member_count == EXPECTED_MEMBER_COUNT),
        "No duplicate member-dates": (duplicate_count == 0),
        "Equal observation length": (
            rows_per_member.min() == EXPECTED_DAY_COUNT
            and rows_per_member.max() == EXPECTED_DAY_COUNT
        ),
        "Correct final target gaps": (len(unlabelled) == expected_unlabelled_rows),
        "CSV and Parquet shapes match": (csv_data.shape == parquet_data.shape),
        "CSV and Parquet columns match": (list(csv_data.columns) == list(parquet_data.columns)),
    }

    for check_name, passed in structural_checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status:<6} {check_name}")

    all_passed = all(structural_checks.values())

    print()
    print(f"Overall structural inspection: {'PASS' if all_passed else 'FAIL'}")


if __name__ == "__main__":
    inspect_dataset()
