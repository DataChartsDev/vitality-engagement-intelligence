# BigQuery and SQL Architecture

## Overview

Stage 3 implements the cloud data, SQL feature-engineering, modelling,
evaluation, and operational scoring foundation for the Vitality Engagement
Intelligence Engine.

The architecture uses Google BigQuery and BigQuery ML to create a reproducible,
leakage-controlled machine-learning workflow. It preserves chronological
evaluation discipline and separates labelled model-development data from
unlabelled operational scoring data.

All project data is synthetic. Results from this system do not establish
real-world behavioural, medical, health, insurance, or causal effectiveness.

## Cloud configuration

The implementation uses the following Google Cloud resources:

| Resource                    | Value                       |
| --------------------------- | --------------------------- |
| Google Cloud project        | `vitality-engagement-43999` |
| BigQuery dataset            | `vitality_engagement_dev`   |
| Region                      | `asia-southeast1`           |
| Active gcloud configuration | `vitality-engagement`       |

Only the BigQuery API was required for Stage 3. This keeps the architecture
focused and avoids introducing cloud services that are not yet necessary.

## Architecture layers

The Stage 3 data flow consists of the following layers:

```text
Synthetic Parquet dataset
        |
        v
engagement_raw
        |
        v
engagement_staging
        |
        v
engagement_features_28d
        |
        v
engagement_modeling_split
        |
        +-------------------+-------------------+------------------+
        |                   |                   |                  |
        v                   v                   v                  v
engagement_train   engagement_validation   engagement_test   engagement_scoring
        |
        v
engagement_logistic_baseline_input
        |
        v
engagement_logistic_baseline
        |
        +-------------------+-------------------+------------------+
        |                   |                   |                  |
        v                   v                   v                  v
Validation          Threshold and       Untouched test       Operational
predictions         calibration         evaluation           scoring
```

Each major layer has a repository-managed SQL file and an accompanying
assertion suite.

## Raw ingestion layer

The generated Parquet dataset is loaded into:

```text
engagement_raw
```

The table contains:

* 90,000 rows
* 34 columns
* 500 synthetic members
* 180 days per member
* Dates from 2025-01-01 to 2025-06-29
* 86,500 labelled rows
* 3,500 unlabelled final-window rows

The raw table preserves the exported synthetic dataset without performing
silent imputation or feature engineering.

## Staging layer

The staging table is:

```text
engagement_staging
```

It is created by:

```text
sql/staging/01_create_engagement_staging.sql
```

The staging layer:

* Preserves all source fields
* Renames `date` to `activity_date`
* Converts date-like values to BigQuery `DATE`
* Preserves missing values
* Preserves delayed-record information
* Avoids silent imputation
* Provides a stable SQL-facing schema

The associated assertions are stored in:

```text
sql/tests/01_assert_engagement_staging.sql
```

They validate row counts, member counts, key uniqueness, date boundaries,
required fields, allowed categories, missingness indicators, delayed-record
logic, target consistency, target null states, and configured data-quality
ranges.

## Leakage-safe feature layer

The feature table is:

```text
engagement_features_28d
```

It is created by:

```text
sql/features/01_create_engagement_features_28d.sql
```

Each row represents one member and one prediction date.

Features use a historical 28-day window:

```text
prediction_date - 28 days
through
prediction_date - 1 day
```

The prediction date itself is excluded from the feature window.

Records are also restricted using:

```sql
available_date <= prediction_date
```

This prevents a record from being used before it would have been available to
the system.

The table contains:

* 76,000 rows
* 500 members
* 55 columns
* Prediction dates from 2025-01-29 to 2025-06-29
* 3,500 unlabelled operational scoring rows
* Between 26 and 28 available records per historical window

The feature assertions are stored in:

```text
sql/tests/02_assert_engagement_features_28d.sql
```

They validate:

* Member-date uniqueness
* Prediction-date boundaries
* Final-window label null states
* Historical-window definitions
* Available-day counts
* Future-record leakage
* Unavailable-record leakage
* Derived rates
* Trend calculations

## Chronological split

The modelling split table is:

```text
engagement_modeling_split
```

It is created by:

```text
sql/splits/01_create_chronological_modeling_split.sql
```

The split is chronological rather than random:

| Split      | Date range               |   Rows | Members | Label status |
| ---------- | ------------------------ | -----: | ------: | ------------ |
| Train      | 2025-01-29 to 2025-04-30 | 46,000 |     500 | Labelled     |
| Validation | 2025-05-01 to 2025-05-31 | 15,500 |     500 | Labelled     |
| Test       | 2025-06-01 to 2025-06-22 | 11,000 |     500 | Labelled     |
| Scoring    | 2025-06-23 to 2025-06-29 |  3,500 |     500 | Unlabelled   |

The following views expose each partition:

```text
engagement_train
engagement_validation
engagement_test
engagement_scoring
```

The split assertions are stored in:

```text
sql/tests/03_assert_chronological_modeling_split.sql
```

They verify exact row counts, member counts, date boundaries, label states,
assignment completeness, and the absence of chronological overlap.

A random split was deliberately avoided because it would obscure temporal
target shift and could produce an unrealistically favourable evaluation.

## Model input layer

The BigQuery ML input view is:

```text
engagement_logistic_baseline_input
```

It is created by:

```text
sql/models/01_create_logistic_baseline_input.sql
```

The persisted view uses dataset-qualified references such as:

```text
vitality_engagement_dev.engagement_modeling_split
```

This avoids relying on an inherited default dataset within a stored view
definition.

The model input excludes fields that should not be used as predictors,
including:

* `member_id`
* `prediction_date`
* Feature-window boundary dates
* `dataset_split`
* `future_7_day_active_minutes`
* `next_week_goal_completed`
* The prediction target
* Hidden synthetic-generation state
* `intervention_profile_as_of`
* Records unavailable at prediction time

## BigQuery ML baseline

The baseline model is:

```text
engagement_logistic_baseline
```

It is trained by:

```text
sql/models/02_train_logistic_baseline.sql
```

The model configuration uses:

* `LOGISTIC_REG`
* `NO_SPLIT`
* Dummy encoding
* A maximum of 30 iterations
* Early stopping
* Global explainability

`NO_SPLIT` is intentional because the repository already defines explicit
chronological train, validation, test, and scoring partitions.

Only the chronological training period is supplied to model fitting.

## Validation and threshold selection

Validation predictions are stored in:

```text
engagement_logistic_validation_predictions
```

Thresholds from `0.000` through `1.000`, in increments of `0.001`, are evaluated
using:

```text
engagement_logistic_validation_thresholds
```

The selected validation threshold is:

```text
0.467
```

This threshold is frozen before test evaluation. It is not retuned using the
test period.

Validation calibration and summary tables are:

```text
engagement_logistic_validation_calibration
engagement_logistic_validation_summary
```

The relevant repository SQL files are stored under:

```text
sql/evaluation/
```

## Untouched test evaluation

Test predictions are stored in:

```text
engagement_logistic_test_predictions
```

Test calibration and summary outputs are:

```text
engagement_logistic_test_calibration
engagement_logistic_test_summary
```

The test period is evaluated once using the validation-selected threshold of
`0.467`.

The test set is not used to select features, tune the model, choose the
threshold, or fit a calibration adjustment.

This preserves the test period as an independent estimate of temporal
generalisation on the synthetic dataset.

## Operational scoring

Unlabelled predictions are stored in:

```text
engagement_logistic_scoring_predictions
```

They are created by:

```text
sql/evaluation/08_create_logistic_scoring_predictions.sql
```

The scoring output contains:

* 3,500 rows
* 500 members
* Prediction dates from 2025-06-23 to 2025-06-29
* No duplicate member-date keys
* Complete probability values
* 1,091 rows classified as high risk at threshold `0.467`

These rows are operational predictions only. Their future targets are
intentionally unavailable, so they must not be presented as confirmed outcomes.

## Evaluation and scoring assertions

Evaluation assertions are stored in:

```text
sql/tests/04_assert_logistic_baseline_evaluation.sql
```

They validate:

* Validation and test row counts
* Frozen-threshold consistency
* Probability completeness
* Confusion-matrix totals
* ROC-AUC validity
* PR-AUC relative to prevalence
* Brier-score ranges
* F1 validity
* Top-decile lift
* Calibration-error limits

Scoring assertions are stored in:

```text
sql/tests/05_assert_logistic_scoring_predictions.sql
```

They validate:

* Row and member counts
* Member-date uniqueness
* Date boundaries
* Probability completeness
* Probability ranges
* Frozen-threshold consistency
* Non-degenerate scoring output

## Reproducible SQL execution

Repository SQL files are executed using:

```text
scripts/run_bigquery_sql.ps1
```

The PowerShell runner:

* Executes version-controlled SQL files
* Supports dry runs
* Uses the active Google Cloud project when no project is supplied
* Handles Windows PowerShell 5.1 stderr behaviour
* Applies a query billing ceiling

Using repository-managed SQL files avoids fragile multiline SQL variables and
creates a reviewable history of cloud transformations.

## Cost controls

Stage 3 uses a deliberately small architecture:

* One BigQuery dataset
* Version-controlled SQL
* BigQuery ML for the baseline model
* No unnecessary orchestration service
* No unnecessary always-on compute
* Query billing limits in the SQL runner
* Assertions designed to calculate related metrics efficiently

Additional services should only be introduced when later stages have a clear
operational requirement.

## Design strengths

The architecture provides:

* Reproducible transformations
* Chronological evaluation
* Explicit leakage controls
* Availability-aware feature engineering
* Validation-only threshold selection
* An untouched test period
* Separate unlabelled scoring data
* SQL-based data-quality checks
* Version-controlled cloud logic
* A low-cost Stage 3 implementation

## Limitations

The current architecture has important limitations:

* All data is synthetic
* The model has not been validated on real members
* Predictive performance does not establish causal intervention effectiveness
* Temporal shift is simulated rather than observed in production
* Calibration weaknesses remain in some high-probability ranges
* The BigQuery ML baseline has not yet been compared with Python models
* Deployment, intervention selection, LLM generation, and monitoring are not
  included in Stage 3

These limitations must remain visible when the project is presented publicly.

## Stage 3 boundary

Stage 3 establishes the cloud data and modelling baseline.

Stage 4 will reproduce and compare the modelling workflow in Python while
preserving:

* The existing chronological split
* Leakage controls
* Validation-only model selection
* The untouched test period
* Calibration discipline
* Reproducibility
* Public GitHub code quality
