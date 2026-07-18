# Model Card: Engagement Risk Logistic Baseline

## Model summary

| Field                      | Value                                         |
| -------------------------- | --------------------------------------------- |
| Model name                 | `python_logistic_baseline`                    |
| Model family               | Binary logistic regression                    |
| Implementation             | scikit-learn pipeline                         |
| Stage                      | Stage 4 — Python modelling                    |
| Status                     | Selected and frozen                           |
| Positive class             | Project-defined missed-goal risk class        |
| Decision threshold         | `0.431`                                       |
| Threshold selection split  | Validation                                    |
| Threshold selection metric | Positive-class F1                             |
| Training data type         | Synthetic behavioural data                    |
| Operational output         | Risk probability and high-risk classification |

The selected model estimates the probability that a member belongs to the project’s positive missed-goal risk class. It produces a continuous risk probability and a Boolean high-risk classification using the frozen validation-selected threshold of `0.431`.

The model is a portfolio and engineering demonstration. It is not a clinical, diagnostic, actuarial, eligibility, or punitive decision system.

## Intended use

The model is intended to support:

* Prioritisation of members for supportive engagement interventions
* Demonstration of leakage-safe temporal model development
* Comparison of SQL and Python modelling approaches
* Reproducible model persistence and operational scoring
* Monitoring of risk-score distributions and intervention workloads

Predictions should support human-reviewed engagement workflows. They should not be used as the sole basis for denying benefits, changing eligibility, penalising members, or making health-related conclusions.

## Data lineage

The modelling dataset is exported from the governed BigQuery feature layer to:

```text
data/modeling/engagement_modeling_split.parquet
```

The export contains:

* 76,000 total rows
* 500 synthetic members
* 47 approved predictors
* 3 categorical predictors
* 44 numeric predictors
* Member and prediction-date identifiers
* A chronological split designation
* A Boolean target for labelled splits
* Null targets for the operational scoring split

### Chronological splits

| Split      |   Rows | Purpose                                  |
| ---------- | -----: | ---------------------------------------- |
| Train      | 46,000 | Model fitting                            |
| Validation | 15,500 | Threshold selection and model comparison |
| Test       | 11,000 | Frozen logistic-model audit evaluation   |
| Scoring    |  3,500 | Unlabelled operational predictions       |

The split ordering is chronological. Random train-test splitting is not used.

## Predictor contract

The model accepts exactly the predictors defined by:

```text
src/vitality_engagement/models/schema.py
```

The input contract contains:

* 47 model predictors
* 3 categorical predictors
* 44 numeric predictors
* No member identifiers as predictors
* No prediction date as a predictor
* No target column as a predictor
* No split column as a predictor
* No prohibited future or outcome-derived columns

The prediction interface rejects missing, additional, or incorrectly ordered feature columns.

## Preprocessing

### Categorical predictors

Categorical predictors use:

1. Most-frequent-value imputation
2. One-hot encoding
3. Unknown-category tolerance during scoring

### Numeric predictors

Numeric predictors use:

1. Median imputation
2. Standard scaling

A numeric coefficient therefore represents the change in log odds associated with an approximate one-standard-deviation increase in that transformed predictor, holding the other model inputs constant.

## Model specification

The selected estimator is configured as:

| Parameter                    | Value                |
| ---------------------------- | -------------------- |
| Estimator                    | `LogisticRegression` |
| Regularisation parameter `C` | `1.0`                |
| Solver                       | `lbfgs`              |
| Maximum iterations           | `1000`               |
| Random state                 | `42`                 |

The complete preprocessing and estimator sequence is persisted as one scikit-learn `Pipeline`.

## Model selection

Two Python candidates were evaluated using the validation split only:

1. Logistic regression
2. Histogram gradient boosting

### Validation comparison

| Metric                     | Logistic regression | Histogram gradient boosting | Better model |
| -------------------------- | ------------------: | --------------------------: | ------------ |
| ROC-AUC                    |              0.9476 |                      0.9404 | Logistic     |
| PR-AUC                     |              0.8705 |                      0.8508 | Logistic     |
| Average precision          |              0.8706 |                      0.8509 | Logistic     |
| Log loss                   |              0.2377 |                      0.2549 | Logistic     |
| Brier score                |              0.0733 |                      0.0792 | Logistic     |
| Positive-class F1          |              0.7721 |                      0.7499 | Logistic     |
| Specificity                |              0.9496 |                      0.9223 | Logistic     |
| Accuracy                   |              0.8991 |                      0.8828 | Logistic     |
| Top-decile lift            |              4.1395 |                      4.0620 | Logistic     |
| Expected calibration error |              0.0136 |                      0.0172 | Logistic     |
| Maximum calibration gap    |              0.0791 |                      0.0924 | Logistic     |

Histogram gradient boosting achieved slightly higher recall at its own validation-selected threshold:

* Logistic recall: `0.7330`
* Histogram gradient boosting recall: `0.7532`

However, this was accompanied by materially lower precision, F1, discrimination, calibration, accuracy, and top-decile lift.

The nonlinear candidate therefore did not justify its additional complexity. Logistic regression was retained as the frozen Stage 4 model.

The histogram gradient boosting model was not evaluated on the test split because it had already been rejected using validation evidence.

## Validation performance

The selected logistic model was trained only on the training split and evaluated on the validation split.

| Metric                     | Result |
| -------------------------- | -----: |
| Rows                       | 15,500 |
| Positive rows              |  3,614 |
| Positive-class prevalence  | 0.2332 |
| ROC-AUC                    | 0.9476 |
| PR-AUC                     | 0.8705 |
| Average precision          | 0.8706 |
| Log loss                   | 0.2377 |
| Brier score                | 0.0733 |
| Selected threshold         |  0.431 |
| Precision                  | 0.8156 |
| Recall                     | 0.7330 |
| Positive-class F1          | 0.7721 |
| Specificity                | 0.9496 |
| Accuracy                   | 0.8991 |
| True positives             |  2,649 |
| False positives            |    599 |
| True negatives             | 11,287 |
| False negatives            |    965 |
| Top-decile recall          | 0.4139 |
| Top-decile precision       | 0.9652 |
| Top-decile lift            | 4.1395 |
| Expected calibration error | 0.0136 |
| Maximum calibration gap    | 0.0791 |

The threshold `0.431` was frozen after validation evaluation and before the selected logistic model was evaluated on the test split.

## Test performance

The frozen logistic model and frozen threshold were evaluated on the chronological test split.

| Metric                     | Result |
| -------------------------- | -----: |
| Rows                       | 11,000 |
| Positive rows              |  2,962 |
| Positive-class prevalence  | 0.2693 |
| ROC-AUC                    | 0.9565 |
| PR-AUC                     | 0.9114 |
| Average precision          | 0.9114 |
| Log loss                   | 0.2259 |
| Brier score                | 0.0690 |
| Frozen threshold           |  0.431 |
| Precision                  | 0.8400 |
| Recall                     | 0.7920 |
| Positive-class F1          | 0.8153 |
| Specificity                | 0.9444 |
| Accuracy                   | 0.9034 |
| True positives             |  2,346 |
| False positives            |    447 |
| True negatives             |  7,591 |
| False negatives            |    616 |
| Top-decile recall          | 0.3683 |
| Top-decile precision       | 0.9918 |
| Top-decile lift            | 3.6833 |
| Expected calibration error | 0.0095 |
| Maximum calibration gap    | 0.0740 |

## Holdout limitation

The logistic model, its hyperparameters, and its threshold were frozen before logistic test evaluation. Its reported test result is therefore valid as an audit of that frozen logistic specification.

However, the test result was observed before development of the histogram gradient boosting candidate. The existing test period can therefore no longer be described as a completely untouched final holdout for the whole Stage 4 model-development process.

To maintain discipline:

* The nonlinear candidate was developed and rejected using train and validation data only.
* Nonlinear test performance was not inspected.
* The test split was not used to select the nonlinear candidate.
* The logistic test result must not be presented as an untouched cross-model selection result.
* A future synthetic time period would be required for a new, fully untouched final model-comparison audit.

## Explainability

The selected model exposes 55 transformed-feature coefficients after preprocessing.

The strongest absolute coefficients included:

| Transformed feature                  | Coefficient | Odds ratio | Model direction          |
| ------------------------------------ | ----------: | ---------: | ------------------------ |
| `avg_daily_steps_28d`                |     -6.3073 |     0.0018 | Decreases predicted risk |
| `weekly_goal_as_of`                  |      3.7714 |    43.4418 | Increases predicted risk |
| `activity_level_as_of_low`           |     -1.0715 |     0.3425 | Decreases predicted risk |
| `step_outlier_day_count_28d`         |      0.8194 |     2.2692 | Increases predicted risk |
| `avg_active_minutes_28d`             |      0.6587 |     1.9323 | Increases predicted risk |
| `sum_active_minutes_28d`             |      0.6220 |     1.8627 | Increases predicted risk |
| `stddev_daily_steps_28d`             |      0.6204 |     1.8597 | Increases predicted risk |
| `reward_profile_as_of_medium`        |     -0.5784 |     0.5608 | Decreases predicted risk |
| `reward_profile_as_of_low`           |     -0.5404 |     0.5825 | Decreases predicted risk |
| `avg_goal_completion_percentage_28d` |     -0.5317 |     0.5876 | Decreases predicted risk |

### Explainability limitations

The coefficient output must be interpreted cautiously:

* The data is synthetic, so effects must not be described as real behavioural or causal findings.
* Coefficients describe conditional model associations, not causal effects.
* Strongly correlated predictors can divide, reverse, or redistribute apparent importance.
* Average and total measures can carry overlapping information.
* Numeric coefficients operate on standardised values rather than original measurement units.
* One-hot encoding retained all categorical levels while the model also fitted an intercept.
* Individual categorical dummy coefficients are therefore not clean reference-category comparisons.
* A large odds ratio does not establish that changing the underlying feature would change the outcome.
* Global coefficients do not explain every individual prediction.

The explanation layer is suitable for technical transparency and model debugging, not causal or clinical interpretation.

## Persistence

The selected fitted pipeline is stored locally at:

```text
models/python_logistic_baseline.pkl
```

Its metadata is stored at:

```text
models/python_logistic_baseline.metadata.json
```

The metadata records:

* Artifact version
* Model name
* Frozen threshold
* Selection split and metric
* Training and validation dates
* Approved feature lists
* Feature-schema fingerprint
* Python version
* scikit-learn version
* pandas version
* NumPy version

The artifact loader validates the metadata against the current source-code contract.

Pickle artifacts can execute arbitrary code during loading. Only model artifacts created by this project and retrieved from trusted controlled storage may be loaded.

## Prediction output contract

Operational scoring returns:

| Column             | Meaning                              |
| ------------------ | ------------------------------------ |
| `member_id`        | Synthetic member identifier          |
| `prediction_date`  | Date associated with the prediction  |
| `risk_probability` | Predicted positive-class probability |
| `is_high_risk`     | `risk_probability >= 0.431`          |
| `model_name`       | Frozen selected model name           |
| `threshold`        | Frozen decision threshold            |

The interface verifies:

* Exact input feature columns
* Exact identifier columns
* Non-empty batches
* Matching identifier and feature row counts
* Non-null identifiers
* Unique member-date identifiers
* Finite probabilities
* Probabilities between zero and one
* Correct threshold application
* Consistent model and threshold metadata

## Operational scoring

The unlabelled scoring split contains:

* 3,500 rows
* 500 members
* Prediction dates from June 23 through June 29, 2025

Generate the verified operational artifacts with:

```powershell
python -m vitality_engagement.models.scoring_artifact
```

The workflow writes:

```text
artifacts/scoring/python_logistic_scoring_predictions.parquet
artifacts/scoring/python_logistic_scoring_predictions.metadata.json
```

Generated data, model, and scoring artifacts are excluded from Git.

Operational high-risk classifications are forecasts. They are not confirmed missed-goal outcomes.

## Reproducibility

Activate the project environment before running the workflow:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run the complete quality gate:

```powershell
ruff format --check .
ruff check .
mypy
pytest -q
pre-commit run --all-files
```

Export or refresh the governed modelling dataset when required:

```powershell
python -m vitality_engagement.models.export_features
```

Persist the selected training-only model:

```powershell
python -c "from vitality_engagement.models.load_data import load_chronological_modeling_data; from vitality_engagement.models.persistence import save_selected_model; save_selected_model(load_chronological_modeling_data())"
```

Generate operational predictions:

```powershell
python -m vitality_engagement.models.scoring_artifact
```

## Known limitations

1. All members and behavioural records are synthetic.
2. Performance does not demonstrate effectiveness on real member populations.
3. The current test period is not an untouched cross-model holdout for all Stage 4 experimentation.
4. The model has not been assessed for real demographic or health-related fairness.
5. Calibration and prevalence may change across future time periods.
6. The threshold reflects validation F1 rather than business capacity or intervention cost.
7. Probability drift and feature drift monitoring are not yet implemented.
8. Individual coefficients are vulnerable to correlated-feature instability.
9. Categorical dummy coefficients do not provide simple reference-level comparisons.
10. Operational scoring currently writes local artifacts rather than serving a deployed endpoint.
11. Model artifacts are not yet registered in a managed model registry.
12. Predictions require human-reviewed, supportive use.

## Stage 4 conclusion

Logistic regression remains the selected Stage 4 model because it outperformed the nonlinear candidate across discrimination, probability quality, calibration, F1, accuracy, specificity, and top-decile lift.

The selected pipeline is:

* Chronologically trained
* Validation selected
* Threshold frozen
* Tested with an explicitly documented holdout limitation
* Globally explainable within stated constraints
* Persistable and reloadable
* Protected by a strict feature contract
* Capable of producing verified operational scoring artifacts
