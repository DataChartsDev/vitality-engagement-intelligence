# Python Logistic-Regression Baseline Results

## Overview

Stage 4 reproduced the engagement-risk logistic-regression baseline in Python
using scikit-learn.

The model predicts whether a synthetic wellness-programme member will miss their
activity goal during the following seven days.

All results are based entirely on synthetic data. They do not demonstrate
real-world behavioural, medical, health, insurance, intervention, or causal
effectiveness.

## Data and chronological split

The Python workflow uses the leakage-safe feature table created during Stage 3.

| Split      | Date range               |   Rows | Positive rows | Label status |
| ---------- | ------------------------ | -----: | ------------: | ------------ |
| Train      | 2025-01-29 to 2025-04-30 | 46,000 |         7,502 | Labelled     |
| Validation | 2025-05-01 to 2025-05-31 | 15,500 |         3,614 | Labelled     |
| Test       | 2025-06-01 to 2025-06-22 | 11,000 |         2,962 | Labelled     |
| Scoring    | 2025-06-23 to 2025-06-29 |  3,500 | Not available | Unlabelled   |

The model was fitted using training rows only.

The validation split was used to select the classification threshold.

The test split was evaluated using the frozen validation threshold.

## Predictor contract

The Python model uses 47 approved predictors:

* 3 categorical predictors
* 44 numeric predictors

The following fields are excluded from the model:

* Member identifiers
* Prediction dates
* Feature-window boundary dates
* Dataset split labels
* Future target fields
* The prediction target
* Hidden synthetic-generation state
* Intervention-profile fields
* Records unavailable at prediction time

## Preprocessing and model

The scikit-learn pipeline performs the following preprocessing:

### Categorical predictors

* Most-frequent-value imputation
* One-hot encoding
* Unknown categories ignored during transformation

### Numeric predictors

* Median imputation
* Standard scaling

### Classifier

* Logistic regression
* `lbfgs` solver
* Maximum 1,000 iterations
* Random seed 42
* Regularisation parameter `C=1.0`

All preprocessing is fitted on the training split only.

## Validation threshold selection

Thresholds from `0.000` through `1.000` were evaluated in increments of `0.001`.

The threshold was selected by maximising positive-class F1 on validation data.

Deterministic tie-breakers were:

1. Higher recall
2. Lower threshold

The selected threshold was:

```text
0.431
```

The threshold was frozen before the Python logistic test evaluation.

## Validation performance

| Metric                     |  Value |
| -------------------------- | -----: |
| Rows                       | 15,500 |
| Positive rows              |  3,614 |
| Positive rate              | 23.32% |
| ROC-AUC                    | 0.9476 |
| PR-AUC                     | 0.8705 |
| Average precision          | 0.8706 |
| Log loss                   | 0.2377 |
| Brier score                | 0.0733 |
| Frozen threshold           |  0.431 |
| Positive precision         | 0.8156 |
| Positive recall            | 0.7330 |
| Positive F1                | 0.7721 |
| Specificity                | 0.9496 |
| Accuracy                   | 0.8991 |
| Top-decile recall          | 0.4139 |
| Top-decile precision       | 0.9652 |
| Top-decile lift            |   4.14 |
| Expected calibration error | 0.0136 |
| Maximum calibration gap    | 0.0791 |

### Validation confusion matrix

| Outcome         |  Count |
| --------------- | -----: |
| True positives  |  2,649 |
| False positives |    599 |
| True negatives  | 11,287 |
| False negatives |    965 |

## Frozen-threshold test performance

The test split was evaluated using the validation-selected threshold of `0.431`.

The threshold was not retuned using test outcomes.

| Metric                     |  Value |
| -------------------------- | -----: |
| Rows                       | 11,000 |
| Positive rows              |  2,962 |
| Positive rate              | 26.93% |
| ROC-AUC                    | 0.9565 |
| PR-AUC                     | 0.9114 |
| Average precision          | 0.9114 |
| Log loss                   | 0.2259 |
| Brier score                | 0.0690 |
| Frozen threshold           |  0.431 |
| Positive precision         | 0.8400 |
| Positive recall            | 0.7920 |
| Positive F1                | 0.8153 |
| Specificity                | 0.9444 |
| Accuracy                   | 0.9034 |
| Top-decile recall          | 0.3683 |
| Top-decile precision       | 0.9918 |
| Top-decile lift            |   3.68 |
| Expected calibration error | 0.0095 |
| Maximum calibration gap    | 0.0740 |

### Test confusion matrix

| Outcome         | Count |
| --------------- | ----: |
| True positives  | 2,346 |
| False positives |   447 |
| True negatives  | 7,591 |
| False negatives |   616 |

## Comparison with the BigQuery ML baseline

| Metric                      | BigQuery ML test | Python test |
| --------------------------- | ---------------: | ----------: |
| ROC-AUC                     |           0.9412 |      0.9565 |
| Approximate PR-AUC / PR-AUC |           0.8738 |      0.9114 |
| Log loss                    |           0.2727 |      0.2259 |
| Brier score                 |           0.0854 |      0.0690 |
| Positive precision          |           0.7554 |      0.8400 |
| Positive recall             |           0.7863 |      0.7920 |
| Positive F1                 |           0.7706 |      0.8153 |
| Specificity                 |           0.9062 |      0.9444 |
| Accuracy                    |           0.8739 |      0.9034 |
| Top-decile recall           |           0.3612 |      0.3683 |
| Top-decile precision        |           0.9727 |      0.9918 |
| Top-decile lift             |             3.61 |        3.68 |
| Expected calibration error  |           0.0269 |      0.0095 |
| Maximum calibration gap     |           0.1056 |      0.0740 |

The Python logistic baseline performed better on the synthetic test period across
the reported metrics.

The BigQuery PR-AUC value is approximate, while the Python PR-AUC was calculated
from the precision-recall curve. The two values are therefore informative but
not perfectly identical implementations.

## Interpretation

The Python logistic model provides strong ranking performance, probability
quality, threshold performance, and calibration on the synthetic chronological
splits.

The frozen threshold produces a relatively selective classifier:

* High precision
* Strong specificity
* Competitive recall
* Strong positive-class F1

The model also identifies a highly concentrated top-risk group. Approximately
99.18% of the highest-risk test decile contains positive outcomes in the
synthetic test data.

These results do not prove that the model would generalise to real wellness
programme members.

## Methodological limitation

The Python logistic test split was opened before development and selection of
the planned nonlinear Python model.

Therefore:

* The test results remain a valid frozen-threshold evaluation of the Python
  logistic baseline.
* The test split can no longer be described as an untouched final holdout for
  the complete Stage 4 model-comparison process.
* Nonlinear model development must use training and validation data only.
* The logistic test results must not be used to choose nonlinear features,
  hyperparameters, preprocessing, or thresholds.
* Any later test comparison must be described as a previously observed audit
  comparison rather than untouched model-selection evidence.

A genuinely untouched final model comparison would require a newly generated
future synthetic holdout after all model choices have been frozen.

## Current decision

The Python logistic baseline is accepted as the Stage 4 linear benchmark.

The next modelling milestone is a deliberately constrained nonlinear comparison
using histogram gradient boosting.

The nonlinear model must earn its additional complexity through meaningful
validation improvement. Complexity will not be added merely to make the project
appear more advanced.
