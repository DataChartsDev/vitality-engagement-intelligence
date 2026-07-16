# BigQuery ML Baseline Results

## Overview

This document records the BigQuery ML logistic-regression baseline developed
during Stage 3 of the Vitality Engagement Intelligence Engine.

The model predicts whether a synthetic wellness-programme member will miss
their activity goal during the next seven days.

All results in this document come from fully synthetic data. They demonstrate
technical implementation and evaluation discipline only. They do not prove
real-world behavioural, medical, or health effectiveness.

## Model

Model name:

```text
engagement_logistic_baseline

Model type:

BigQuery ML logistic regression

Prediction target:

will_miss_goal_next_7_days

The model was trained using only the chronological training split.

BigQuery ML was configured with:

Logistic regression
No internal random data split
Dummy encoding for categorical variables
Early stopping
Global explanation support
A maximum of 30 iterations
Leakage controls

The baseline does not use the following as predictors:

member_id
prediction_date
Feature-window dates
dataset_split
future_7_day_active_minutes
next_week_goal_completed
The target column itself
Hidden synthetic-generation state
intervention_profile_as_of
Records unavailable at the prediction timestamp

Features use the previous 28 calendar days only.

A source record is included only when:

available_date <= prediction_date

Current-day activity and future activity are excluded from each feature row.

Chronological split

The model uses a chronological split rather than a random split.

Split	Date range	Rows	Label status
Train	2025-01-29 to 2025-04-30	46,000	Labelled
Validation	2025-05-01 to 2025-05-31	15,500	Labelled
Test	2025-06-01 to 2025-06-22	11,000	Labelled
Scoring	2025-06-23 to 2025-06-29	3,500	Unlabelled

The final seven days are reserved for operational scoring because their future
seven-day outcomes are unavailable.

Temporal target shift

The positive target rate increased over time.

Split	Positive rows	Positive rate
Train	7,502	16.31%
Validation	3,614	23.32%
Test	2,962	26.93%

This temporal shift is intentionally preserved.

Using a random split would hide part of the realistic deployment challenge and
produce a less trustworthy estimate of future performance.

Training behaviour

Training loss decreased across the reported iterations.

Iteration	Training loss
0	0.6072
1	0.5021
2	0.3993
3	0.3324
4	0.3037
5	0.2979
6	0.2941
7	0.2888
8	0.2825
9	0.2759

Evaluation loss was not calculated internally because the model used:

DATA_SPLIT_METHOD = 'NO_SPLIT'

Validation and test evaluation were instead performed using the explicit
chronological periods defined by the repository.

Threshold selection

The default probability threshold of 0.5 was evaluated but was not assumed
to be optimal.

Validation precision, recall, specificity, accuracy, and positive-class F1
were calculated at 1,001 thresholds between:

0.000

and:

1.000

The selected validation threshold was:

0.467

The threshold was frozen before the test period was evaluated.

It was not retuned after viewing test performance.

Validation comparison
Metric	Threshold 0.467	Threshold 0.500
Positive precision	0.7416	0.7512
Positive recall	0.7504	0.7286
Positive F1	0.7460	0.7397
Accuracy	0.8808	0.8805
Predicted positives	3,657	3,505

The selected threshold improved positive-class F1 and recall while adding only
152 predicted interventions across the validation month.

Validation results
Metric	Result
ROC-AUC	0.9307
Approximate PR-AUC	0.8342
Log loss	0.2742
Brier score	0.0857
Positive precision	0.7416
Positive recall	0.7504
Positive F1	0.7460
Specificity	0.9205
Accuracy	0.8808
Top-decile recall	0.4037
Top-decile precision	0.9413
Top-decile lift	4.04
Expected calibration error	0.0238
Maximum calibration gap	0.0899
Validation confusion matrix
Outcome	Count
True positives	2,712
False positives	945
True negatives	10,941
False negatives	902
Validation probability comparison
Measure	Result
Actual positive rate	23.32%
Mean predicted probability	24.71%
Difference	+1.39 percentage points

The model mildly overpredicted aggregate validation risk.

Validation calibration
Bin	Mean predicted probability	Observed positive rate	Difference
1	0.19%	0.00%	+0.19 pp
2	0.66%	0.19%	+0.46 pp
3	1.26%	1.03%	+0.23 pp
4	2.12%	1.87%	+0.25 pp
5	3.66%	6.84%	-3.18 pp
6	7.94%	8.06%	-0.12 pp
7	21.02%	22.65%	-1.63 pp
8	42.84%	34.97%	+7.88 pp
9	72.41%	63.42%	+8.99 pp
10	95.01%	94.13%	+0.88 pp

The clearest validation weakness appears in bins 8 and 9, where the model is
overconfident.

The highest-risk bin remains well calibrated.

Untouched test results

The test period was evaluated once using the validation-frozen threshold of:

0.467
Metric	Result
ROC-AUC	0.9412
Approximate PR-AUC	0.8738
Log loss	0.2727
Brier score	0.0854
Positive precision	0.7554
Positive recall	0.7863
Positive F1	0.7706
Specificity	0.9062
Accuracy	0.8739
Top-decile recall	0.3612
Top-decile precision	0.9727
Top-decile lift	3.61
Expected calibration error	0.0269
Maximum calibration gap	0.1056
Test confusion matrix
Outcome	Count
True positives	2,329
False positives	754
True negatives	7,284
False negatives	633

The frozen threshold classified:

3,083

of the 11,000 test rows as high risk.

The test period contained:

2,962

actual positive rows.

Validation and test comparison
Metric	Validation	Test
ROC-AUC	0.9307	0.9412
Approximate PR-AUC	0.8342	0.8738
Brier score	0.0857	0.0854
Positive precision	0.7416	0.7554
Positive recall	0.7504	0.7863
Positive F1	0.7460	0.7706
Top-decile precision	0.9413	0.9727
Top-decile recall	0.4037	0.3612
Top-decile lift	4.04	3.61
Expected calibration error	0.0238	0.0269
Maximum calibration gap	0.0899	0.1056

Ranking and positive-class classification performance improved on the test
period.

Brier score remained stable, suggesting probability error did not materially
degrade.

Top-decile lift decreased because the test period had a higher overall
positive rate.

Calibration weakened slightly and remains the main limitation of the baseline.

Test calibration
Bin	Mean predicted probability	Observed positive rate	Difference
1	0.17%	0.00%	+0.17 pp
2	0.56%	0.82%	-0.26 pp
3	1.10%	0.91%	+0.19 pp
4	1.88%	2.00%	-0.12 pp
5	3.66%	4.00%	-0.34 pp
6	10.27%	13.27%	-3.00 pp
7	29.50%	28.64%	+0.87 pp
8	56.20%	46.00%	+10.20 pp
9	86.92%	76.36%	+10.56 pp
10	98.49%	97.27%	+1.21 pp

The model is overconfident in the eighth and ninth probability bins.

The highest-risk test bin remains strongly calibrated:

Measure	Result
Mean predicted probability	98.49%
Observed positive rate	97.27%

No recalibration was fitted using test data.

Operational scoring output

The final seven-day scoring period is unlabelled.

The scoring output contains:

Measure	Result
Rows	3,500
Members	500
Minimum date	2025-06-23
Maximum date	2025-06-29
Duplicate member-date keys	0
Null probabilities	0
High-risk classifications	1,091

The minimum predicted probability was approximately:

0.000046

The maximum predicted probability was approximately:

0.999735

The 1,091 high-risk classifications use the frozen threshold of 0.467.

These rows are operational model outputs only. Their labels are unavailable,
so they cannot be used to estimate scoring-period accuracy, precision, recall,
or effectiveness.

Interpretation

The logistic-regression baseline demonstrates strong synthetic ranking
performance.

The model is particularly effective at concentrating positive cases in the
highest-risk decile.

The baseline also remains reasonably stable across chronological validation
and test periods despite a substantial increase in target prevalence.

The main technical weakness is probability overconfidence in some
middle-to-high risk bands.

Potential future work includes:

Calibration using validation data
Comparison with nonlinear Python models
Stability analysis across time periods
Subgroup performance analysis
Feature importance and explainability review
Monitoring of prevalence and calibration drift
Controlled intervention experiments
Limitations

All results come from synthetic data generated by this project.

The results demonstrate:

BigQuery ingestion
SQL feature engineering
Leakage-safe historical joins
Chronological model validation
BigQuery ML implementation
Threshold selection
Probability evaluation
Calibration analysis
Reproducible operational scoring

They do not demonstrate:

Real-world behavioural effectiveness
Medical or health effectiveness
Causal intervention impact
Fairness in a real population
Generalisation to actual wellness-programme members
Production readiness for sensitive decisions

The model must not be used for:

Medical diagnosis
Healthcare treatment
Insurance eligibility
Insurance pricing
Employment decisions
Real-world automated intervention
Claims of proven behavioural effectiveness

A real deployment would require privacy review, security review, legal review,
fairness testing, real-world validation, monitoring, and controlled
experimentation.
