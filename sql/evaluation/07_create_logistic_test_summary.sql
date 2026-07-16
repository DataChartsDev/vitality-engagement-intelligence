CREATE OR REPLACE TABLE
    `vitality_engagement_dev.engagement_logistic_test_summary`
OPTIONS (
    description = 'Untouched test metrics using the validation-frozen threshold.'
)
AS
WITH standard_metrics AS (
    SELECT *
    FROM ML.EVALUATE(
        MODEL
            `vitality_engagement_dev.engagement_logistic_baseline`,
        (
            SELECT
                * EXCEPT(dataset_split)
            FROM
                `vitality_engagement_dev.engagement_logistic_baseline_input`
            WHERE dataset_split = 'test'
        ),
        STRUCT(0.467 AS threshold)
    )
),
confusion_counts AS (
    SELECT
        COUNTIF(
            actual_label
            AND predicted_label_at_frozen_threshold
        ) AS true_positives,

        COUNTIF(
            NOT actual_label
            AND predicted_label_at_frozen_threshold
        ) AS false_positives,

        COUNTIF(
            NOT actual_label
            AND NOT predicted_label_at_frozen_threshold
        ) AS true_negatives,

        COUNTIF(
            actual_label
            AND NOT predicted_label_at_frozen_threshold
        ) AS false_negatives

    FROM
        `vitality_engagement_dev.engagement_logistic_test_predictions`
),
positive_class_metrics AS (
    SELECT
        true_positives,
        false_positives,
        true_negatives,
        false_negatives,

        true_positives + false_positives
            AS predicted_positive_count,

        SAFE_DIVIDE(
            true_positives,
            true_positives + false_positives
        ) AS precision,

        SAFE_DIVIDE(
            true_positives,
            true_positives + false_negatives
        ) AS recall,

        SAFE_DIVIDE(
            true_negatives,
            true_negatives + false_positives
        ) AS specificity,

        SAFE_DIVIDE(
            true_positives + true_negatives,
            true_positives
                + false_positives
                + true_negatives
                + false_negatives
        ) AS accuracy,

        SAFE_DIVIDE(
            2 * true_positives,
            2 * true_positives
                + false_positives
                + false_negatives
        ) AS f1_score

    FROM confusion_counts
),
thresholds AS (
    SELECT
        threshold_index / 1000.0 AS threshold
    FROM UNNEST(
        GENERATE_ARRAY(0, 1000)
    ) AS threshold_index
),
pr_counts AS (
    SELECT
        thresholds.threshold,

        COUNTIF(
            predictions.actual_label
            AND predictions.predicted_probability
                >= thresholds.threshold
        ) AS true_positives,

        COUNTIF(
            NOT predictions.actual_label
            AND predictions.predicted_probability
                >= thresholds.threshold
        ) AS false_positives,

        COUNTIF(
            predictions.actual_label
            AND predictions.predicted_probability
                < thresholds.threshold
        ) AS false_negatives

    FROM
        `vitality_engagement_dev.engagement_logistic_test_predictions`
            AS predictions
    CROSS JOIN thresholds
    GROUP BY thresholds.threshold
),
pr_curve AS (
    SELECT
        threshold,

        SAFE_DIVIDE(
            true_positives,
            true_positives + false_negatives
        ) AS recall,

        COALESCE(
            SAFE_DIVIDE(
                true_positives,
                true_positives + false_positives
            ),
            1.0
        ) AS precision

    FROM pr_counts
),
ordered_pr_curve AS (
    SELECT
        threshold,
        recall,
        precision,

        LAG(recall) OVER (
            ORDER BY threshold DESC
        ) AS previous_recall,

        LAG(precision) OVER (
            ORDER BY threshold DESC
        ) AS previous_precision

    FROM pr_curve
),
pr_auc AS (
    SELECT
        SUM(
            (recall - previous_recall)
            * (precision + previous_precision)
            / 2.0
        ) AS approximate_pr_auc

    FROM ordered_pr_curve
    WHERE previous_recall IS NOT NULL
),
ranked_predictions AS (
    SELECT
        actual_label,
        predicted_probability,

        NTILE(10) OVER (
            ORDER BY predicted_probability DESC
        ) AS risk_decile

    FROM
        `vitality_engagement_dev.engagement_logistic_test_predictions`
),
custom_metrics AS (
    SELECT
        SAFE_DIVIDE(
            COUNTIF(actual_label),
            COUNT(*)
        ) AS base_positive_rate,

        AVG(predicted_probability)
            AS mean_predicted_probability,

        AVG(
            POW(
                predicted_probability
                    - IF(actual_label, 1.0, 0.0),
                2
            )
        ) AS brier_score,

        SAFE_DIVIDE(
            COUNTIF(
                risk_decile = 1
                AND actual_label
            ),
            COUNTIF(actual_label)
        ) AS top_decile_recall,

        SAFE_DIVIDE(
            COUNTIF(
                risk_decile = 1
                AND actual_label
            ),
            COUNTIF(risk_decile = 1)
        ) AS top_decile_precision,

        SAFE_DIVIDE(
            SAFE_DIVIDE(
                COUNTIF(
                    risk_decile = 1
                    AND actual_label
                ),
                COUNTIF(risk_decile = 1)
            ),
            SAFE_DIVIDE(
                COUNTIF(actual_label),
                COUNT(*)
            )
        ) AS top_decile_lift

    FROM ranked_predictions
),
calibration_metrics AS (
    SELECT
        SAFE_DIVIDE(
            SUM(
                row_count
                * absolute_calibration_difference
            ),
            SUM(row_count)
        ) AS expected_calibration_error,

        MAX(
            absolute_calibration_difference
        ) AS maximum_calibration_gap

    FROM
        `vitality_engagement_dev.engagement_logistic_test_calibration`
)
SELECT
    0.467 AS frozen_threshold,

    standard_metrics.precision
        AS macro_precision_at_frozen_threshold,

    standard_metrics.recall
        AS macro_recall_at_frozen_threshold,

    standard_metrics.accuracy
        AS accuracy_at_frozen_threshold,

    standard_metrics.f1_score
        AS macro_f1_at_frozen_threshold,

    standard_metrics.log_loss,
    standard_metrics.roc_auc,

    pr_auc.approximate_pr_auc,

    positive_class_metrics.precision
        AS positive_precision,

    positive_class_metrics.recall
        AS positive_recall,

    positive_class_metrics.specificity,
    positive_class_metrics.accuracy,

    positive_class_metrics.f1_score
        AS positive_f1,

    custom_metrics.base_positive_rate,
    custom_metrics.mean_predicted_probability,
    custom_metrics.brier_score,

    custom_metrics.top_decile_recall,
    custom_metrics.top_decile_precision,
    custom_metrics.top_decile_lift,

    calibration_metrics.expected_calibration_error,
    calibration_metrics.maximum_calibration_gap,

    positive_class_metrics.predicted_positive_count,
    positive_class_metrics.true_positives,
    positive_class_metrics.false_positives,
    positive_class_metrics.true_negatives,
    positive_class_metrics.false_negatives

FROM standard_metrics
CROSS JOIN positive_class_metrics
CROSS JOIN pr_auc
CROSS JOIN custom_metrics
CROSS JOIN calibration_metrics;
