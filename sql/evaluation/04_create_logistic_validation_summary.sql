CREATE OR REPLACE TABLE
    `vitality_engagement_dev.engagement_logistic_validation_summary`
OPTIONS (
    description = 'Validation metrics and frozen threshold for the logistic baseline.'
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
            WHERE dataset_split = 'validation'
        ),
        STRUCT(0.5 AS threshold)
    )
),
selected_threshold AS (
    SELECT
        threshold,
        precision,
        recall,
        specificity,
        accuracy,
        f1_score,
        true_positives,
        false_positives,
        true_negatives,
        false_negatives,
        predicted_positive_count
    FROM
        `vitality_engagement_dev.engagement_logistic_validation_thresholds`
    WHERE threshold = 0.467
),
pr_curve AS (
    SELECT
        threshold,
        recall,
        COALESCE(precision, 1.0) AS precision,
        LAG(recall) OVER (
            ORDER BY threshold DESC
        ) AS previous_recall,
        LAG(
            COALESCE(precision, 1.0)
        ) OVER (
            ORDER BY threshold DESC
        ) AS previous_precision
    FROM
        `vitality_engagement_dev.engagement_logistic_validation_thresholds`
),
pr_auc AS (
    SELECT
        SUM(
            (recall - previous_recall)
            * (precision + previous_precision)
            / 2.0
        ) AS approximate_pr_auc
    FROM pr_curve
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
        `vitality_engagement_dev.engagement_logistic_validation_predictions`
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
        `vitality_engagement_dev.engagement_logistic_validation_calibration`
)
SELECT
    standard_metrics.precision
        AS macro_precision_at_0_5,

    standard_metrics.recall
        AS macro_recall_at_0_5,

    standard_metrics.accuracy
        AS accuracy_at_0_5,

    standard_metrics.f1_score
        AS macro_f1_at_0_5,

    standard_metrics.log_loss,
    standard_metrics.roc_auc,

    pr_auc.approximate_pr_auc,

    custom_metrics.base_positive_rate,
    custom_metrics.mean_predicted_probability,
    custom_metrics.brier_score,

    custom_metrics.top_decile_recall,
    custom_metrics.top_decile_precision,
    custom_metrics.top_decile_lift,

    calibration_metrics.expected_calibration_error,
    calibration_metrics.maximum_calibration_gap,

    selected_threshold.threshold
        AS selected_threshold,

    selected_threshold.precision
        AS selected_threshold_positive_precision,

    selected_threshold.recall
        AS selected_threshold_positive_recall,

    selected_threshold.specificity
        AS selected_threshold_specificity,

    selected_threshold.accuracy
        AS selected_threshold_accuracy,

    selected_threshold.f1_score
        AS selected_threshold_positive_f1,

    selected_threshold.predicted_positive_count,

    selected_threshold.true_positives,
    selected_threshold.false_positives,
    selected_threshold.true_negatives,
    selected_threshold.false_negatives

FROM standard_metrics
CROSS JOIN selected_threshold
CROSS JOIN pr_auc
CROSS JOIN custom_metrics
CROSS JOIN calibration_metrics;
