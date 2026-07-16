CREATE OR REPLACE TABLE
    `vitality_engagement_dev.engagement_logistic_validation_thresholds`
OPTIONS (
    description = 'Positive-class validation metrics across probability thresholds.'
)
AS
WITH thresholds AS (
    SELECT
        threshold_index / 1000.0 AS threshold
    FROM UNNEST(
        GENERATE_ARRAY(0, 1000)
    ) AS threshold_index
),
confusion_counts AS (
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
            NOT predictions.actual_label
            AND predictions.predicted_probability
                < thresholds.threshold
        ) AS true_negatives,

        COUNTIF(
            predictions.actual_label
            AND predictions.predicted_probability
                < thresholds.threshold
        ) AS false_negatives

    FROM
        `vitality_engagement_dev.engagement_logistic_validation_predictions`
            AS predictions
    CROSS JOIN thresholds
    GROUP BY thresholds.threshold
)
SELECT
    threshold,

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

FROM confusion_counts;
