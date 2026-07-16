CREATE OR REPLACE TABLE
    `vitality_engagement_dev.engagement_logistic_test_calibration`
OPTIONS (
    description = 'Probability calibration deciles for the logistic test predictions.'
)
AS
WITH binned_predictions AS (
    SELECT
        actual_label,
        predicted_probability,
        NTILE(10) OVER (
            ORDER BY predicted_probability
        ) AS calibration_bin
    FROM
        `vitality_engagement_dev.engagement_logistic_test_predictions`
)
SELECT
    calibration_bin,
    COUNT(*) AS row_count,
    MIN(predicted_probability) AS minimum_probability,
    MAX(predicted_probability) AS maximum_probability,
    AVG(predicted_probability) AS mean_predicted_probability,
    AVG(
        IF(actual_label, 1.0, 0.0)
    ) AS observed_positive_rate,
    AVG(predicted_probability)
        - AVG(IF(actual_label, 1.0, 0.0))
        AS calibration_difference,
    ABS(
        AVG(predicted_probability)
        - AVG(IF(actual_label, 1.0, 0.0))
    ) AS absolute_calibration_difference
FROM binned_predictions
GROUP BY calibration_bin;
