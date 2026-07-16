CREATE OR REPLACE TABLE
    `vitality_engagement_dev.engagement_logistic_test_predictions`
OPTIONS (
    description = 'Logistic-regression predictions for the untouched test period.'
)
AS
WITH predictions AS (
    SELECT *
    FROM ML.PREDICT(
        MODEL
            `vitality_engagement_dev.engagement_logistic_baseline`,
        (
            SELECT
                * EXCEPT(dataset_split)
            FROM
                `vitality_engagement_dev.engagement_logistic_baseline_input`
            WHERE dataset_split = 'test'
        )
    )
)
SELECT
    will_miss_goal_next_7_days AS actual_label,

    (
        SELECT probability_entry.prob
        FROM UNNEST(
            predicted_will_miss_goal_next_7_days_probs
        ) AS probability_entry
        WHERE SAFE_CAST(
            probability_entry.label AS BOOL
        ) IS TRUE
        LIMIT 1
    ) AS predicted_probability,

    (
        SELECT probability_entry.prob
        FROM UNNEST(
            predicted_will_miss_goal_next_7_days_probs
        ) AS probability_entry
        WHERE SAFE_CAST(
            probability_entry.label AS BOOL
        ) IS TRUE
        LIMIT 1
    ) >= 0.467 AS predicted_label_at_frozen_threshold

FROM predictions;
