CREATE OR REPLACE TABLE
    `vitality_engagement_dev.engagement_logistic_validation_predictions`
OPTIONS (
    description = 'Logistic-regression predictions for the untouched validation period.'
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
            WHERE dataset_split = 'validation'
        )
    )
)
SELECT
    will_miss_goal_next_7_days AS actual_label,

    SAFE_CAST(
        predicted_will_miss_goal_next_7_days AS BOOL
    ) AS predicted_label_at_0_5,

    (
        SELECT probability_entry.prob
        FROM UNNEST(
            predicted_will_miss_goal_next_7_days_probs
        ) AS probability_entry
        WHERE SAFE_CAST(
            probability_entry.label AS BOOL
        ) IS TRUE
        LIMIT 1
    ) AS predicted_probability

FROM predictions;
