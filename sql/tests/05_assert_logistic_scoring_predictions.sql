DECLARE metrics STRUCT<
    row_count INT64,
    member_count INT64,
    duplicate_key_count INT64,
    minimum_prediction_date DATE,
    maximum_prediction_date DATE,
    null_probability_count INT64,
    invalid_probability_count INT64,
    threshold_mismatch_count INT64,
    high_risk_count INT64
>;

SET metrics = (
    SELECT AS STRUCT
        COUNT(*) AS row_count,

        COUNT(DISTINCT member_id)
            AS member_count,

        COUNT(*)
            - COUNT(
                DISTINCT TO_JSON_STRING(
                    STRUCT(
                        member_id,
                        prediction_date
                    )
                )
            )
            AS duplicate_key_count,

        MIN(prediction_date)
            AS minimum_prediction_date,

        MAX(prediction_date)
            AS maximum_prediction_date,

        COUNTIF(
            predicted_probability IS NULL
        ) AS null_probability_count,

        COUNTIF(
            predicted_probability < 0.0
            OR predicted_probability > 1.0
        ) AS invalid_probability_count,

        COUNTIF(
            predicted_high_risk
            IS DISTINCT FROM
            (predicted_probability >= 0.467)
        ) AS threshold_mismatch_count,

        COUNTIF(predicted_high_risk)
            AS high_risk_count

    FROM
        `vitality_engagement_dev.engagement_logistic_scoring_predictions`
);

ASSERT metrics.row_count = 3500
AS 'Expected exactly 3500 scoring predictions';

ASSERT metrics.member_count = 500
AS 'Expected exactly 500 scoring members';

ASSERT metrics.duplicate_key_count = 0
AS 'Duplicate scoring member-date keys detected';

ASSERT metrics.minimum_prediction_date = DATE '2025-06-23'
AS 'Unexpected minimum scoring date';

ASSERT metrics.maximum_prediction_date = DATE '2025-06-29'
AS 'Unexpected maximum scoring date';

ASSERT metrics.null_probability_count = 0
AS 'Scoring predictions contain null probabilities';

ASSERT metrics.invalid_probability_count = 0
AS 'A scoring probability falls outside zero to one';

ASSERT metrics.threshold_mismatch_count = 0
AS 'A high-risk classification does not use the frozen threshold';

ASSERT metrics.high_risk_count BETWEEN 1 AND 3499
AS 'Scoring output classifies either no rows or all rows as high risk';
