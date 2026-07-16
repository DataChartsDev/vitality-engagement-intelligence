DECLARE metrics STRUCT<
    validation_row_count INT64,
    test_row_count INT64,
    frozen_threshold FLOAT64,
    validation_roc_auc FLOAT64,
    test_roc_auc FLOAT64,
    validation_pr_auc FLOAT64,
    test_pr_auc FLOAT64,
    validation_brier_score FLOAT64,
    test_brier_score FLOAT64,
    validation_positive_f1 FLOAT64,
    test_positive_f1 FLOAT64,
    validation_top_decile_lift FLOAT64,
    test_top_decile_lift FLOAT64,
    validation_calibration_error FLOAT64,
    test_calibration_error FLOAT64,
    test_probability_null_count INT64,
    test_confusion_total INT64
>;

SET metrics = (
    SELECT AS STRUCT
        (
            SELECT COUNT(*)
            FROM
                `vitality_engagement_dev.engagement_logistic_validation_predictions`
        ) AS validation_row_count,

        (
            SELECT COUNT(*)
            FROM
                `vitality_engagement_dev.engagement_logistic_test_predictions`
        ) AS test_row_count,

        test_summary.frozen_threshold,

        validation_summary.roc_auc
            AS validation_roc_auc,

        test_summary.roc_auc
            AS test_roc_auc,

        validation_summary.approximate_pr_auc
            AS validation_pr_auc,

        test_summary.approximate_pr_auc
            AS test_pr_auc,

        validation_summary.brier_score
            AS validation_brier_score,

        test_summary.brier_score
            AS test_brier_score,

        validation_summary.selected_threshold_positive_f1
            AS validation_positive_f1,

        test_summary.positive_f1
            AS test_positive_f1,

        validation_summary.top_decile_lift
            AS validation_top_decile_lift,

        test_summary.top_decile_lift
            AS test_top_decile_lift,

        validation_summary.expected_calibration_error
            AS validation_calibration_error,

        test_summary.expected_calibration_error
            AS test_calibration_error,

        (
            SELECT COUNTIF(predicted_probability IS NULL)
            FROM
                `vitality_engagement_dev.engagement_logistic_test_predictions`
        ) AS test_probability_null_count,

        test_summary.true_positives
            + test_summary.false_positives
            + test_summary.true_negatives
            + test_summary.false_negatives
            AS test_confusion_total

    FROM
        `vitality_engagement_dev.engagement_logistic_validation_summary`
            AS validation_summary
    CROSS JOIN
        `vitality_engagement_dev.engagement_logistic_test_summary`
            AS test_summary
);

ASSERT metrics.validation_row_count = 15500
AS 'Unexpected validation prediction count';

ASSERT metrics.test_row_count = 11000
AS 'Unexpected test prediction count';

ASSERT metrics.frozen_threshold = 0.467
AS 'The validation-selected threshold changed';

ASSERT metrics.test_probability_null_count = 0
AS 'Test predictions contain null probabilities';

ASSERT metrics.test_confusion_total = metrics.test_row_count
AS 'Test confusion-matrix counts do not match the test size';

ASSERT metrics.validation_roc_auc BETWEEN 0.5 AND 1.0
AS 'Validation ROC-AUC is outside its valid useful range';

ASSERT metrics.test_roc_auc BETWEEN 0.5 AND 1.0
AS 'Test ROC-AUC is outside its valid useful range';

ASSERT metrics.validation_pr_auc
    > (
        SELECT base_positive_rate
        FROM
            `vitality_engagement_dev.engagement_logistic_validation_summary`
    )
AS 'Validation PR-AUC does not exceed the prevalence baseline';

ASSERT metrics.test_pr_auc
    > (
        SELECT base_positive_rate
        FROM
            `vitality_engagement_dev.engagement_logistic_test_summary`
    )
AS 'Test PR-AUC does not exceed the prevalence baseline';

ASSERT metrics.validation_brier_score BETWEEN 0.0 AND 0.25
AS 'Validation Brier score is outside the expected range';

ASSERT metrics.test_brier_score BETWEEN 0.0 AND 0.25
AS 'Test Brier score is outside the expected range';

ASSERT metrics.validation_positive_f1 BETWEEN 0.0 AND 1.0
AS 'Validation positive-class F1 is invalid';

ASSERT metrics.test_positive_f1 BETWEEN 0.0 AND 1.0
AS 'Test positive-class F1 is invalid';

ASSERT metrics.validation_top_decile_lift > 1.0
AS 'Validation top-decile lift does not beat random selection';

ASSERT metrics.test_top_decile_lift > 1.0
AS 'Test top-decile lift does not beat random selection';

ASSERT metrics.validation_calibration_error BETWEEN 0.0 AND 0.10
AS 'Validation calibration error exceeds the accepted baseline limit';

ASSERT metrics.test_calibration_error BETWEEN 0.0 AND 0.10
AS 'Test calibration error exceeds the accepted baseline limit';
