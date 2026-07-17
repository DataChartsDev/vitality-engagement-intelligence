CREATE OR REPLACE MODEL
    `vitality_engagement_dev.engagement_logistic_baseline`
OPTIONS (
    model_type = 'LOGISTIC_REG',
    input_label_cols = ['will_miss_goal_next_7_days'],
    data_split_method = 'NO_SPLIT',
    category_encoding_method = 'DUMMY_ENCODING',
    max_iterations = 30,
    early_stop = TRUE,
    enable_global_explain = TRUE
)
AS
SELECT
    * EXCEPT(dataset_split)
FROM
    `vitality_engagement_dev.engagement_logistic_baseline_input`
WHERE dataset_split = 'train';
