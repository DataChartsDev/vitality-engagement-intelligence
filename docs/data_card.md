# Data Card: Vitality Engagement Synthetic Development Dataset

## 1. Dataset overview

**Dataset name:** Vitality Engagement Synthetic Development Dataset
**Version:** 0.1.0
**Status:** Development dataset
**Data type:** Fully synthetic longitudinal member-engagement data
**Primary format:** Parquet
**Secondary format:** CSV

This dataset supports the development of the Vitality Engagement Intelligence Engine. The intended machine-learning task is to predict whether a synthetic wellness-programme member will miss their activity goal during the next seven days.

No real member records, clinical records, wearable-device records, or personally identifiable information are included.

## 2. Dataset dimensions

| Attribute                       |      Value |
| ------------------------------- | ---------: |
| Synthetic members               |        500 |
| Observation days per member     |        180 |
| Total daily records             |     90,000 |
| Columns                         |         34 |
| Observation start               | 2025-01-01 |
| Observation end                 | 2025-06-29 |
| Labelled records                |     86,500 |
| Unlabelled final-window records |      3,500 |

Each row represents one synthetic member on one calendar date.

The expected unique key is:

```text
member_id + date
```

The generated development files are:

```text
data/sample/generated/engagement_development.csv
data/sample/generated/engagement_development.parquet
```

Generated data files are not intended to be committed to Git.

## 3. Intended uses

This dataset is intended for:

* Developing reproducible data pipelines
* Practising SQL and BigQuery workflows
* Training classification models
* Testing time-based train, validation, and test splits
* Evaluating calibration, ranking, and top-decile lift
* Developing behavioural intervention rules
* Demonstrating synthetic uplift-modelling workflows
* Testing data-quality and monitoring logic
* Developing a portfolio-ready API and dashboard

## 4. Uses outside scope

This dataset must not be used to:

* Make decisions about real wellness-programme members
* Infer real population health or behaviour
* Estimate real-world treatment or intervention effects
* Diagnose a medical or psychological condition
* Provide clinical guidance
* Establish causal conclusions about demographic groups
* Claim that a model or intervention is effective in practice

Results produced using this dataset demonstrate technical implementation only.

## 5. Prediction target

The primary target is:

```text
will_miss_goal_next_7_days
```

The target is `True` when a member's active minutes during the seven days immediately after the observation date are below that member's weekly activity goal.

The future outcome fields are:

* `future_7_day_active_minutes`
* `next_week_goal_completed`
* `will_miss_goal_next_7_days`

These fields must not be used as prediction features.

The final seven daily records for every member have missing future outcomes because a complete seven-day prediction window is unavailable.

### Target distribution

| Target status             | Records |              Rate |
| ------------------------- | ------: | ----------------: |
| Will miss goal            |  16,174 |            18.70% |
| Will complete goal        |  70,326 |            81.30% |
| Future window unavailable |   3,500 | 3.89% of all rows |

The positive rate is a synthetic modelling-design choice. It is not an estimate of real-world wellness-programme disengagement.

## 6. Generation process

The dataset is generated in reproducible layers.

### 6.1 Member profiles

Each synthetic member receives:

* A synthetic member ID
* An age band
* Membership duration
* Baseline activity level
* Reward-response profile
* Intervention-response profile

### 6.2 Member-day structure

Each member receives one row for every date in the configured observation period.

### 6.3 Daily behaviour

Daily signals include:

* Steps
* Active minutes
* Sleep hours
* Weekly activity goal
* App sessions

The behavioural process includes member-level variation and random daily variation.

### 6.4 Time-dependent patterns

The generator includes:

* Weekend activity differences
* Seasonal activity variation
* Temporary new-member enthusiasm
* Gradual disengagement for a subset of members

The hidden synthetic mechanism used to create disengagement must not be treated as an observable model feature.

### 6.5 Goal history

Generated goal-related fields include:

* Weekly active minutes accumulated so far
* Goal-completion percentage
* Previous goal streak
* Previous failed goals
* Future seven-day activity
* Future goal-completion label

Historical streak and failure fields use prior weeks only.

### 6.6 Reward and intervention history

The dataset includes:

* Rewards viewed
* Rewards redeemed
* Intervention assignment
* Intervention type
* Intervention opened
* Intervention clicked
* Days since the last app session

Interventions are randomly assigned on eligible weekly opportunity days to support a future synthetic uplift-modelling demonstration.

## 7. Main column groups

### Member attributes

* `member_id`
* `age_band`
* `membership_months`
* `activity_level`
* `reward_profile`
* `intervention_profile`

### Daily engagement

* `date`
* `daily_steps`
* `active_minutes`
* `sleep_hours`
* `weekly_goal`
* `app_sessions`
* `days_since_last_app_session`

### Goal history

* `week_start`
* `weekly_active_minutes_so_far`
* `goal_completion_percentage`
* `previous_goal_streak`
* `previous_failed_goals`

### Reward and intervention history

* `rewards_viewed`
* `rewards_redeemed`
* `intervention_received`
* `intervention_type`
* `intervention_opened`
* `intervention_clicked`

### Outcome fields

* `future_7_day_active_minutes`
* `next_week_goal_completed`
* `will_miss_goal_next_7_days`

### Data-quality metadata

* `sleep_hours_missing`
* `app_sessions_missing`
* `is_step_outlier`
* `is_late_record`
* `record_delay_days`
* `available_date`
* `activity_level_changed`

## 8. Data-quality characteristics

Controlled data-quality issues are added after the clean future target is calculated.

| Data-quality issue         | Observed rate |
| -------------------------- | ------------: |
| Missing sleep values       |         3.07% |
| Missing app-session values |         2.03% |
| Step-count outliers        |         0.47% |
| Late-arriving records      |         2.98% |
| Activity-category changes  |         0.21% |

This ordering preserves the clean latent outcome while making observed feature data imperfect.

The quality-indicator fields are designed mainly for validation and monitoring. They should not automatically be treated as prediction features.

## 9. Missing-value semantics

Missing values do not all have the same meaning.

### Artificial missingness

* Missing `sleep_hours`
* Missing `app_sessions`

These simulate incomplete observed data.

### Structurally unavailable outcomes

The final seven days per member have missing future-outcome values because the required future window does not exist.

### Not-applicable intervention fields

`intervention_opened` and `intervention_clicked` are missing when no intervention was sent. These values mean “not applicable,” not data loss.

## 10. Key distributions

| Variable                        |      Mean | Maximum |
| ------------------------------- | --------: | ------: |
| Daily steps                     |  7,122.48 |  59,959 |
| Active minutes                  |     40.67 |     135 |
| Sleep hours                     |      7.41 |    10.5 |
| Weekly goal                     |    235.58 |     380 |
| App sessions                    |      1.35 |       9 |
| Goal-completion percentage      |    70.18% |  217.3% |
| Future seven-day active minutes |    284.93 |     773 |
| Record delay                    | 0.06 days |  3 days |

Values above 35,000 steps are intentionally generated outliers and are identified by `is_step_outlier`.

Goal-completion percentages may exceed 100 because members can exceed their weekly goal.

## 11. Member distributions

### Age bands

| Age band | Members |
| -------- | ------: |
| 18–24    |  20.00% |
| 25–34    |  29.20% |
| 35–44    |  25.60% |
| 45–54    |  17.20% |
| 55+      |   8.00% |

### Baseline activity levels

| Activity level | Members |
| -------------- | ------: |
| Low            |  32.80% |
| Moderate       |  45.00% |
| High           |  22.20% |

### Reward profiles

| Profile | Members |
| ------- | ------: |
| Low     |  30.40% |
| Medium  |  48.20% |
| High    |  21.40% |

### Intervention-response profiles

| Profile | Members |
| ------- | ------: |
| Low     |  25.40% |
| Medium  |  50.20% |
| High    |  24.40% |

## 12. Subgroup target rates

| Group                 | Future goal-miss rate |
| --------------------- | --------------------: |
| Age 18–24             |                16.71% |
| Age 25–34             |                18.49% |
| Age 35–44             |                20.99% |
| Age 45–54             |                16.26% |
| Age 55+               |                22.33% |
| Low activity          |                21.16% |
| Moderate activity     |                16.92% |
| High activity         |                18.67% |
| Low reward profile    |                21.30% |
| Medium reward profile |                17.13% |
| High reward profile   |                18.52% |

These differences are properties of the synthetic generation process and random composition. They are not evidence of real behavioural differences between demographic groups.

Subgroup performance must be evaluated later to determine whether models amplify these synthetic disparities.

## 13. Interaction summary

| Interaction metric                     | Observed rate |
| -------------------------------------- | ------------: |
| Intervention received per daily record |         4.85% |
| Interventions sent                     |         4,362 |
| Open rate among sent                   |        50.73% |
| Click rate among sent                  |        11.65% |

The daily intervention rate is low because interventions are only eligible on weekly opportunity days.

## 14. Reproducibility

The generator accepts:

* Member count
* Day count
* Random seed
* Fixed start date
* Configurable data-quality rates

The default development configuration uses random seed `42`.

Identical configurations should produce identical datasets.

The generator does not use the current system date as an uncontrolled input.

## 15. Leakage controls

The following controls apply:

* Raw `member_id` must not be used as a predictor.
* Future outcome fields must not be used as predictors.
* Feature windows must use only information available by the observation date.
* Late records must respect `available_date`.
* Random train-test splitting must not be used.
* Model evaluation must use a time-based split.
* Synthetic hidden-generation states must not be exposed as features.

The planned split is:

* Training: months 1–4
* Validation: month 5
* Test: month 6

## 16. Validation status

The development dataset passed the following structural checks:

* Expected row count
* Expected member count
* No duplicate member-date keys
* Equal observation length for all members
* Correct missing future windows
* Matching CSV and Parquet dimensions
* Matching CSV and Parquet columns
* Missingness-indicator consistency
* Step-outlier-indicator consistency
* Record-delay consistency
* Target consistency

## 17. Limitations

The dataset has important limitations:

1. All behaviour is generated from explicit assumptions and mathematical rules.
2. Real wellness behaviour is more complex and less predictable.
3. Demographic groups are simplified categories.
4. No real medical, disability, socioeconomic, geographic, or environmental context is represented.
5. Intervention responses are simulated.
6. Reward behaviour is simulated.
7. Synthetic correlations may appear without an intentional causal mechanism.
8. The target prevalence was calibrated for a useful modelling demonstration.
9. Model performance on this data will not predict performance on real programme data.
10. The dataset cannot demonstrate that any intervention improves health or engagement.

## 18. Ethical considerations

The project must:

* Clearly label the data as synthetic
* Avoid medical diagnosis or clinical advice
* Avoid causal claims about demographic groups
* Report subgroup model performance
* Document target calibration
* Include a no-intervention option
* Distinguish risk prediction from uplift prediction
* Avoid presenting synthetic intervention effects as real evidence

## 19. Maintenance

The data card should be updated whenever changes are made to:

* The schema
* Generation assumptions
* Target definition
* Goal calibration
* Missingness or outlier rates
* Intervention assignment
* Observation period
* Dataset size
* Export formats
* Validation rules

## 20. Responsible interpretation

This dataset demonstrates engineering, analytical, and modelling methods. It does not demonstrate real-world member behaviour, programme effectiveness, health improvement, or intervention impact.
