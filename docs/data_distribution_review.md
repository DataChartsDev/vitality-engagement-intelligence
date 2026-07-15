# Synthetic Development Dataset Distribution Review

## Review status

**Decision:** Approved for Stage 3 development
**Dataset:** `engagement_development`
**Rows:** 90,000
**Members:** 500
**Observation period:** 2025-01-01 to 2025-06-29

## Structural review

The generated CSV and Parquet exports both contain 90,000 rows and 34 columns.

The dataset contains:

* 500 unique synthetic members
* 180 records per member
* No duplicate member-date records
* 3,500 final-window records without future labels
* Matching CSV and Parquet schemas

All structural inspection checks passed.

## Target review

The original synthetic target rate was 2.94%, which was too rare for a useful portfolio classification problem.

Goal difficulty was recalibrated without directly assigning or manipulating the target. The future goal-miss label remains derived from each member's future seven-day active minutes relative to their weekly goal.

The recalibrated target rate is:

```text
18.70%
```

This produces:

* 16,174 positive labelled records
* 70,326 negative labelled records
* 86,500 total labelled records

This level of class imbalance is suitable for demonstrating:

* Precision-recall evaluation
* Probability calibration
* Risk ranking
* Top-decile lift
* Targeted intervention selection
* Subgroup performance analysis

The rate is a synthetic modelling choice and must not be interpreted as real-world prevalence.

## Activity and goal review

The main numeric distributions are internally coherent:

* Mean daily steps: 7,122
* Mean active minutes: 40.67
* Mean weekly goal: 235.58 active minutes
* Mean future seven-day active minutes: 284.93
* Mean goal-completion percentage: 70.18%

Step counts above 35,000 are intentionally generated outliers and are flagged.

Goal-completion percentage can exceed 100 when a member exceeds their goal.

## Data-quality review

Observed data-quality rates closely match their configured probabilities:

| Issue                     | Observed |
| ------------------------- | -------: |
| Missing sleep values      |    3.07% |
| Missing app sessions      |    2.03% |
| Step outliers             |    0.47% |
| Late records              |    2.98% |
| Activity-category changes |    0.21% |

The controlled defects are suitable for later data-quality testing and monitoring.

## Intervention review

Interventions appear on 4.85% of daily records because assignment occurs only on weekly opportunity days.

Among sent interventions:

* Open rate: 50.73%
* Click rate: 11.65%

Missing intervention-open and intervention-click values are expected when no intervention was sent.

## Subgroup review

Future goal-miss rates vary by subgroup.

Activity-level miss rates are:

* Low: 21.16%
* Moderate: 16.92%
* High: 18.67%

Age-band miss rates range from 16.26% to 22.33%.

Reward-profile miss rates range from 17.13% to 21.30%.

These differences are accepted for the synthetic development dataset but must not be interpreted as causal or representative of real populations.

Future modelling work must report subgroup performance and determine whether trained models amplify these synthetic differences.

## Leakage review

The following fields are prohibited as model inputs:

* `member_id`
* `future_7_day_active_minutes`
* `next_week_goal_completed`
* `will_miss_goal_next_7_days`
* Any hidden synthetic-generation state

Late-arriving fields must be filtered using `available_date` so that features reflect only information available at prediction time.

The future model split must be chronological rather than random.

## Final decision

The dataset is approved for:

* BigQuery loading
* SQL staging
* SQL data-quality checks
* Feature-table development
* BigQuery ML baseline training

The dataset is not approved for real-world behavioural, medical, demographic, or intervention-effect claims.
