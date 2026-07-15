# Synthetic Data Generation Specification

## Purpose

Generate realistic synthetic wellness-programme engagement data for developing and testing the Vitality Engagement Intelligence Engine.

The dataset is synthetic and must not be presented as evidence of real-world behavioural or health effectiveness.

## Initial development scope

The first development dataset will contain:

- 500 members
- 180 calendar days per member
- Approximately 90,000 daily records
- A fixed random seed for reproducibility

The dataset may later scale to approximately 20,000 members and 3.6 million daily records.

## Unit of observation

Each row represents one member on one calendar date.

The expected unique key is:

```text
member_id + date
## Latent truth and observed data quality

Future goal outcomes are calculated from the clean latent behavioural process before artificial data-quality problems are added.

Observed feature fields may then contain missing values, outliers, delays, or category changes. This allows the project to test realistic data validation and preprocessing without allowing artificial corruption to redefine the synthetic target.

Data-quality indicator fields are intended primarily for monitoring and validation. They should not automatically be treated as model predictors.
## Weekly-goal calibration

Weekly goals are based on baseline activity level and include a reproducible member-specific difficulty factor.

Goals are stable within each member, rounded to 10-minute increments, and designed to produce a useful but still imbalanced future goal-miss target.

The expected positive target rate for the default development dataset is approximately 12% to 30%. This is a synthetic modelling-design choice and is not presented as a real-world prevalence estimate.
