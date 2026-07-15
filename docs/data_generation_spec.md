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
