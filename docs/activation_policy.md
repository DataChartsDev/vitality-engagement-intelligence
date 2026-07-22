# Stage 5 Activation Policy Contract

## Status

Stage 5 implements a deterministic, capacity-aware, auditable activation workflow for synthetic engagement-risk forecasts.

The workflow independently verifies scoring and contact-context artifacts, applies governed decision rules, preserves immutable lineage, and writes local review artifacts.

The local activation command does not upload data, contact members, send messages, trigger dashboard actions, or dispatch interventions.

## Intended use

The activation workflow produces supportive recommendations for mandatory human review.

It must not be used to:

- Deny benefits
- Change eligibility
- Penalise members
- Make clinical or diagnostic conclusions
- Infer real health status
- Claim causal intervention effects
- Automatically contact or target members

All current data is synthetic. Predictions are forecasts, not confirmed missed-goal outcomes.

## Scoring contract

The verified Stage 4 scoring artifact contains:

- `member_id`
- `prediction_date`
- `risk_probability`
- `is_high_risk`
- `model_name`
- `threshold`

The frozen threshold remains `0.431`. The activation layer may not alter or retune it.

## Contact-context contract

Every legitimate run requires a verified contact-context snapshot containing:

- `member_id`
- `contact_allowed`
- `opted_out`
- `active_case_open`
- `last_contacted_at`
- `interventions_last_28d`
- `context_as_of`

Its metadata must preserve the source name, immutable snapshot reference, source-query digest, artifact digest, timezone-aware snapshot timestamp, row and member counts, and governed output columns.

The context snapshot must not be later than the activation decision timestamp.

The repository does not fabricate or provide a production contact-context artifact.

## Deterministic policy

The engineering defaults are:

- High-risk predictions only
- Maximum prediction age of 7 days
- Contact cooldown of 7 days
- Maximum 2 interventions per member in 28 days
- Maximum 100 selected records per run
- Mandatory human review
- Supportive use only

These defaults are for a synthetic portfolio project. They are not validated clinical, behavioural, legal, or production operating rules.

## Decision order

1. Validate scoring rows and frozen-threshold classifications.
2. Retain the latest prediction per member.
3. Audit older predictions as superseded no-contact decisions.
4. Audit below-threshold predictions as no-contact decisions.
5. Apply missing-context, permission, and opt-out exclusions.
6. Apply recency, active-case, cooldown, and prior-contact suppressions.
7. Rank eligible members deterministically.
8. Apply the per-run capacity limit.
9. Produce supportive recommendations for human review.
10. Write one audit outcome for every scoring row.

## Run identity and lineage

The deterministic run ID incorporates the policy fingerprint, model name, frozen threshold, scoring artifact digest, contact-context artifact and source lineage, contact-context snapshot timestamp, and decision timestamp normalised to UTC.

Changing any governed lineage input produces a different run ID.

## Local artifacts

The verified local outputs are:

- `artifacts/activation/activation_decisions.parquet`
- `artifacts/activation/activation_decisions.metadata.json`
- `artifacts/activation/human_review_queue.parquet`
- `artifacts/activation/human_review_queue.metadata.json`

The human-review queue is a deterministic projection of records already marked `selected_for_review`. It is ordered by priority rank and uses only the fixed status `pending_human_review`.

Its metadata preserves the SHA-256 digests and paths of both source activation artifacts. A queue is invalid when its contents, metadata, or source activation artifacts no longer match.

Generated artifacts are ignored by Git. The activation and review-queue writers validate temporary Parquet and metadata files before atomically replacing final outputs.

## BigQuery boundary

BigQuery persistence is a separate explicit operation and is not exposed by the local CLI.

No upload may occur unless the scoring artifact is verified, the contact context is legitimate and approved, all lineage is verified, the activation artifact passes verification, human and governance approval is recorded, and the destination project, dataset, region, and schema are confirmed.

Warehouse persistence does not authorise outreach.

## Human-review boundary

`selected_for_review` means only that a record may be considered by an authorised reviewer. It does not mean approved for contact, automatically messaged, automatically enrolled, assigned to treatment, diagnosed, or confirmed to disengage.

## Prohibited automation

Stage 5 must not automatically:

- Send email, SMS, notifications, or other outreach
- Trigger dashboard actions
- Open or close cases
- Change benefits or eligibility
- Assign experimental treatment
- Apply penalties
- Make clinical conclusions
- Claim intervention causality

See `docs/activation_runbook.md` for the local operating procedure.
