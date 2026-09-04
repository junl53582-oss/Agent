# PREDICTION_V2_EVENT_AND_EXPECTATION_DATA_ACQUISITION_REPORT

Status: `PREDICTION_V2_DATA_ACQUISITION_BLOCKED`

## Work completed

- Frozen supplier-neutral schemas and PIT/revision rules for announcement documents, analyst estimates, and actual versions.
- Added deterministic validators and a strict pre-release earnings-surprise constructor.
- Performed one bounded Eastmoney schema probe; no response rows were committed or admitted for training.
- Audited candidate import locations without training a model or reading return labels.

## Public analyst-source probe

- Requests this run: 0
- Requests across the immutable probe lineage: 1
- Raw response SHA256: `85a3472f6c7244c4b4494a7a3f7d9682406dc7041fffda5fe3d704fd8775f70a`
- Response currentYear: `2026`
- Forecast fields: `predictLastYearEps, predictLastYearPe, predictNextTwoYearEps, predictNextTwoYearPe, predictNextYearEps, predictNextYearPe, predictThisYearEps, predictThisYearPe`
- Explicit forecast period per value: `False`
- Revision/supersession link: `False`
- Training admission: `REJECTED`. Current/next-year dynamic fields cannot be safely rebound to old report dates.

## Import gates

- announcement_documents: `FAIL` — NO_APPROVED_IMPORT_DELIVERED
- analyst_estimates: `FAIL` — NO_APPROVED_IMPORT_DELIVERED
- actual_versions: `FAIL` — NO_APPROVED_IMPORT_DELIVERED

## Required external action

Request sample extracts from Wind, RESSET, and optionally CSMAR using the committed schema and supplier questionnaire. No purchase should be approved until a sample demonstrates original publication timestamps, stable record IDs, historical revisions, delisted-stock coverage, and permitted local model use.

CNInfo body reconstruction may proceed as a separate archive job, but reconstructed documents remain non-training evidence until announcement-version lineage is proved. Level-2 remains deferred.

## Model boundary

Gen2 and production DAILY prediction remain unchanged. `PREDICTION_V2_BOUNDED_CHALLENGER_EXPERIMENT` was not started.

## Final status

`PREDICTION_V2_DATA_ACQUISITION_BLOCKED`
