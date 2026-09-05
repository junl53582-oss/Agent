# Prediction V2 historical-data sample request

Please provide a non-production sample before commercial approval. The sample is used only to validate point-in-time semantics, revision lineage, coverage, units, and licensing.

## Analyst estimates

- Individual estimate records, not only today's consensus snapshot.
- A stable estimate ID, institution/analyst identity, source publication timestamp with timezone, forecast period, metric, value and unit.
- Original, revised and withdrawn versions linked through a supersession identifier.
- Historical and delisted A-share securities.
- At least two real revision chains and five years of sample dates.

## Actual results

- Preliminary, original and revised actuals as separate timestamped versions.
- Stable actual IDs, report period, metric, value/unit, publication timestamp and supersession link.
- The original source identifier or an immutable vendor lineage hash.

## Announcement documents

- Announcement ID, source publication timestamp/date, issuer, source URL, PDF hash and extracted-text hash.
- Explicit correction, cancellation or supersession relationship where applicable.
- A statement explaining whether historical documents can be revised in place.

## Commercial and technical confirmation

- Permission to retain raw hashes and locally derived model features.
- Permission for internal historical model evaluation and forward prediction.
- API/export limits, delisted-stock availability and survivorship policy.
- Full field dictionary, units, null semantics and historical restatement policy.

A current-only snapshot, undocumented restatement, unstable identifier, or missing publication time will be rejected.
