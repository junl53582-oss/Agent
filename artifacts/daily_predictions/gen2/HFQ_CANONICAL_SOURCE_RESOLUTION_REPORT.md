# HFQ_CANONICAL_SOURCE_RESOLUTION_REPORT

## Final Status

`DAILY_PREDICTION_STILL_BLOCKED`

The historical canonical HFQ lineage was identified, but the repository has no
Tencent-normalized source covering 2026-09-03 for the three affected symbols.
The existing immutable daily partition is Eastmoney-normalized and must not be
overwritten or silently replaced. No prediction retry was therefore permitted.

## 1. Root Cause

`HFQ_MISMATCH_ROOT_CAUSE = CROSS_PROVIDER_HFQ_LINEAGE_CHANGE_TENCENT_TO_EASTMONEY`

The V10 historical caches for `000630`, `000792`, and `601607` were produced by
the repository's Tencent normalization path. Their `amount` values exactly obey
the Tencent adapter formula, and their volume series contain the fractional
units produced by that adapter.

The immutable 2026-09-03 acquisition selected Eastmoney for all three symbols.
Classifying all 300 retained incremental cache files by the repository's own
normalizer signature yields exactly 238 Eastmoney and 62 Tencent files, matching
the immutable source receipt. This independently identifies the three failed
files as Eastmoney.

Both providers label the series HFQ, but their adjusted price histories are not
related by one sufficiently stable OHLC factor. A particular dividend, rights
issue, split, suspension, or resumption is not required to explain the failure
and was not asserted without provider-specific evidence. The proven cause is
the provider-lineage change.

## 2. Missing Provenance

The historical V10 manifest records only aggregate source counts; it does not
record a per-symbol provider map. The 2026-09-03 receipt likewise records the
238/62 aggregate split but not the provider selected for each symbol.

Per-symbol lineage is nevertheless recoverable through deterministic repository
normalization signatures and the retained raw caches. What is missing is a
Tencent-normalized cache covering 2026-09-03 for the three failed symbols.
Repository search covered the current worktree, four related worktrees, all Git
objects and all branches. The latest matching canonical caches end on
2026-08-28.

No new provider request was made. The immutable partition was not overwritten.

## 3. Canonical Source

- Historical canonical semantics: Tencent HFQ normalized by
  `research_v10.history_data._normalize_hfq`.
- Historical raw source: the retained
  `data/raw_v10_hfq/*_2010-01-01_2026-08-21_hfq.csv` caches.
- Last independently retained matching incremental evidence: the three
  `data/prediction_forward/v30r1/cache_hfq/*_2026-08-01_2026-08-28_hfq.csv`
  files.
- The 2026-08-28 audit records all three as accepted with stitch factor `1.0`,
  ratio deviation `0.0`, and return difference `0.0`.
- Canonical Tencent evidence for 2026-09-03: **MISSING**.

The installed aggregate was rebuilt using the original V10 builder and retained
raw caches. For the three symbols, installed open/high/low/close/volume values
are exactly equal to their retained long caches. Exact float comparison finds
only CSV amount serialization tails: maximum absolute difference
`3.725290298461914e-09`.

## 4. SHA

- Installed historical aggregate:
  `efaa21915d24665a63215816d4d0c3f1713202e48f4c0581672e225fd86ded47`
- Locked original aggregate identity:
  `fcc1ea387719476410f8d4dcc49840e74bd087aec351ed02cb40d1318dd45fc3`
- Immutable 2026-09-03 market partition:
  `9b8d17147ff6273a4440b5b33cc047bf5c5e88266dd7926ecc87f087aa002353`
- Immutable source receipt:
  `135c8ef786eeb035f5019561f535f7c92b949421fe565c3e65f94834763e8c4a`
- Overlap diagnostic:
  `23d9f5d15aed4fe2345a252290dcf2235a244c21cc966236a54a0aa627997456`
- Machine-readable resolution report:
  `27af4734fc072eae335a10ed6f3f199bfb9beb4ce8f910e4cd7d62759955372b`

`CANONICAL_HASH_MATCH: FALSE`

The byte-identical `fcc1...` aggregate is absent from Git and local retained
artifacts. Its OHLC training semantics are preserved by the retained caches and
the reconstructed aggregate, but byte identity is not claimed.

## 5. Before Schema

Before the preceding schema-remediation PR, an unrelated test fixture had
overwritten the operational file with five columns:

`date,symbol,open,close,volume`

That corrupt file is preserved separately and is not a canonical source.

## 6. After Schema

The current historical aggregate remains:

`date,symbol,open,high,low,close,volume,amount`

- Rows: 2,680,710
- Symbols: 799
- Date range: 2010-01-04 through 2026-08-21
- Duplicate date/symbol rows: 0
- Missing or invalid OHLCVA rows: 0
- Future rows beyond the frozen boundary: 0

No historical row was changed during this source-resolution phase.

## 7. Overlap Consistency Result

| Symbol | Days | Median factor | Max ratio deviation | Max return difference | First ratio mismatch | First return mismatch | Result |
|---|---:|---:|---:|---:|---|---|---|
| 000630 | 25 | 0.9210503952 | 0.0039665329 | 0.0019077889 | 2026-07-20 | none | FAIL |
| 000792 | 25 | 0.8741377635 | 0.0037431444 | 0.0013250808 | 2026-07-20 | none | FAIL |
| 601607 | 25 | 5.1859855927 | 0.0047150377 | 0.0046919108 | 2026-07-20 | 2026-08-20 | FAIL |

Limits remain unchanged at `0.0025` for ratio deviation and `0.0025` for
absolute return difference. The full per-date high/low/close values and ratios
are in `hfq_overlap_diagnostic.json`.

## 8. Prediction Retry Result

- HFQ overlap consistency: **FAIL**
- 71-column feature partition: not created
- Seal: not created
- Gen2 invocation: not reached
- Ranking: not created
- Top10/Top20: not created
- Forward evidence: not registered
- Provider requests in this phase: 0
- Execution authorized: FALSE

The downstream prediction command was deliberately not executed because its
mandatory input-consistency precondition remains false.

## Verification

- HFQ, DAILY PIT, prediction, forward-evidence, sandbox, and remediation tests:
  `69 passed`
- Full repository regression:
  `694 passed, 1 xfailed, 24 subtests passed`
- Full regression duration: 167.28 seconds
- Existing warnings: 2 constant-input correlation warnings
- Frozen history and 2026-09-03 market/manifest/receipt hashes before and after
  tests: unchanged
- Immutable diagnostic and machine-readable report seals: verified
- Activation locks: verified intact
- `git diff --check`: passed

## Integrity Boundary

- Frozen Gen2 modified: 0
- 007–012 modified: 0
- DAILY PIT semantics modified: 0
- Sandbox modified: 0
- Validator thresholds changed: FALSE
- Manual price edits or scaling: 0
- Synthetic OHLCVA: FALSE
- Failed symbols removed: FALSE
- Research only: TRUE
- Execution: DISABLED
