# FROZEN_V10_MARKET_HISTORY_SCHEMA_REMEDIATION_REPORT

## Root Cause

- Frozen historical file: `data/market_history_v10_hfq.csv`
- Header before: `date,symbol,open,close,volume`
- Missing: `high`, `low`, `amount`
- Before SHA256: `fbda08bd1e013769153bde0258af0b3814e9e35350f36ea4dba09b471b622976`
- Before shape: 5,400 rows, 60 symbols, 2026-05-01 through 2026-09-03
- Git status/history: ignored, untracked, and no committed history
- Cause: the forward-evidence test fixture resolved its frozen market path to the
  default operational path and overwrote it with a five-column fixture during a
  full-repository test run.

The corrupt hash has zero repository references. The 34 frozen references bind
`fcc1ea...`, whose locked quality record explicitly describes the original
eight-column, 2,680,710-row HFQ artifact; they do not bind the corrupt file.

## Current Incremental Market

- Header: `date,symbol,open,high,low,close,volume,amount`
- OHLCVA complete: **TRUE**
- Rows/symbols: 10,200 / 300; target-date rows: 300
- Date range: 2026-07-20 through 2026-09-03
- Source: existing immutable AkShare Eastmoney/Tencent evidence
- Market SHA256: `9b8d17147ff6273a4440b5b33cc047bf5c5e88266dd7926ecc87f087aa002353`
- Manifest SHA256: `bbea044b18a41a74156c44f698860a3bcece9378f8642198932eabfc0cc9a319`
- Source receipt SHA256: `135c8ef786eeb035f5019561f535f7c92b949421fe565c3e65f94834763e8c4a`

`CURRENT_INCREMENTAL_MARKET_SCHEMA_VALID = TRUE`. It was not downloaded or
overwritten during remediation; retry provider requests were zero.

## Recovery Source

- Canonical source: 799 existing
  `data/raw_v10_hfq/*_2010-01-01_2026-08-21_hfq.csv` provider-cache files
- Source type: real HFQ provider cache retained from the original V10 build
- Normalization: repository canonical `research_v10.history_data.fetch_hfq_history`
  and `stockpilot.data.validate_panel`
- Adjustment: HFQ
- Date range: 2010-01-04 through 2026-08-21
- Symbols: 799
- Original source mix: cache 784, Eastmoney 11, Tencent 4; one failed symbol
- Per-file inventory artifact SHA256:
  `c3e5222a5eed0f51feefcb81cbbab29814d9ec391ea0be2c02b3f492b70f9232`

The recovered values join all 947,079 Gen2 training-cache rows. Open, high, low,
close, and volume have zero differences. Three historical amount values differ
by only `3.725290298461914e-09` after CSV float re-serialization; this is
recorded rather than hidden. No field was inferred or synthesized.

## Frozen History After Recovery

- Header: `date,symbol,open,high,low,close,volume,amount`
- Rows: 2,680,710
- Symbols: 799
- Date min/max: 2010-01-04 / 2026-08-21
- SHA256: `efaa21915d24665a63215816d4d0c3f1713202e48f4c0581672e225fd86ded47`
- Original locked byte identity: `fcc1ea387719476410f8d4dcc49840e74bd087aec351ed02cb40d1318dd45fc3`
- Duplicates/nulls/invalid OHLC/negative volume or amount: 0
- Deterministic date/symbol ordering: TRUE
- Rows after allowed history boundary: 0

The erroneous five-column file remains recoverable at
`data/market_schema_remediation_20260903/invalid_5column_fbda08bd1e013769.csv`.
Historical locks were not rewritten. The new identity and byte-level distinction
from the unavailable original aggregate CSV are explicit in the JSON receipt.

## Integrity

- Synthetic columns created: **FALSE**
- Future data used: **FALSE**
- Gen2 modified: 0
- 007–012 semantics modified: 0
- DAILY PIT core semantics modified: 0
- Sandbox modified: 0
- Schema validation lowered: FALSE
- Broker requests/orders/trades: 0 / 0 / 0
- Execution authorized: FALSE

The test fixture was isolated to its temporary directory, and an independent
regression verifies that a five-column frozen history fails closed without
mutating either input or manufacturing missing columns. Existing valid-OHLCVA
materialization coverage remains in the locked DAILY PIT suite.

## Feature Materialization

- Status: **BLOCKED**
- Rows/columns/schema: not created
- PIT: fail closed
- Reason: `000630`, `000792`, and `601607` lack a valid HFQ overlap anchor or
  exceed the frozen overlap-consistency limits.

This is a second, independent data-consistency failure discovered only after the
missing schema was restored. No threshold, data row, or symbol was altered to
force acceptance.

## Prediction Retry

- Status: **NO_PREDICTION / FEATURE_INVALID**
- Provider requests on retry: 0
- Seal/model/ranking: not reached
- Top10/Top20: not produced
- `latest`: no formal daily prediction
- `history`: empty

## Forward Evidence

- Prospective semantics valid: **FALSE**
- Registered: FALSE
- Maturity: `NOT_REGISTERED`
- State: `FORWARD_EVIDENCE_BLOCKED`

## Verification

- Frozen environment: Python 3.11.9, pandas 2.2.3, numpy 2.1.3,
  scikit-learn 1.6.1, jieba 0.42.1
- Focused remediation and isolation regression: `32 passed`
- Full repository regression: `694 passed, 1 xfailed, 24 subtests passed`
- Full repository regression duration: 239.76 seconds
- Ruff: passed
- `git diff --check`: passed
- Frozen history and 2026-09-03 immutable input hashes before/after tests:
  unchanged

## Git

- Branch: `codex/frozen-v10-market-schema-remediation`
- Intended stacked base: `codex/daily-stock-prediction` (PR #9 head)
- This remediation branch contains no prediction ranking and does not merge or
  bypass the still-open parent prediction PR.

## Final Status

`DAILY_PREDICTION_STILL_BLOCKED`

The frozen V10 market-history schema is restored from existing real HFQ provider
cache without synthetic columns, but the original 2026-09-03 prediction remains
blocked by a separate strict HFQ overlap-consistency failure. No stock ranking is
presented as formal output.
