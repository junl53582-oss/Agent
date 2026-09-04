# DAILY_PIT_PROVIDER_LINEAGE_ALIGNMENT_REPORT

Final status: `PROVIDER_LINEAGE_ALIGNMENT_BLOCKED`

The data-provider lineage itself is aligned and fully validated, but the formal
2026-09-03 prediction is correctly blocked by the unchanged same-day product
gate because the retry occurred after the prediction date. No seal, Gen2
ranking, Top10, or Top20 was created.

## Provider routing and cause

- Current provider in the original immutable partition: `akshare-eastmoney`
- Historical provider: `akshare-tencent`
- Mismatch cause: `CROSS_PROVIDER_HFQ_LINEAGE_CHANGE_TENCENT_TO_EASTMONEY`
- New candidate provider: `akshare-tencent`
- Locked routing: Eastmoney primary, Tencent fallback
- Why Eastmoney was used: the primary request succeeded, so fallback routing
  never called Tencent.
- Locked router modified: `FALSE`

## Isolated Tencent HFQ candidate

- Location: `data/prospective_gen2/provider_lineage_candidates/tencent_hfq/daily_inputs/2026-09-03`
- Provider / adjustment: `akshare-tencent` / `hfq`
- Acquisition timestamp: `2026-09-03T16:16:08.587180+00:00`
- Coverage: 10,200 rows, 300 symbols, 2026-07-20 through 2026-09-03
- Required columns: `date,symbol,open,high,low,close,volume,amount`
- Provider requests / failures: 300 / 0
- Mixed provider: `FALSE`
- Market SHA256: `a96d248237609ce629950e00c89bef0a6280140bed3727fc7c9cf052bb452198`
- Source receipt SHA256: `fe129d753e3e6eaff9590984feceb1a942b38d99d0c188e44f7fcfe79d3591eb`
- Market manifest SHA256: `d57de682d1966a648f7f3b67c1cdd2c14e108417277adf1eb680f8f5a3b942aa`

The original Eastmoney partition was not overwritten. Its market, receipt, and
manifest hashes remained respectively `9b8d1714…`, `135c8ef7…`, and
`bbea044b…` before and after candidate acquisition.

## Overlap result

Status: `PASS`

- Current / covered / anchored symbols: 300 / 300 / 300
- Isolated non-current symbols: 0
- Overlap / extension rows: 7,500 / 2,700
- Unchanged ratio and return limits: 0.0025 / 0.0025
- Maximum observed ratio deviation: 0.0016669169561821517
- Maximum observed return difference: 0.0016991629310467715

| Symbol | Overlap rows | Factor | Ratio deviation | Return difference | Result |
|---|---:|---:|---:|---:|---|
| 000630 | 25 | 1.0 | 0.0 | 0.0 | PASS |
| 000792 | 25 | 1.0 | 0.0 | 0.0 | PASS |
| 601607 | 25 | 1.0 | 0.0 | 0.0 | PASS |

## PIT feature materialization

Status: `PASS`

- Rows / eligible symbols: 238 / 238
- Columns / model features: 71 / 61
- Duplicate keys / null cells: 0 / 0
- Panel SHA256: `3c7cc4c38719b4ba1e638498eba38165aac682b803d9506f7de2ef01ed52ca0d`
- Manifest SHA256: `53fade1ae88ac6a55659605ea5e84a510b878ab7a80477d4af5fee7dc04634ad`
- Feature semantics modified: `FALSE`

The candidate-only adapter carries the `broad_sector` value already emitted by
the locked feature builder into the locked 71-column schema. It does not infer
or transform a model feature, and the original DAILY PIT module remains
unchanged.

## Prediction retry

- Retry performed: `TRUE`
- Result: `NO_PREDICTION`
- Reason: `TARGET_DATE_MUST_BE_TODAY`
- Attempt: `2026-09-03T162802.940418_0000`
- Attempt manifest SHA256: `e774c1acea96c98b14c33a8ad74d3b7306011bfbd2a52451496ec790363332a9`
- Provider requests during retry: 0
- Seal created: `FALSE`
- Gen2 ranking created: `FALSE`
- Top10 / Top20 created: `FALSE` / `FALSE`
- Forward evidence: not registered because no formal prediction exists
- Research only: `TRUE`
- Execution: `DISABLED`

## Integrity

- Validator modified: 0
- Overlap threshold modified: 0
- Stock prices modified: 0
- Failed symbols deleted: 0
- Gen2 modified: 0
- Feature semantics modified: 0
- Model retrained: 0
- Broker requests / orders / trades: 0 / 0 / 0

The only permitted final state is therefore:

`PROVIDER_LINEAGE_ALIGNMENT_BLOCKED`
