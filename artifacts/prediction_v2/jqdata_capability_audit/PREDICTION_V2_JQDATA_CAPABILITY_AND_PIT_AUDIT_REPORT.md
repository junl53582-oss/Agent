# PREDICTION_V2_JQDATA_CAPABILITY_AND_PIT_AUDIT_REPORT

Status: `PREDICTION_V2_JQDATA_FOUNDATION_BLOCKED`

## Connection and entitlement

- Authentication: `PASS`
- SDK/server: `1.9.8` / `2.0.0`
- Licensed date range: `2025-05-28 00:00:00` to `2026-06-04 00:00:00`
- Account expiry: `2026-12-06 00:00:00`
- Query quota after audit: `{'total': 1000000, 'spare': 999769}`
- Credential values or account identifier persisted: `FALSE`

## Market and fundamentals

- HFQ OHLCVA schema probe: `PASS`
- JQData `money` maps to StockPilot `amount`; no synthetic fields were created.
- Fundamentals `date`/as-of probe: `PASS`
- `statDate` query mode used: `FALSE`.
- Production Tencent-first routing changed: `FALSE`.

## Event and expectation data

- Finance tables enumerated: `77`
- Relevant tables: `FUND_REPORT_DATE, STK_FIN_FORCAST, STK_PERFORMANCE_LETTERS, STK_REPORT_DISCLOSURE`
- Historical analyst consensus/vintage table: `NOT AVAILABLE`.
- `STK_FIN_FORCAST` is issuer earnings guidance, not sell-side analyst consensus.
- Issuer forecast observed range: `2025-06-20` to `2026-08-25`.
- The issuer table has publication date and forecast-period/value fields, but no verified intraday timestamp or explicit revision/supersession link.

## Admission decision

- Entitled history: `1.019` years; required: `5` years.
- Historical coverage: `FAIL`.
- Analyst vintages: `NOT_AVAILABLE`.
- Approved use now: `BOUNDED_RECENT_SCHEMA_VALIDATION_ONLY`.
- Model training/challenger: `NOT STARTED`.
- Gen2, 007–012, DAILY PIT, and production provider code: unchanged.

## Final status

`PREDICTION_V2_JQDATA_FOUNDATION_BLOCKED`
