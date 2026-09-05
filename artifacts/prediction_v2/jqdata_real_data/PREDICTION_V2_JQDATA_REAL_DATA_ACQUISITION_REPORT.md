# PREDICTION_V2_JQDATA_REAL_DATA_ACQUISITION_REPORT

Status: `PREDICTION_V2_JQDATA_RESEARCH_DATA_READY`

JQData adapter usable: `YES`
Real provider queries: `67`
Real rows acquired: `263716`
Raw partitions: `13`
Normalized rows: `266216`

## Actual per-dataset coverage

- UNIVERSE: rows=300, dates=2026-06-04..2026-06-04, symbols=300, status=`RESEARCH_SCOPE_ONLY`, sha256=`29324fc57a68b0694a12894be0d78984792227f7d3efaa10298b000fe28a31e0`, schema=`0ef607d1105138fc656df992f736109ec0fe5fcbdda3eb56e4b1294e2ee83b4d`, query={"date": "2026-06-04", "index": "000300.XSHG"}
- TRADING_CALENDAR: rows=313, dates=2025-05-28..2026-09-04, symbols=0, status=`PIT_REFERENCE_CALENDAR`, sha256=`f8e726860b50d9ca037c09a3558d5199b75d91de916342b31db67364bbe4a22c`, schema=`c0b51e9635d9382b73323c15e9da25f3c543621165c91ed71097838b492abea6`, query={"end": "2026-09-05", "start": "2025-05-28"}
- GET_HISTORY_INDUSTRY: rows=93, dates=1991-01-29..2026-07-01, symbols=50, status=`PIT_SAFE_HISTORICAL_MEMBERSHIP`, sha256=`f00174fc9cc47880e510943d833111615ce3c248d04e7553325899e1b2a32299`, schema=`e087553d924b6912c32d74a530cc2f9881d73cbfc83a5470dc26ab0ce04ce1ee`, query={"classification": "sw_l1", "symbols": "core50_sha256"}
- INDUSTRY_CATALOG: rows=31, dates=2004-02-09..2021-12-11, symbols=0, status=`PIT_REFERENCE_INDUSTRY_CATALOG`, sha256=`b28a40318ce1136741ea6263b83839dd560991cf0eef4e5b66fd89e723608b5f`, schema=`cf97f21a27ad8fb7c2926fc20a5ce027740b7e547dbcde568595fe5b872aec36`, query={"classification": "sw_l1", "date": "2026-06-04"}
- VALUATION: rows=2500, dates=2025-05-28..2026-06-02, symbols=50, status=`PIT_SAFE_PROVIDER_ASOF`, sha256=`f83c855adc8aa6c580693fb4c9ae4ce10b422180c12637aaf711db3fde763d5b`, schema=`fafa6a5a966e67839cdba1fa3c6adbeb6b00d977333c7e7e7a0e44613456e945`, query={"dates": ["2025-05-28", "2025-06-05", "2025-06-12", "2025-06-19", "2025-06-26", "2025-07-03", "2025-07-10", "2025-07-17", "2025-07-24", "2025-07-31", "2025-08-07", "2025-08-14", "2025-08-21", "2025-08-28", "2025-09-04", "2025-09-11", "2025-09-18", "2025-09-25", "2025-10-10", "2025-10-17", "2025-10-24", "2025-10-31", "2025-11-07", "2025-11-14", "2025-11-21", "2025-11-28", "2025-12-05", "2025-12-12", "2025-12-19", "2025-12-26", "2026-01-06", "2026-01-13", "2026-01-20", "2026-01-27", "2026-02-03", "2026-02-10", "2026-02-25", "2026-03-04", "2026-03-11", "2026-03-18", "2026-03-25", "2026-04-01", "2026-04-09", "2026-04-16", "2026-04-23", "2026-04-30", "2026-05-12", "2026-05-19", "2026-05-26", "2026-06-02"], "symbols": "core50_sha256"}
- STK_REPORT_DISCLOSURE: rows=1500, dates=2025-07-26..2026-08-31, symbols=300, status=`PIT_SAFE_WITH_LAG_FOR_PUB_DATE_ONLY`, sha256=`ca9b93d4eb7f7483e6fed6c10c85cf09d17f4bee531c3e0e9cc529598d5680c8`, schema=`73595fab8ea1a30f985469de3e4d53a813098aa57911905d69ec33997e63787d`, query={"date_restriction": "provider_entitlement", "universe": "csi300_reference"}
- STK_FIN_FORCAST: rows=298, dates=2025-06-25..2026-08-25, symbols=152, status=`PIT_SAFE_WITH_LAG`, sha256=`7ab71fda3256e2187a7579585daa62679e930d70db32d5cee383cb6626004181`, schema=`875621b4bc6cad5ca1ef9d0cd0b399fa4d3bf9139a4df08c47e10ca077d7045e`, query={"date_restriction": "provider_entitlement", "universe": "csi300_reference"}
- STK_PERFORMANCE_LETTERS: rows=176, dates=2025-07-12..2026-08-24, symbols=68, status=`PIT_SAFE_WITH_LAG_RESTATEMENT_RISK`, sha256=`5288ffcd221ce5e60a0b41e62ac488f39336a636417e1b0f3f7df8ddd001db6e`, schema=`6399c32241adbe8d5a0e35824256fcfbba3660acef3aaf5cfc4c147f186e9c7f`, query={"date_restriction": "provider_entitlement", "universe": "csi300_reference"}
- FINANCE_BALANCE_SHEET: rows=3005, dates=2025-06-27..2026-08-31, symbols=300, status=`PIT_RESTATEMENT_RISK`, sha256=`4e61b0002cd0cae2f8f65b0830dde46bf41c0c2c10e04d27b8ef02bcf705fe75`, schema=`24914aac3466a7b8000f654f0e0b9b549409a7a3648ce02958a9f1ed1eba202a`, query={"date_restriction": "provider_entitlement", "universe": "csi300_reference"}
- FINANCE_INCOME_STATEMENT: rows=3005, dates=2025-06-27..2026-08-31, symbols=300, status=`PIT_RESTATEMENT_RISK`, sha256=`931d116d6c8daa12bcb331f22d2fe1387d3bedae09e8775379ec89dbd29abd14`, schema=`9415cfe85d72fdee78b95c850e52ad4283c694f71998d7e06c020512fe203c44`, query={"date_restriction": "provider_entitlement", "universe": "csi300_reference"}
- FINANCE_CASHFLOW_STATEMENT: rows=3005, dates=2025-06-27..2026-08-31, symbols=300, status=`PIT_RESTATEMENT_RISK`, sha256=`34959c88aeb88df3621fc7501f9e8e552dd0c3a69bb81137ba873129ba493e23`, schema=`56edf628830e852b6786f5f0b03262d6544444fbbdc5e2519f16ddf03d9a6a05`, query={"date_restriction": "provider_entitlement", "universe": "csi300_reference"}
- STK_HK_HOLD_INFO: rows=1490, dates=2025-06-30..2026-06-30, symbols=298, status=`PIT_SAFE_WITH_LAG`, sha256=`b4e7998e334609620cb769e059e3701dea72349093d34beb3c8527f482171747`, schema=`7844891b741012251016a7654bf0cac8435d21c5c01eda064e636b249e228fec`, query={"date_restriction": "provider_entitlement", "universe": "csi300_reference"}
- FACTOR_LIBRARY: rows=248000, dates=2025-05-28..2026-06-04, symbols=50, status=`PIT_SAFE_PROVIDER_ASOF_WITH_LAG`, sha256=`2e3ac3093b959208470a58b337b89ec777bf02c5973e1f5fe104b4990be0704c`, schema=`76f8ee73ad0eb9ed94b44f84d2128e5500749e778f1b36f623e744b8248b7a32`, query={"end": "2026-06-04", "factors": ["roe_ttm", "roic_ttm", "net_operate_cash_flow_to_asset", "debt_to_asset_ratio", "net_profit_growth_rate", "operating_revenue_growth_rate", "net_operate_cashflow_growth_rate", "total_asset_growth_rate", "Variance20", "Variance60", "Skewness20", "Kurtosis20", "Price1M", "Price3M", "ROC20", "Rank1M", "money_flow_20", "turnover_volatility", "VROC12", "DAVOL20"], "start": "2025-05-28", "symbols": "core50_sha256", "transport_batches": ["quality", "growth", "risk", "momentum", "emotion"]}

## Classification

- PIT-safe: `GET_HISTORY_INDUSTRY, VALUATION, STK_REPORT_DISCLOSURE, STK_FIN_FORCAST, STK_HK_HOLD_INFO, FACTOR_LIBRARY`
- PIT-safe-with-lag: `STK_REPORT_DISCLOSURE, STK_FIN_FORCAST, STK_HK_HOLD_INFO, FACTOR_LIBRARY`
- Research-only/risk: `UNIVERSE, STK_PERFORMANCE_LETTERS, FINANCE_BALANCE_SHEET, FINANCE_INCOME_STATEMENT, FINANCE_CASHFLOW_STATEMENT`
- Rejected: `MONEYFLOW_HISTORY_DAILY(REJECTED_ACCESS_DENIED)`
- `STK_FIN_FORCAST` is normalized as company earnings forecast, never analyst estimates.
- Performance letters are earnings-event data; actual_versions contract remains failed.
- Three statement tables retain `PIT_RESTATEMENT_RISK` and are excluded from PIT-safe features.

## Feature store

- Status: `READY`
- Long rows / wide rows: `355204` / `14044`
- Features / symbols: `71` / `300`
- Date range: `2025-05-28` to `2026-08-25`
- Every long-form feature row carries source dataset, raw SHA256, available_at, and feature_asof_date.

## Remaining gates

- Analyst vintages: `NOT_AVAILABLE`
- Full 5-year historical gate: `FAIL`
- Complete Prediction V2 acquisition: `BLOCKED`

## Integrity

- Model training: `FALSE`
- Return labels read: `FALSE`
- RankIC: `NOT_COMPUTED`
- Credentials/account identity persisted: `FALSE`
- Gen2 modified: `0`
- 007–012 modified: `0`
- DAILY PIT modified: `0`
- daily_prediction modified: `0`
- Tencent-first routing modified: `0`

## Final status

`PREDICTION_V2_JQDATA_RESEARCH_DATA_READY`

`PREDICTION_V2_DATA_ACQUISITION_BLOCKED` remains simultaneously true.
