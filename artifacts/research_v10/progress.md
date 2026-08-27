# V10 execution progress

- Production model remains V6.
- V9 remains rejected and unchanged.
- Stage 1 core replication: passed after normalizing mixed provider volume fields.
  Net tracking error 0.158%, minimum weight coverage 99.80%.
- Invalid input-schema audit is preserved under `invalid_input_schema_run_151951`.
- Historical membership: 42 snapshots, 800 symbols, 2010-01-04 onward, 300 each.
- Market: 2,680,710 positive-price HFQ rows, 799/800 symbols, volume and amount 100%.
- PIT fundamentals: 66,607 rows, 796/800 symbols, zero announcement violations.
- PIT industry: 2,096 changes, 758/800 symbols, zero effective-date violations.
- Model: 61 features; 5d/20d 30/70; Ridge/LightGBM 60/40; V6/new alpha 40/60.
- Portfolio: benchmark-relative, sector neutral, stock active cap 0.75%, active budget 15%,
  ex-ante tracking error cap 6%, technology specialist gated by two validation years.
- Final lock: `422E767B00A9E3936B909C95DCC424D797CFF9842437B0C0F2F87922ED72D3E1`.
- Frozen 2020-2025 validation completed.
- V10 full: return 29.37%, benchmark 31.59%, excess -2.22%, 2/6 positive
  excess years, 5d IC 0.03564, 20d IC 0.04136, Top30 selected excess -0.109%,
  technology IC -0.00694/-0.00623, drawdown -33.06%, realized tracking error 0.85%.
- Passed: core replication, both global IC gates, tracking error, turnover, cost,
  stock cap, and sector neutrality.
- Failed: cumulative excess, 4/6 years, Top30 excess, both technology IC gates,
  and absolute drawdown.
- Decision: keep V6; do not start V10 future shadow; dashboard remains V6.
- Final regression: 59 tests passed and the final lock verified.
