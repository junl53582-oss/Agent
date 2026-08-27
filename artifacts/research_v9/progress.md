# V9 execution progress

- Protocol: three isolated lines (`V9-Alpha`, `V9-Nonlinear`, `V9-Portfolio`).
- Production model remains V6 until every replacement gate passes.
- Membership: completed and structurally validated (28 snapshots, 667 symbols, 300 each).
- Market bars: completed (1,651,470 rows, 667/667 symbols, 2015-01-05 to 2026-08-21).
- PIT fundamentals: completed (53,744 rows, 667/667 symbols, zero failures).
- PIT industry history: completed (1,773 changes, 643/667 symbols; 24 remain unknown).
- V9 code: implemented in an isolated package; production V6 files untouched.
- Pure logic tests: 6 passed (PIT weights, filing deltas, residual target,
  label maturity, 75/25 portfolio budget, constrained model blend).
- Feature pipeline smoke test: 40,403 rows, 39 finite features, 25,853 mature targets.
- Final lock: `51CCF2958BA2692BEB044E500730E1C1A4F59FF0C188092B3EFFE16E6E880988`.
- Frozen walk-forward: completed for 2020-2025.
- V9 full: return 21.19%, benchmark 52.37%, excess -31.18%, 3/6 positive
  excess years, Rank IC 0.02241, technology IC -0.01563, drawdown -24.31%,
  one-way turnover 6.84%, average cost 0.01431%.
- Gates passed: turnover and cost only. Excess, annual consistency, V6 IC,
  technology IC, and drawdown gates failed.
- Decision: keep V6 in production; do not start V9 future shadow.
- Audit note: the first run under lock `30CACE...` was invalidated by a
  deterministic symbol/row-index return-alignment bug. Its full evidence is in
  `invalid_run_30CACE`; no model parameter or threshold changed in the fix.
- Final regression: 52 tests passed and the final lock verified.
