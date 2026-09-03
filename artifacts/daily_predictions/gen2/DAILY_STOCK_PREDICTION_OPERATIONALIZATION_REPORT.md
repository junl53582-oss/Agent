# DAILY_STOCK_PREDICTION_OPERATIONALIZATION_REPORT

## 1. Status

`DAILY_STOCK_PREDICTION_READY`

The product entry point is complete and fail-closed. The first real invocation
on 2026-09-03 occurred before the frozen 18:30 Asia/Shanghai market-data window,
so the current daily status is `NO_PREDICTION / DATA_WINDOW_NOT_OPEN`. Product
readiness does not override daily eligibility.

## 2. Baseline

- Baseline Git SHA: `0119ca98e4db9156ec1008b8155fa4342131943d`
- Model: `GEN2-LGBM-20D-SECTOR-BALANCED-TOP20`
- Horizon: 20 trading sessions
- Semantics: cross-sectional relative-strength ranking
- Feature contract: frozen Gen2/V31 train-only feature policy
- Locks: effective 007–011 chain, verified before prediction side effects
- Evidence level: `WEAK_REGIME_DEPENDENT`

## 3. Prediction CLI

- `predict DATE`: one command for acquisition, PIT validation, feature
  materialization, seal, Gen2 scoring, ranking, and publishing.
- `latest`: reads the latest verified immutable product artifact.
- `history`: reads prior product artifacts without recomputation.
- `explain SYMBOL DATE`: returns rank/score context. Per-stock feature
  contributions are explicitly unavailable because the frozen runtime did not
  persist LightGBM `pred_contrib` output; no driver narrative is invented.

## 4. Daily Lifecycle

```text
existing validated real provider + explicit acknowledgement
→ DAILY PIT validation
→ frozen 71-column feature materialization
→ immutable input seal
→ frozen Gen2 deterministic prediction
→ complete-universe ranking
→ Top10 / Top20 display views
→ immutable user-facing report and manifest
```

The Top20 display is the first 20 model ranks. The existing sector-balanced,
20-session portfolio decision is retained separately in
`selected_for_frozen_portfolio`; this product does not change portfolio policy
and does not authorize execution.

## 5. Example Prediction

No valid real prediction can be shown yet.

- Date checked: 2026-09-03
- Status: `NO_PREDICTION`
- Reason: `DATA_WINDOW_NOT_OPEN`
- Next eligible time: `2026-09-03T18:30:00+08:00`
- Universe count: unavailable because acquisition was correctly not attempted
- Top10: unavailable
- Top20: unavailable
- Provider requests: 0
- Broker requests: 0

No fixture or sandbox result is presented as a formal daily prediction.

## 6. Output Artifacts

For each successful date under
`artifacts/daily_predictions/gen2/predictions/YYYY-MM-DD/`:

- `prediction.json`
- `ranking.csv`
- `top10.csv`
- `top20.csv`
- `DAILY_STOCK_PREDICTION_REPORT_YYYY-MM-DD.md`
- `prediction_manifest.json`

Payload and sidecars are immutable. `latest.json` is an atomic pointer to the
most recent verified prediction. Failed daily attempts have their own immutable
status JSON, report, and manifest and never masquerade as predictions.

## 7. Safety

- Broker calls: 0
- Real orders: false
- Real trades: false
- Execution authorization: false
- Research only: true
- Price targets: not produced
- Rise probabilities: not produced
- Automatic model promotion: false

## 8. Tests

The product test suite covers all required behaviors:

1. valid date produces a complete ranking;
2. Top10 and Top20 generation;
3. deterministic output;
4. immutable artifacts;
5. identical-run idempotency;
6. conflicting-run rejection;
7. pre-window rejection without a provider call;
8. invalid PIT, feature, seal, and lock rejection;
9. zero execution authority and broker activity;
10. Markdown/JSON/CSV ranking correspondence;
11. verified latest reads;
12. history reads without recomputation;
13. explain does not invent feature contributions;
14. auditable no-prediction reports.

Product-only result: `19 passed`.

Combined DAILY PIT / Gen2 runtime / forward-evidence result: `100 passed`.

Full repository regression: `693 passed, 1 xfailed, 24 subtests passed`.

Scoped Ruff and `git diff --check`: passed. The effective lock chain and frozen
Gen2 boundary are also exercised by the combined regression and the prediction
preflight; no lock or frozen-model file is modified by this task.

## 9. Automation

The existing thread heartbeat is updated to `StockPilot 每日股票预测`, scheduled
for weekdays at 18:45 Asia/Shanghai. It runs the product command with the
repository's required explicit real-provider acknowledgement, then synchronizes
the independent forward-evidence monitor. Non-sessions and invalid inputs fail
closed. It never submits an order.

## 10. Git / PR

- Branch: `codex/daily-stock-prediction`
- Parent: verified PR #8 head `52a14664b0dc8ded1c71fcad71fbefc1e6f7c481`
- Change class: additive presentation/product layer only
- Frozen Gen2 modified: 0
- 007–012 modified: 0
- DAILY PIT core modified: 0
- Sandbox core modified: 0
- PR/CI: recorded after final push

## 11. Final User Instructions

Today's prediction after the legal window:

```powershell
.\.venv\Scripts\python.exe -m stockpilot.daily_prediction predict 2026-09-03 --confirm-real-provider-acquisition
```

Latest prediction:

```powershell
.\.venv\Scripts\python.exe -m stockpilot.daily_prediction latest
```

Prediction history:

```powershell
.\.venv\Scripts\python.exe -m stockpilot.daily_prediction history
```

Explain a stock without fabricated feature drivers:

```powershell
.\.venv\Scripts\python.exe -m stockpilot.daily_prediction explain 000001 2026-09-03
```

## 12. Final Decision

`DAILY_STOCK_PREDICTION_READY`

The StockPilot system can now produce an explicit daily research-only stock
ranking using the verified Gen2 DAILY PIT prediction pipeline.

This ranking is not a guarantee of future returns and does not authorize
trading.
