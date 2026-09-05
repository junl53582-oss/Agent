# DAILY PREDICTION FRESHNESS AND DASHBOARD OPERATIONALIZATION REPORT

## Root cause

The dashboard's first and most prominent prediction view scanned
`artifacts/research_v6/live/signals/*.csv` and
`artifacts/research_v6/live/predictions/*.csv`. Those paths contain the last V6
research snapshot, not the formal DAILY Gen2 product. The V6 producer is a
manual research workflow and has no formal daily freshness contract. Therefore
the page continued to show 2026-08-28 even when users reasonably expected a
current trading-session prediction.

The formal product already had a separate immutable source:
`artifacts/daily_predictions/gen2/latest.json`. The old dashboard did not read
it.

## Implemented source-of-truth contract

The primary dashboard section now reads only the formal DAILY Gen2 chain:

```text
latest.json
→ predictions/YYYY-MM-DD/prediction_manifest.json
→ prediction.json / ranking.csv / top10.csv / top20.csv
```

The reader verifies the payload SHA-256 sidecars, all manifest bindings, the
latest pointer's prediction and manifest hashes, prediction identity, artifact
directory boundary, formal status, and verified-session date.

The verified XSHG calendar and frozen 18:30 Asia/Shanghai data window determine
the latest completed session. Dashboard status is one of:

- `CURRENT`
- `STALE`
- `NO_FORMAL_PREDICTION`
- `INVALID`

`STALE`, missing, future-dated, off-calendar, pointer mismatch, or integrity
failure is explicit and cannot silently fall back to V6 or V30/V30r1.

## Dashboard behavior

- Formal DAILY Gen2 status is rendered before all historical research content.
- The formal section refreshes every 60 seconds with a Streamlit fragment.
- Refresh is read-only and invokes no provider, feature, model, settlement,
  broker, or execution path.
- A valid formal result displays the complete Top10 and Top20 view plus model,
  feature-manifest, input-seal, and prediction identities.
- V6 and V30/V30r1 are explicitly labelled historical research snapshots.
- The legacy demo-generation action was removed from the operational homepage.
- Missing historical research artifacts no longer prevent the formal freshness
  section from rendering.

## Current observation

Audit time: 2026-09-06 Asia/Shanghai.

- Latest completed verified session: `2026-09-04`
- Formal DAILY prediction: not present
- Dashboard status: `NO_FORMAL_PREDICTION`
- Next verified prediction date: `2026-09-07`
- Earliest legal data time: `2026-09-07T18:30:00+08:00`
- Scheduled run time: `2026-09-07T18:45:00+08:00`
- 2026-09-01 / 2026-09-03 / 2026-09-04 backfill: forbidden
- Retrospective Top10/Top20 generated: false

## Automation

The existing `StockPilot 每日股票预测` heartbeat is active on weekdays at
18:45 Asia/Shanghai. It is independent of Streamlit, targets the unique
operational main checkout, requires `HEAD == origin/main` and a clean tracked
tree, uses only the current verified session, and fails closed otherwise.

The dashboard observes immutable output from that job. It does not run the job
itself.

## Safety and protected surfaces

- Gen2 model modified: 0
- 007–012 semantics modified: 0
- DAILY PIT core semantics modified: 0
- Provider routing modified: 0
- Sandbox modified: 0
- Alpha/model training performed: false
- Return labels read: false
- Broker requests: 0
- Execution authorized: false

## Verification

- Freshness/product unit tests: 25 passed
- Combined freshness, DAILY product, forward evidence, and DAILY PIT tests:
  43 passed
- Streamlit no-server render: 0 exceptions
- Full repository: 732 passed, 1 xfailed, 24 subtests passed; 3 pre-existing
  local-data/lock tests fail in an isolated worktree because ignored historical
  data files are absent. No failure references the changed dashboard or
  freshness module.
- Scoped Ruff: passed
- `git diff --check`: passed

## Status

`DAILY_PREDICTION_FRESHNESS_AND_DASHBOARD_READY`

This status means the page can accurately represent freshness and failure. It
does not claim that a formal prediction exists for 2026-09-04 and does not
authorize a retrospective prediction.
