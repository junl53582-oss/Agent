# PROVIDER_ROUTING_POLICY_REPORT

Status: `PROVIDER_ROUTING_POLICY_FIXED`

## Old routing

`Eastmoney HFQ -> Tencent HFQ fallback`

Eastmoney normally succeeded, so the Tencent fallback was never reached even
though the frozen V10 historical lineage is Tencent. This allowed a
cross-provider candidate to reach the downstream overlap gate.

## New routing

`Tencent HFQ -> Eastmoney HFQ fallback`

The acquisition policy now resolves the frozen lineage from immutable
provenance before provider selection. For the current canonical history:

- Historical lineage: `akshare-tencent`
- Primary provider: `akshare-tencent`
- Fallback provider: `akshare-eastmoney`
- Fallback trigger: Tencent is unavailable for that symbol
- Fallback warning: `EASTMONEY_FALLBACK_USED`
- Fallback block: `HFQ_LINEAGE_FALLBACK_BLOCKED`

Every routed candidate is passed through the unchanged
`stockpilot.prediction_forward.stitch_hfq_market` overlap validator before any
immutable market partition is published. An inconsistent Eastmoney fallback
therefore leaves no partial partition behind.

The default acquisition runner used by `daily_prediction predict` now points to
the lineage-aligned router. The frozen 011 DAILY PIT validator, runtime, feature
schema, thresholds, and model remain byte-for-byte unchanged.

## Tests

- Historical Tencent + daily Tencent: `PASS`
- Tencent success skips Eastmoney entirely: `PASS`
- Historical Tencent + inconsistent daily Eastmoney: blocked before immutable
  publication with `HFQ_LINEAGE_FALLBACK_BLOCKED`
- Default daily prediction router: `PASS`
- Focused suite: `52 passed`
- Full regression: `701 passed, 1 xfailed, 24 subtests passed`
- Ruff: `PASS`

## Prediction readiness

The next same-day DAILY prediction is ready to use Tencent-first acquisition
after 18:30 Asia/Shanghai, subject to every existing PIT, coverage, overlap,
seal, model, and product gate.

The historical 2026-09-03 retry remains blocked by the unchanged
`TARGET_DATE_MUST_BE_TODAY` rule. This routing fix does not authorize backfill,
execution, orders, or trades.

- Research only: `TRUE`
- Execution: `DISABLED`
