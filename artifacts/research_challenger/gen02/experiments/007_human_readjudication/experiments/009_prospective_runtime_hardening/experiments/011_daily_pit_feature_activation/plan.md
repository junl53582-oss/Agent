# Gen2 Daily PIT Feature Activation 011

Status: operational input activation only. Gen2 remains research-only and V6 remains champion.

This append-only amendment binds the daily input chain:

`verified session -> market acquisition -> immutable raw evidence -> PIT joins -> frozen V10 feature builder -> exact 71-column daily parquet -> immutable manifest -> Gen2 seal-inputs`

The amendment does not alter Gen2 model selection, feature definitions, training policy, portfolio construction, cost assumptions, seed, 20-session horizon, Top-20 selection, V6, V1r4, runtime 009, or activation 010r3.

Operational invariants:

- Market acquisition ends at the target trading date and may not use future bars.
- A missing target-date bar is never replaced with a previous-day bar.
- PIT membership, fundamentals, and industry joins are backward-as-of only.
- Every daily partition is append-only and conflicts fail closed.
- Acquisition and feature materialization never create a prediction or reservation.
- The blocked 2026-09-01 prediction is permanent and cannot be backfilled.
- Real provider access is a separately acknowledged operational phase.
- Formal seal, preflight, and predict must enter through the 011 wrapper.
- Research-only, no promotion, no execution authorization.
