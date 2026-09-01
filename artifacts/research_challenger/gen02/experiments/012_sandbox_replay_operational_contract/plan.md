# 012 Side-Effect-Free Sandbox Replay Operational Contract

This additive contract introduces an explicit replay-only DAILY PIT lifecycle.
It does not modify the frozen 007-011 surfaces or production runtime defaults.

The sandbox must bind every mutable root below one unique run directory, accept
only hash-verified replay bundles, reuse the frozen validation/model/ranking and
settlement semantics, and reject any provider, broker, production reservation,
prediction, or settlement path.

The unified run manifest is the audit root for market, feature, seal, prediction,
candidate, gate, decision, execution simulation, settlement simulation, and all
zero-side-effect assertions.
