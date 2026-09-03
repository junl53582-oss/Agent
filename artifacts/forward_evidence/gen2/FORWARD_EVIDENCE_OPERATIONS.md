# StockPilot Gen2 Forward Evidence Operations

This directory defines an observational, research-only stream for the frozen
`GEN2-LGBM-20D-SECTOR-BALANCED-TOP20` model. It does not authorize trading,
promotion, model changes, feature changes, label changes, threshold changes, or
historical optimization.

## Daily command

After the verified session has closed and the frozen 18:30 Asia/Shanghai window
has opened:

```powershell
.\.venv\Scripts\python.exe -m stockpilot.forward_evidence run `
  --date YYYY-MM-DD `
  --confirm-real-provider-acquisition
```

The confirmation flag acknowledges use of the repository's existing AkShare
provider path. It does not authorize a broker connection. Provider acquisition,
DAILY PIT materialization, input sealing, Gen2 scoring, and evidence registration
remain fail-closed and idempotent.

Before 18:30, on a non-session, or without explicit provider confirmation, the
runner records an immutable `NO_FORWARD_PREDICTION` attempt and makes no market
request. Missing days are never fabricated or backfilled.

## Settlement

Settlement is allowed only after the immutable prediction's actual 20-session
maturity. A market file passed with `--settlement-market` must already have the
validated `.sha256` and `.witness.json` sidecars required by frozen runtime 009.
The monitor does not manufacture a witness or an empty corporate-action source.

```powershell
.\.venv\Scripts\python.exe -m stockpilot.forward_evidence run `
  --date YYYY-MM-DD `
  --settlement-market PATH_TO_VALIDATED_MARKET.csv
```

Forward metrics are recomputed exclusively from immutable mature settlement
records. Historical Gen2 metrics appear only in `protocol.json` as a labelled
reference and are never appended to the prospective sample.

## Recovery and verification

```powershell
.\.venv\Scripts\python.exe -m stockpilot.forward_evidence verify
.\.venv\Scripts\python.exe -m stockpilot.forward_evidence status
```

Each run verifies the frozen 007–011 locks, protected Git surfaces, protocol,
prediction chain, settlement chain, and hashes before continuing. Live state is
updated atomically in `forward_evidence_state.json`; predictions, settlements,
attempts, and checkpoints are append-only. An integrity conflict stops the run
as `FORWARD_EVIDENCE_INVALID`.

## Evidence checkpoints

Immutable reporting checkpoints are written at 5, 10, 20, 40, and 60 matured
sessions. They are observation checkpoints only. No checkpoint changes the
model, Top20 policy, costs, or decision thresholds, and no champion is promoted
automatically.
