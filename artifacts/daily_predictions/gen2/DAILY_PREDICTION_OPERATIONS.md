# Daily Prediction Operations

Run after 18:30 Asia/Shanghai on a verified trading session:

```powershell
.\.venv\Scripts\python.exe -m stockpilot.daily_prediction predict YYYY-MM-DD --confirm-real-provider-acquisition
```

The provider flag acknowledges only the existing validated market-data path. It
does not authorize a broker or execution. A repeated identical date returns the
existing prediction identity; an identity mismatch returns
`PREDICTION_CONFLICT` and does not overwrite the artifact.

Read operations:

```powershell
.\.venv\Scripts\python.exe -m stockpilot.daily_prediction latest
.\.venv\Scripts\python.exe -m stockpilot.daily_prediction history --limit 20
.\.venv\Scripts\python.exe -m stockpilot.daily_prediction explain SYMBOL YYYY-MM-DD
```

`explain` does not recompute the model. Until the frozen runtime stores native
per-row feature contributions, it reports contribution data as unavailable.

## Dashboard freshness contract

The dashboard's primary prediction section reads only:

```text
artifacts/daily_predictions/gen2/latest.json
→ predictions/YYYY-MM-DD/prediction_manifest.json
→ prediction.json / ranking.csv / top10.csv / top20.csv
```

Every payload is verified against its immutable SHA-256 sidecar and the latest
pointer's recorded hashes. The status is compared with the most recent
completed session in the verified XSHG calendar, using the frozen 18:30
Asia/Shanghai data window:

- `CURRENT`: latest formal prediction matches the latest completed session.
- `STALE`: a valid formal prediction exists but is one or more sessions old.
- `NO_FORMAL_PREDICTION`: no formal DAILY product exists.
- `INVALID`: the calendar, pointer, product identity, or manifest failed
  validation.

The page refreshes this read-only status every 60 seconds. It never starts
provider acquisition, feature materialization, inference, backfill, settlement,
or execution. V6 and V30/V30r1 remain separately labelled historical research
snapshots and can never substitute for a missing formal DAILY prediction.

The prediction job is scheduled independently for verified trading sessions at
18:45 Asia/Shanghai. The dashboard only observes its immutable output.
