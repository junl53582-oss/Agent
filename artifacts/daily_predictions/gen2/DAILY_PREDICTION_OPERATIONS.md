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
