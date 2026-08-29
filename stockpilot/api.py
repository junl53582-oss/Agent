from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from fastapi import FastAPI, HTTPException
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("请先安装应用依赖: pip install -e .[app]") from exc

from .config import Settings
from .future_test import future_test_status
from .pipeline import run_demo, run_file
from .prediction.config import PredictionSettings
from .prediction.inference import prediction_history, prediction_status
from .prediction_v30r1.config import V30R1Settings
from .prediction_v30r1.inference import v30r1_status

app = FastAPI(title="StockPilot CN API", version="0.1.0")


def _artifact(name: str) -> Path:
    return Settings.from_env().artifact_dir / name


def _read_csv(name: str) -> list[dict]:
    path = _artifact(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="尚无回测产物，请先运行 stockpilot demo")
    return pd.read_csv(path).replace({float("nan"): None}).to_dict(orient="records")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "artifacts_ready": _artifact("summary.json").exists()}


@app.get("/summary")
def summary() -> dict:
    path = _artifact("summary.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="尚无回测产物")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/signals/latest")
def latest_signals() -> list[dict]:
    return _read_csv("latest_signals.csv")


@app.get("/equity")
def equity() -> list[dict]:
    return _read_csv("equity.csv")


@app.get("/validation-v2")
def validation_v2() -> dict:
    path = _artifact("validation_v2/report.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="尚无严格验证报告")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/future-test/status")
def future_status() -> dict:
    manifest = _artifact("future_test/manifest.lock.json")
    market = Settings.from_env().data_dir / "market_history.csv"
    if not manifest.exists():
        raise HTTPException(status_code=404, detail="未来测试协议尚未冻结")
    if not market.exists():
        raise HTTPException(status_code=404, detail="历史行情文件不存在")
    return future_test_status(
        manifest,
        market,
        Settings.from_env().data_dir / "shadow" / "bars",
        Settings.from_env().artifact_dir / "future_test" / "signals",
    )


@app.get("/future-test/signals/latest")
def future_latest_signals() -> list[dict]:
    directory = Settings.from_env().artifact_dir / "future_test" / "signals"
    paths = sorted(directory.glob("*.csv"))
    if not paths:
        raise HTTPException(status_code=404, detail="尚无未来影子信号")
    return (
        pd.read_csv(paths[-1], dtype={"symbol": str})
        .replace({float("nan"): None})
        .to_dict(orient="records")
    )


@app.get("/future-test/evaluation")
def future_evaluation() -> dict:
    path = _artifact("future_test/evaluation.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="尚未生成未来影子结算状态")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/future-test/ledger")
def future_ledger() -> list[dict]:
    path = _artifact("future_test/ledger.csv")
    if not path.exists():
        raise HTTPException(status_code=404, detail="尚未生成未来影子净值账本")
    return pd.read_csv(path).replace({float("nan"): None}).to_dict(orient="records")


@app.get("/future-test/adjudication")
def future_adjudication() -> dict:
    path = _artifact("future_test/adjudication_status.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="尚未生成未来测试裁决状态")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/future-test/audit")
def future_audit() -> dict:
    from .audit import verify_audit_chain, verify_protocol_addendum

    addendum = _artifact("future_test/protocol.addendum.lock.json")
    chain = _artifact("future_test/audit_chain.jsonl")
    if not addendum.exists() or not chain.exists():
        raise HTTPException(status_code=404, detail="未来测试完整协议锁尚未建立")
    return {
        "protocol": verify_protocol_addendum(addendum, raise_on_error=False),
        "audit_chain": verify_audit_chain(chain, raise_on_error=False),
    }


def _prediction_snapshot() -> pd.DataFrame:
    settings = PredictionSettings()
    latest = settings.artifact_dir / "live" / "latest.json"
    if not latest.exists():
        raise HTTPException(status_code=404, detail="V30 latest prediction is not available")
    metadata = json.loads(latest.read_text(encoding="utf-8"))
    path = Path(metadata["snapshot_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="V30 immutable snapshot is missing")
    return pd.read_csv(path, dtype={"symbol": str})


@app.get("/predictions/latest")
def latest_predictions(
    limit: int = 20,
    min_probability: float = 0.0,
    horizon: int = 5,
    confidence: str | None = None,
) -> list[dict]:
    if horizon not in (1, 5, 20):
        raise HTTPException(status_code=422, detail="horizon must be 1, 5, or 20")
    if not 0 <= min_probability <= 1:
        raise HTTPException(status_code=422, detail="min_probability must be in [0,1]")
    frame = _prediction_snapshot()
    frame = frame[frame[f"p_up_{horizon}d"] >= min_probability]
    if confidence:
        level = confidence.upper()
        if level not in {"LOW", "MEDIUM", "HIGH"}:
            raise HTTPException(status_code=422, detail="confidence must be LOW, MEDIUM, or HIGH")
        frame = frame[frame["confidence_level"] == level]
    frame = frame.sort_values(f"rank_{horizon}d").head(max(0, min(int(limit), 1000)))
    return frame.replace({np.nan: None}).to_dict(orient="records")


@app.get("/predictions/{symbol}/history")
def symbol_prediction_history(symbol: str) -> list[dict]:
    frame = prediction_history(symbol)
    if frame.empty:
        raise HTTPException(status_code=404, detail="prediction history not found")
    return frame.replace({np.nan: None}).to_dict(orient="records")


@app.get("/predictions/{symbol}")
def symbol_prediction(symbol: str) -> dict:
    normalized = str(symbol).zfill(6)
    frame = _prediction_snapshot()
    selected = frame[frame["symbol"].astype(str).str.zfill(6) == normalized]
    if selected.empty:
        raise HTTPException(status_code=404, detail="symbol prediction not found")
    return selected.replace({np.nan: None}).iloc[0].to_dict()


@app.get("/prediction-model/status")
def prediction_model_status() -> dict:
    return prediction_status()


@app.get("/prediction-model/validation")
def prediction_model_validation() -> dict:
    path = PredictionSettings().validation_dir / "report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="V30 validation is not available")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/prediction-model/v30r1/status")
def prediction_model_v30r1_status() -> dict:
    return v30r1_status()


@app.get("/prediction-model/v30r1/validation")
def prediction_model_v30r1_validation() -> dict:
    path = V30R1Settings().validation_dir / "report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="V30r1 validation is not available")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/run")
def run(input_path: str | None = None) -> dict:
    result = run_file(input_path) if input_path else run_demo()
    return result.metrics
