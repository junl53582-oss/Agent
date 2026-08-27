from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DECISION_DIR = "artifacts/research_v17/shadow/decisions"
INDEX_PARAM = "sh000300"
KLINE_URL = (
    "https://web.ifzq.gtimg.cn/appstock/app/kline/kline?param="
    + INDEX_PARAM
    + ",day,,,60"
)
WINDOW = 20
THRESHOLD = 0.0


def fetch_index_closes() -> list[tuple[str, float]]:
    """抓取 000300 指数日线，返回 (日期, 收盘价) 列表（按日期升序）。"""
    request = urllib.request.Request(KLINE_URL, headers={"User-Agent": "Mozilla/5.0"})
    payload = json.loads(urllib.request.urlopen(request, timeout=12).read().decode("utf-8", "ignore"))
    klines = payload["data"][INDEX_PARAM].get("day") or payload["data"][INDEX_PARAM].get("qfqday")
    if not klines:
        raise RuntimeError("腾讯指数日线返回为空")
    return [(str(k[0]), float(k[2])) for k in klines]


def record_decision(date_text: str, momentum: float, in_market: bool) -> dict:
    target = Path(DECISION_DIR)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{date_text}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    record = {
        "date": date_text,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "000300 index close-to-close",
        "window_trading_days": WINDOW,
        "threshold": THRESHOLD,
        "prior_20d_return": momentum,
        "in_market": bool(in_market),
        "execution_authorized": False,
    }
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def run_signal() -> dict:
    closes = fetch_index_closes()
    if len(closes) < WINDOW + 1:
        raise RuntimeError(f"指数日线样本不足: {len(closes)}")
    dates = [item[0] for item in closes]
    values = [item[1] for item in closes]
    momentum = float(values[-1] / values[-1 - WINDOW] - 1.0)
    in_market = momentum > THRESHOLD
    latest_date = dates[-1]
    record = record_decision(latest_date, momentum, in_market)
    return {
        "latest_date": latest_date,
        "prior_20d_return": momentum,
        "in_market": in_market,
        "decision": record,
    }


if __name__ == "__main__":
    print(json.dumps(run_signal(), ensure_ascii=False, indent=2))
