from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stockpilot.data import load_panel
from stockpilot.shadow import load_shadow_panel


BASELINE = "data/market_history_v10_hfq.csv"
BAR_DIR = "data/shadow/bars"
DECISION_DIR = "artifacts/research_v17/shadow/decisions"
WINDOW = 20
THRESHOLD = 0.0


def equal_weight_daily_return(panel: pd.DataFrame) -> pd.Series:
    """每个交易日的沪深300成分股等权收益（基准代理）。"""
    daily = panel.sort_values("date").copy()
    daily["symbol"] = daily["symbol"].astype(str).str.zfill(6)
    daily["prev_close"] = daily.groupby("symbol")["close"].shift(1)
    daily["ret"] = daily["close"] / daily["prev_close"] - 1
    counts = daily.groupby("date")["ret"].count()
    means = daily.groupby("date")["ret"].mean()
    return means[counts >= 100]


def _adjustment_discontinuity(panel: pd.DataFrame) -> float:
    """检测拼接后是否存在复权口径断裂（某交易日全体股票的跳变中位数异常）。

    口径一致时每个交易日的等权收益应在涨跌停范围(-10%~+10%)附近；
    后复权与未复权拼接会在交界处产生 |跳变| 远大于 20% 的中位数。
    """
    daily = equal_weight_daily_return(panel).dropna()
    if daily.empty:
        return float("nan")
    return float(daily.abs().max())


def record_decision(date_text: str, momentum: float, in_market: bool) -> dict:
    target = Path(DECISION_DIR)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{date_text}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    record = {
        "date": date_text,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "window_trading_days": WINDOW,
        "threshold": THRESHOLD,
        "prior_20d_equal_weight_return": momentum,
        "in_market": bool(in_market),
        "execution_authorized": False,
    }
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def run_signal() -> dict:
    panel = load_shadow_panel(BASELINE, BAR_DIR)
    discontinuity = _adjustment_discontinuity(panel)
    if not (-0.20 <= discontinuity <= 0.20):
        raise RuntimeError(
            f"检测到复权口径断裂（单日等权收益中位数绝对值 {discontinuity:.4f} 超过20%），"
            "拒绝产出信号。需先统一 baseline 与 shadow 的复权口径。"
        )
    daily = equal_weight_daily_return(panel).dropna()
    if len(daily) < WINDOW + 1:
        raise RuntimeError(f"等权基准收益样本不足: {len(daily)}")
    momentum = float((1.0 + daily.iloc[-WINDOW:]).prod() - 1.0)
    in_market = momentum > THRESHOLD
    latest_date = str(pd.to_datetime(daily.index[-1]).date())
    record = record_decision(latest_date, momentum, in_market)
    return {
        "latest_date": latest_date,
        "prior_20d_equal_weight_return": momentum,
        "in_market": in_market,
        "decision": record,
    }


if __name__ == "__main__":
    result = run_signal()
    print(json.dumps(result, ensure_ascii=False, indent=2))
