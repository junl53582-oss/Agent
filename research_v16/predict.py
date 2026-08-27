from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_v14.features import load_announcements
from research_v15.features import build_event_documents

from .config import V16Settings
from .data import load_v16_dataset
from .model import fit_v16_models, score_v16
from .text_model import EnsembleTextCorpus


STRENGTH_THRESHOLD_STRONG = 90.0
STRENGTH_THRESHOLD_MEDIUM = 75.0
TOP_CANDIDATES = 30


def _as_float(series: pd.Series, default: float = 0.5) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def strength_score(frame: pd.DataFrame) -> pd.Series:
    """v16_score 在当日截面内的百分位 → 0-100 强度分。"""
    return (frame["v16_score"].rank(pct=True, method="average") * 100).clip(0, 100)


def buy_grade(strength: pd.Series) -> pd.Series:
    def grade(value: float) -> str:
        if value >= STRENGTH_THRESHOLD_STRONG:
            return "强"
        if value >= STRENGTH_THRESHOLD_MEDIUM:
            return "中"
        return "观望"

    return strength.map(grade)


def risk_rating(frame: pd.DataFrame) -> pd.Series:
    """综合风险评级：波动率 + 低流动性 + 涨停不可买。"""
    if "volatility_60_rank" in frame:
        volatility = _as_float(frame["volatility_60_rank"])
    elif "volatility_20_rank" in frame:
        volatility = _as_float(frame["volatility_20_rank"])
    else:
        volatility = pd.Series(0.5, index=frame.index)

    if "amount" in frame:
        amount = _as_float(frame["amount"])
        liquidity_penalty = 1 - amount.rank(pct=True, method="average").fillna(0.5)
    else:
        liquidity_penalty = pd.Series(0.0, index=frame.index)

    if "entry_tradable_20" in frame:
        untradable = (~frame["entry_tradable_20"].fillna(True).astype(bool)).astype(float)
    else:
        untradable = pd.Series(0.0, index=frame.index)

    risk = volatility * 0.5 + liquidity_penalty * 0.35 + untradable * 0.15

    def rate(value: float) -> str:
        if value >= 0.6:
            return "高"
        if value >= 0.35:
            return "中"
        return "低"

    return risk.map(rate)


def _load_names(path: str | Path = "data/stock_names.csv") -> dict[str, str]:
    target = Path(path)
    if not target.exists():
        return {}
    frame = pd.read_csv(target, dtype={"symbol": str})
    if "name" not in frame.columns:
        return {}
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    return dict(zip(frame["symbol"], frame["name"]))


def run_predict(
    settings: V16Settings | None = None,
) -> dict:
    settings = settings or V16Settings()
    settings.ensure_dirs()

    print("V16 predict: 加载冻结行情与 PIT 特征", flush=True)
    dataset = load_v16_dataset()
    print(f"V16 predict: dataset rows={len(dataset)}", flush=True)

    print("V16 predict: 构建含最新公告的标题语料", flush=True)
    announcements = load_announcements("data/announcements_pit_v14.csv")
    events = build_event_documents(dataset, announcements)
    corpus = EnsembleTextCorpus.build(events, settings)

    latest_date = pd.to_datetime(dataset["date"]).max()
    year = int(latest_date.year)
    print(f"V16 predict: 拟合截至 {year} 的模型（含嵌套验证，耗时数分钟）", flush=True)
    baseline_cache: dict = {}
    models = fit_v16_models(dataset, corpus, year, settings, baseline_cache)
    _, v5_models, v4_specs = baseline_cache[year]

    current = dataset[
        (pd.to_datetime(dataset["date"]) == latest_date)
        & dataset["eligible"].fillna(False)
    ].copy()
    print(f"V16 predict: 最新截面 {latest_date.date()} 共 {len(current)} 只", flush=True)
    scored = score_v16(current, models, v5_models, v4_specs, settings)

    scored["strength"] = strength_score(scored)
    scored["buy_grade"] = buy_grade(scored["strength"])
    scored["risk_rating"] = risk_rating(scored)
    scored["pred_rank"] = scored["strength"].rank(ascending=False, method="first").astype(int)

    names = _load_names()
    scored["name"] = scored["symbol"].map(names).fillna("")

    ordered = scored.sort_values("pred_rank").reset_index(drop=True)
    candidates = ordered.head(TOP_CANDIDATES).copy()

    output_dir = Path("artifacts/research_v16/live")
    date_text = str(latest_date.date())
    prediction_dir = output_dir / "predictions"
    signal_dir = output_dir / "signals"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    signal_dir.mkdir(parents=True, exist_ok=True)

    prediction_columns = [
        "date", "symbol", "name", "close", "broad_sector", "strength", "buy_grade",
        "risk_rating", "pred_rank", "v16_score", "v13_comparable_score",
        "text_event_score", "recent_text_events",
    ]
    present = [column for column in prediction_columns if column in ordered.columns]
    ordered[present].to_csv(prediction_dir / f"{date_text}.csv", index=False, encoding="utf-8-sig")
    candidates[present].to_csv(signal_dir / f"{date_text}.csv", index=False, encoding="utf-8-sig")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "research_v16_wordchar_ensemble_text",
        "latest_prediction_date": date_text,
        "training_cutoff": f"{year}-01-01 (embargoed)",
        "prediction_count": int(len(ordered)),
        "candidate_count": int(len(candidates)),
        "execution_authorized": False,
        "protocol_status": "research_only",
        "disclaimer": "研究信号，非投资建议；历史全截面IC为负，单点预测不可靠。",
        "signal_path": str(signal_dir / f"{date_text}.csv"),
        "prediction_path": str(prediction_dir / f"{date_text}.csv"),
    }
    (output_dir / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"V16 每日决策清单  {date_text}")
    print(f"候选 Top{TOP_CANDIDATES} | 强度分 = v16 评分截面百分位 | 风险 = 波动+流动性+可买性")
    print("=" * 72)
    header = f"{'排名':<4}{'代码':<8}{'名称':<12}{'行业':<14}{'强度':<7}{'分级':<6}{'风险':<6}"
    print(header)
    print("-" * 72)
    for row in candidates.itertuples():
        name = getattr(row, "name", "") or ""
        sector = str(getattr(row, "broad_sector", ""))
        print(
            f"{row.pred_rank:<4}{row.symbol:<8}{name[:10]:<12}{sector[:12]:<14}"
            f"{row.strength:<7.1f}{row.buy_grade:<6}{row.risk_rating:<6}"
        )
    print("=" * 72)
    print("提示：本清单为研究辅助，不构成投资建议；模型全截面排序能力为负，仅供相对参考。")
    return report


if __name__ == "__main__":
    run_predict()
