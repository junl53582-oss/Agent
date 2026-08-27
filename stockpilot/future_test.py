from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import load_panel


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_inputs(manifest_path: str | Path, raise_on_error: bool = True) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    for name, item in manifest["frozen_inputs"].items():
        path = Path(item["path"])
        checks[name] = path.exists() and _sha256(path) == item["sha256"]
    if raise_on_error and not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"冻结输入完整性校验失败: {failed}")
    return checks


def freeze_future_test(
    market_path: str | Path,
    membership_path: str | Path,
    exposure_path: str | Path,
    selected_config_path: str | Path,
    output_path: str | Path,
    evaluation_start: str,
    minimum_trading_days: int = 126,
    validation_report_path: str | Path | None = None,
) -> dict:
    """Freeze an append-only future shadow test without authorizing order execution."""
    target = Path(output_path)
    if target.exists():
        raise FileExistsError("未来测试协议已冻结，禁止覆盖；请保留原文件作为审计依据")
    panel = load_panel(market_path)
    research_cutoff = pd.Timestamp(panel["date"].max()).normalize()
    start = pd.Timestamp(evaluation_start).normalize()
    if start <= research_cutoff:
        raise ValueError("未来测试起点必须晚于当前研究数据截止日")
    selected = json.loads(Path(selected_config_path).read_text(encoding="utf-8"))
    prior_decision = "unknown"
    if validation_report_path and Path(validation_report_path).exists():
        report = json.loads(Path(validation_report_path).read_text(encoding="utf-8"))
        prior_decision = str(report.get("decision", "unknown"))
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "shadow_observation",
        "execution_authorized": False,
        "prior_research_decision": prior_decision,
        "research_cutoff": str(research_cutoff.date()),
        "evaluation_start": str(start.date()),
        "minimum_trading_days": int(minimum_trading_days),
        "selected_config": selected,
        "pass_gates": {
            "excess_return": "> 0",
            "mean_rank_ic": "> 0",
            "max_drawdown": "> -20%",
            "positive_excess_year_ratio": ">= 50%",
            "exposure_coverage": ">= 95%",
        },
        "frozen_inputs": {
            "market": {"path": str(Path(market_path)), "sha256": _sha256(market_path)},
            "membership": {
                "path": str(Path(membership_path)),
                "sha256": _sha256(membership_path),
            },
            "exposure": {"path": str(Path(exposure_path)), "sha256": _sha256(exposure_path)},
            "selected_config": {
                "path": str(Path(selected_config_path)),
                "sha256": _sha256(selected_config_path),
            },
        },
        "rules": [
            "冻结后不得修改模型、特征、权重或门槛",
            "只允许追加evaluation_start之后首次观察到的数据",
            "达到最少交易日之前不得给出通过结论",
            "本协议只收集影子信号，不连接券商或自动下单",
        ],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def future_test_status(
    manifest_path: str | Path,
    market_path: str | Path,
    shadow_bar_dir: str | Path | None = None,
    signal_dir: str | Path | None = None,
) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    panel = load_panel(market_path)
    start = pd.Timestamp(manifest["evaluation_start"])
    dates = set(panel.loc[panel["date"] >= start, "date"].dt.normalize())
    latest_bar_symbols = None
    if shadow_bar_dir:
        for path in Path(shadow_bar_dir).glob("*.csv"):
            snapshot = pd.read_csv(path)
            snapshot_dates = snapshot["date"]
            dates.update(pd.to_datetime(snapshot_dates, errors="coerce").dropna().dt.normalize())
            if "symbol" in snapshot and (
                latest_bar_symbols is None or pd.Timestamp(snapshot_dates.iloc[0]) == max(dates)
            ):
                latest_bar_symbols = int(snapshot["symbol"].nunique())
    observed = len(dates)
    required = int(manifest["minimum_trading_days"])
    integrity = verify_frozen_inputs(manifest_path, raise_on_error=False)
    return {
        "mode": manifest["mode"],
        "execution_authorized": False,
        "evaluation_start": manifest["evaluation_start"],
        "observed_trading_days": observed,
        "minimum_trading_days": required,
        "remaining_trading_days": max(required - observed, 0),
        "ready_for_evaluation": observed >= required,
        "latest_observation": str(pd.Timestamp(max(dates)).date()) if observed else None,
        "signal_snapshots": len(list(Path(signal_dir).glob("*.csv"))) if signal_dir else 0,
        "latest_bar_symbols": latest_bar_symbols,
        "frozen_inputs": integrity,
        "frozen_inputs_intact": all(integrity.values()),
    }
