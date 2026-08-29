from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .adjudication import adjudicate_future_test
from .audit import (
    bootstrap_audit_chain,
    create_protocol_addendum,
    verify_audit_chain,
    verify_protocol_addendum,
)
from .config import Settings
from .data import fetch_akshare, save_panel
from .exposure import fetch_exposures
from .future_test import freeze_future_test, future_test_status
from .membership import (
    export_qlib_intervals,
    fetch_membership_history,
    load_membership_history,
)
from .pipeline import run_comparison, run_demo, run_file
from .shadow import update_shadow_observation
from .shadow_evaluate import evaluate_shadow_outcomes
from .universe import fetch_index_snapshot, save_universe
from .validation import run_validation_v2


def _settings(args: argparse.Namespace) -> Settings:
    base = Settings.from_env()
    changes = {}
    for name in ["top_n", "horizon", "min_train_days", "retrain_every"]:
        value = getattr(args, name, None)
        if value is not None:
            changes[name] = value
    if getattr(args, "model", None):
        changes["model_name"] = args.model
    for name in ["label_mode", "weighting", "hold_buffer", "industry_cap"]:
        value = getattr(args, name, None)
        if value is not None:
            changes[name] = value
    return replace(base, **changes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stockpilot", description="A股走步选股研究系统")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="生成离线演示数据并完成回测")
    demo.add_argument("--top-n", type=int)
    demo.add_argument("--horizon", type=int)
    demo.add_argument("--min-train-days", type=int)
    demo.add_argument("--retrain-every", type=int)

    fetch = sub.add_parser("fetch", help="通过AKShare下载A股前复权日线")
    fetch.add_argument("--symbols", required=True, help="逗号分隔，如 000001,600519")
    fetch.add_argument("--start", default="2018-01-01")
    fetch.add_argument("--end", required=True)
    fetch.add_argument("--output", default="data/market.csv")
    fetch.add_argument("--provider", choices=["auto", "eastmoney", "tencent"], default="auto")
    fetch.add_argument("--workers", type=int, default=1)

    universe = sub.add_parser("universe", help="下载中证指数最新成分股快照")
    universe.add_argument("--index", default="000300", dest="index_code")
    universe.add_argument("--output")

    fetch_index = sub.add_parser("fetch-index", help="下载指数成分股快照及其日线")
    fetch_index.add_argument("--index", default="000300", dest="index_code")
    fetch_index.add_argument("--start", default="2018-01-01")
    fetch_index.add_argument("--end", required=True)
    fetch_index.add_argument("--limit", type=int, help="按最新权重只下载前N只，适合试跑")
    fetch_index.add_argument("--output", default="data/market.csv")
    fetch_index.add_argument("--provider", choices=["auto", "eastmoney", "tencent"], default="auto")
    fetch_index.add_argument("--workers", type=int, default=1)

    backtest = sub.add_parser("backtest", help="对CSV行情执行回测")
    backtest.add_argument("--input", default="data/market.csv")
    backtest.add_argument("--top-n", type=int)
    backtest.add_argument("--horizon", type=int)
    backtest.add_argument("--min-train-days", type=int)
    backtest.add_argument("--retrain-every", type=int)
    backtest.add_argument("--membership", help="point-in-time历史成分CSV")
    backtest.add_argument("--exposures", help="point-in-time历史流通市值与行业CSV")
    backtest.add_argument(
        "--model",
        choices=[
            "ridge",
            "lightgbm",
            "momentum_20",
            "momentum_60",
            "mean_reversion_5",
            "low_volatility",
        ],
        default="ridge",
    )
    backtest.add_argument("--label-mode", choices=["neutral", "market_relative"], default="neutral")
    backtest.add_argument("--weighting", choices=["equal", "inverse_volatility"], default="equal")
    backtest.add_argument("--hold-buffer", type=int, default=0)
    backtest.add_argument("--industry-cap", type=float, default=1.0)

    compare = sub.add_parser("compare", help="在相同样本外区间比较多个模型")
    compare.add_argument("--input", default="data/market_history.csv")
    compare.add_argument("--membership", required=True)
    compare.add_argument("--top-n", type=int, default=10)
    compare.add_argument("--horizon", type=int)
    compare.add_argument("--min-train-days", type=int)
    compare.add_argument("--retrain-every", type=int)
    compare.add_argument(
        "--models",
        default="ridge,lightgbm,momentum_20,momentum_60,mean_reversion_5,low_volatility",
    )

    validate = sub.add_parser("validate-v2", help="验证期选型后只打开一次最终测试期")
    validate.add_argument("--input", default="data/market_history.csv")
    validate.add_argument("--membership", required=True)
    validate.add_argument("--validation-start", required=True)
    validate.add_argument("--test-start", required=True)
    validate.add_argument("--test-end")
    validate.add_argument("--exposures", help="point-in-time历史流通市值与行业CSV")
    validate.add_argument("--retrain-every", type=int, default=60)
    validate.add_argument("--force", action="store_true", help="覆盖已经打开的测试报告")

    exposure = sub.add_parser("exposure-fetch", help="下载历史流通市值和申万行业变更")
    exposure.add_argument("--membership", required=True)
    exposure.add_argument("--start", required=True)
    exposure.add_argument("--end", required=True)
    exposure.add_argument("--output", default="data/exposures.csv")
    exposure.add_argument("--workers", type=int, choices=[1, 2], default=1)
    exposure.add_argument("--limit", type=int, help="仅处理前N只，用于接口试跑")
    exposure.add_argument("--active-only", action="store_true", help="只处理最新成分快照")

    future_freeze = sub.add_parser("future-freeze", help="冻结真正未触碰的未来影子测试协议")
    future_freeze.add_argument("--input", default="data/market_history.csv")
    future_freeze.add_argument("--membership", required=True)
    future_freeze.add_argument("--exposures", required=True)
    future_freeze.add_argument(
        "--selected", default="artifacts/validation_v2/selected_config.lock.json"
    )
    future_freeze.add_argument("--start", required=True)
    future_freeze.add_argument("--minimum-days", type=int, default=126)
    future_freeze.add_argument("--output", default="artifacts/future_test/manifest.lock.json")

    future_status = sub.add_parser("future-status", help="查看未来影子测试收集进度")
    future_status.add_argument("--input", default="data/market_history.csv")
    future_status.add_argument("--manifest", default="artifacts/future_test/manifest.lock.json")
    future_status.add_argument("--bars", default="data/shadow/bars")
    future_status.add_argument("--signals", default="artifacts/future_test/signals")

    shadow = sub.add_parser("shadow-update", help="追加未来行情并生成不可覆盖的影子信号")
    shadow.add_argument("--end", required=True)
    shadow.add_argument("--manifest", default="artifacts/future_test/manifest.lock.json")
    shadow.add_argument("--bars", default="data/shadow/bars")
    shadow.add_argument("--signals", default="artifacts/future_test/signals")
    shadow.add_argument("--shadow-exposures", default="data/shadow/exposures")
    shadow.add_argument("--predictions", default="artifacts/future_test/predictions")
    shadow.add_argument("--provider", choices=["auto", "eastmoney", "tencent"], default="tencent")
    shadow.add_argument("--workers", type=int, choices=range(1, 9), default=4)

    shadow_evaluate = sub.add_parser("shadow-evaluate", help="结算已成熟影子信号并重建净值账本")
    shadow_evaluate.add_argument("--manifest", default="artifacts/future_test/manifest.lock.json")
    shadow_evaluate.add_argument("--bars", default="data/shadow/bars")
    shadow_evaluate.add_argument("--signals", default="artifacts/future_test/signals")
    shadow_evaluate.add_argument("--outcomes", default="artifacts/future_test/outcomes")
    shadow_evaluate.add_argument("--ledger", default="artifacts/future_test/ledger.csv")
    shadow_evaluate.add_argument("--summary", default="artifacts/future_test/evaluation.json")
    shadow_evaluate.add_argument("--shadow-exposures", default="data/shadow/exposures")
    shadow_evaluate.add_argument("--predictions", default="artifacts/future_test/predictions")

    complete_lock = sub.add_parser(
        "future-complete-lock", help="冻结完整运行配置、源码摘要并建立影子哈希链"
    )
    complete_lock.add_argument("--manifest", default="artifacts/future_test/manifest.lock.json")
    complete_lock.add_argument(
        "--output", default="artifacts/future_test/protocol.addendum.lock.json"
    )
    complete_lock.add_argument("--chain", default="artifacts/future_test/audit_chain.jsonl")

    audit_verify = sub.add_parser("future-audit-verify", help="校验完整协议锁与追加哈希链")
    audit_verify.add_argument(
        "--addendum", default="artifacts/future_test/protocol.addendum.lock.json"
    )
    audit_verify.add_argument("--chain", default="artifacts/future_test/audit_chain.jsonl")

    adjudicate = sub.add_parser("future-adjudicate", help="按冻结门槛生成未来测试裁决状态")
    adjudicate.add_argument("--manifest", default="artifacts/future_test/manifest.lock.json")
    adjudicate.add_argument(
        "--addendum", default="artifacts/future_test/protocol.addendum.lock.json"
    )
    adjudicate.add_argument("--input", default="data/market_history.csv")
    adjudicate.add_argument("--bars", default="data/shadow/bars")
    adjudicate.add_argument("--signals", default="artifacts/future_test/signals")
    adjudicate.add_argument("--ledger", default="artifacts/future_test/ledger.csv")
    adjudicate.add_argument("--status", default="artifacts/future_test/adjudication_status.json")
    adjudicate.add_argument("--decision", default="artifacts/future_test/decision.lock.json")
    adjudicate.add_argument("--chain", default="artifacts/future_test/audit_chain.jsonl")

    history = sub.add_parser("history-fetch", help="从公开数据库下载逐期历史指数成分")
    history.add_argument("--index", default="000300", dest="index_code")
    history.add_argument("--start", required=True)
    history.add_argument("--end", required=True)
    history.add_argument("--output")

    fetch_history = sub.add_parser("fetch-history-bars", help="下载历史成分股并集的日线")
    fetch_history.add_argument("--membership", required=True)
    fetch_history.add_argument("--start", required=True)
    fetch_history.add_argument("--end", required=True)
    fetch_history.add_argument(
        "--provider", choices=["auto", "eastmoney", "tencent"], default="auto"
    )
    fetch_history.add_argument("--output", default="data/market_history.csv")
    fetch_history.add_argument("--workers", type=int, default=4)

    export = sub.add_parser("membership-export-qlib", help="导出Qlib成分区间文件")
    export.add_argument("--membership", required=True)
    export.add_argument("--output", required=True)

    sub.add_parser("prediction-validate", help="运行V30严格时间样本外概率认证")
    predict_latest = sub.add_parser("predict-latest", help="生成V30最新PIT概率预测快照")
    predict_latest.add_argument("--limit", type=int, default=20)
    sub.add_parser("prediction-status", help="查看V30预测认证与长期确认状态")
    prediction_history = sub.add_parser("prediction-history", help="查看证券的不可变历史预测")
    prediction_history.add_argument("symbol")
    predict_v30r1 = sub.add_parser(
        "predict-v30r1-latest", help="生成V30r1独立修正版最新PIT概率预测快照"
    )
    predict_v30r1.add_argument("--limit", type=int, default=20)
    sub.add_parser("prediction-v30r1-status", help="查看V30r1独立修正版认证状态")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prediction-validate":
        from .prediction.pipeline import run_prediction_validation

        print(json.dumps(run_prediction_validation(), ensure_ascii=False, indent=2))
        return
    if args.command == "predict-latest":
        from .prediction.config import PredictionSettings
        from .prediction.inference import generate_latest_predictions

        result = generate_latest_predictions()
        predictions = pd.read_csv(result["snapshot_path"], dtype={"symbol": str}).head(args.limit)
        columns = [
            "rank_5d", "symbol", "name", "p_up_1d", "p_up_5d", "p_up_20d",
            "expected_return_5d", "expected_return_20d", "confidence_level", "prediction_ready",
        ]
        print(f"Prediction date: {result['prediction_date']}")
        print(predictions[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "prediction-status":
        from .prediction.inference import prediction_status

        print(json.dumps(prediction_status(), ensure_ascii=False, indent=2))
        return
    if args.command == "prediction-history":
        from .prediction.inference import prediction_history

        history = prediction_history(args.symbol)
        if history.empty:
            print(f"No prediction history for {str(args.symbol).zfill(6)}")
        else:
            print(history.to_string(index=False))
        return
    if args.command == "predict-v30r1-latest":
        from .prediction_v30r1.inference import generate_latest_v30r1_predictions

        result = generate_latest_v30r1_predictions()
        predictions = pd.read_csv(result["snapshot_path"], dtype={"symbol": str}).head(args.limit)
        columns = [
            "rank_5d", "symbol", "name", "p_up_1d", "p_up_5d", "p_up_20d",
            "expected_return_5d", "expected_return_20d", "confidence_level", "prediction_ready",
        ]
        print(f"Prediction date: {result['prediction_date']} (V30r1 research revision)")
        print(predictions[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "prediction-v30r1-status":
        from .prediction_v30r1.inference import v30r1_status

        print(json.dumps(v30r1_status(), ensure_ascii=False, indent=2))
        return
    if args.command == "future-complete-lock":
        files: list[tuple[Path, str]] = [
            (Path(args.manifest), "base_manifest"),
        ]
        for directory, category in [
            ("data/shadow/bars", "shadow_bar"),
            ("data/shadow/exposures", "shadow_exposure"),
            ("artifacts/future_test/signals", "shadow_signal"),
            ("artifacts/future_test/predictions", "prediction_snapshot"),
            ("artifacts/future_test/outcomes", "matured_outcome"),
        ]:
            files.extend((path, category) for path in Path(directory).glob("*.*"))
        addendum = create_protocol_addendum(
            args.manifest,
            args.output,
            adopted_files=[path for path, _ in files],
        )
        files.append((Path(args.output), "protocol_addendum"))
        chain = bootstrap_audit_chain(args.chain, files)
        print(
            json.dumps({"addendum": addendum, "audit_chain": chain}, ensure_ascii=False, indent=2)
        )
        return
    if args.command == "future-audit-verify":
        report = {
            "protocol": verify_protocol_addendum(args.addendum),
            "audit_chain": verify_audit_chain(args.chain),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "future-adjudicate":
        report = adjudicate_future_test(
            args.manifest,
            args.addendum,
            args.input,
            args.bars,
            args.signals,
            args.ledger,
            args.status,
            args.decision,
            args.chain,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "shadow-evaluate":
        report = evaluate_shadow_outcomes(
            args.manifest,
            args.bars,
            args.signals,
            args.outcomes,
            args.ledger,
            args.summary,
            args.shadow_exposures,
            args.predictions,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "shadow-update":
        report = update_shadow_observation(
            args.manifest,
            args.end,
            args.bars,
            args.signals,
            args.provider,
            args.workers,
            shadow_exposure_dir=args.shadow_exposures,
            prediction_dir=args.predictions,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "future-freeze":
        manifest = freeze_future_test(
            args.input,
            args.membership,
            args.exposures,
            args.selected,
            args.output,
            args.start,
            args.minimum_days,
            "artifacts/validation_v2/report.json",
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    if args.command == "future-status":
        status = future_test_status(args.manifest, args.input, args.bars, args.signals)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return
    if args.command == "exposure-fetch":
        history = load_membership_history(args.membership)
        if args.active_only:
            latest_snapshot = history["snapshot_date"].max()
            symbols = history.loc[
                history["snapshot_date"] == latest_snapshot, "symbol"
            ].sort_values()
        else:
            symbols = history["symbol"].drop_duplicates().sort_values()
        if args.limit:
            symbols = symbols.head(args.limit)
        exposure = fetch_exposures(symbols, args.start, args.end, args.output, workers=args.workers)
        print(
            f"已保存 {exposure['symbol'].nunique()} 只、{len(exposure):,} 行暴露数据到 "
            f"{args.output}；再次运行将复用逐股票缓存。"
        )
        return
    if args.command == "validate-v2":
        report = run_validation_v2(
            args.input,
            args.membership,
            args.validation_start,
            args.test_start,
            args.test_end,
            _settings(args),
            force=args.force,
            exposure_path=args.exposures,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "compare":
        comparison = run_comparison(
            args.input,
            [name.strip() for name in args.models.split(",") if name.strip()],
            _settings(args),
            args.membership,
        )
        print(comparison.to_string(index=False))
        return
    if args.command == "history-fetch":
        output = args.output or f"data/universes/{args.index_code}/history.csv"
        history = fetch_membership_history(args.index_code, args.start, args.end, output)
        print(
            f"已保存 {history['snapshot_date'].nunique()} 个变更快照、"
            f"{history['symbol'].nunique()} 只历史成员到 {output}"
        )
        return
    if args.command == "fetch-history-bars":
        history = load_membership_history(args.membership)
        symbols = history["symbol"].drop_duplicates().sort_values()
        panel = fetch_akshare(
            symbols, args.start, args.end, provider=args.provider, workers=args.workers
        )
        names = history.sort_values("snapshot_date").drop_duplicates("symbol", keep="last")
        panel = panel.merge(names[["symbol"]], on="symbol", how="inner")
        target = save_panel(panel, args.output)
        downloaded = set(panel["symbol"].astype(str).str.zfill(6))
        missing = sorted(set(symbols) - downloaded)
        print(
            f"已请求 {len(symbols)} 只历史成员，成功 {len(downloaded)} 只、{len(panel):,} 行到 {target}"
        )
        if missing:
            print(f"未取得 {len(missing)} 只：{','.join(missing)}；再次运行会只重试未缓存证券。")
        return
    if args.command == "membership-export-qlib":
        target = export_qlib_intervals(load_membership_history(args.membership), args.output)
        print(f"已导出Qlib成分区间到 {target}")
        return
    if args.command == "universe":
        universe = fetch_index_snapshot(args.index_code)
        output = args.output or f"data/universe_{args.index_code}.csv"
        target = save_universe(universe, output)
        print(f"已保存 {len(universe)} 只成分股到 {target}")
        return
    if args.command == "fetch-index":
        universe = fetch_index_snapshot(args.index_code)
        save_universe(universe, f"data/universe_{args.index_code}.csv")
        selected = universe.head(args.limit) if args.limit else universe
        panel = fetch_akshare(
            selected["symbol"],
            args.start,
            args.end,
            provider=args.provider,
            workers=args.workers,
        )
        metadata = selected[["symbol", "name", "snapshot_date", "index_code", "weight"]].rename(
            columns={"weight": "index_weight"}
        )
        panel = panel.merge(metadata, on="symbol", how="left")
        target = save_panel(panel, args.output)
        print(f"已保存 {selected.shape[0]} 只股票、{len(panel):,} 行到 {target}")
        print("注意：这是最新成分股回溯历史，存在幸存者偏差；正式研究需历史成分快照。")
        return
    if args.command == "fetch":
        panel = fetch_akshare(
            args.symbols.split(","),
            args.start,
            args.end,
            provider=args.provider,
            workers=args.workers,
        )
        target = save_panel(panel, args.output)
        print(f"已保存 {len(panel):,} 行到 {target}")
        return
    settings = _settings(args)
    result = (
        run_demo(settings)
        if args.command == "demo"
        else run_file(
            args.input,
            settings,
            membership_path=args.membership,
            exposure_path=args.exposures,
        )
    )
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))
    print(f"产物目录: {settings.artifact_dir.resolve()}")


if __name__ == "__main__":
    main()
