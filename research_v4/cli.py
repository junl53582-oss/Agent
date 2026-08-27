from __future__ import annotations

import argparse
import json

from .names import fetch_stock_names
from .predict import update_latest_prediction
from .validation import run_research_v4


def main() -> None:
    parser = argparse.ArgumentParser(description="StockPilot预注册Research V4")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="执行锁定历史回测")
    run.add_argument("--input", default="data/market_history.csv")
    run.add_argument("--membership", default="data/universes/000300/history.csv")
    run.add_argument("--exposures", default="data/exposures.csv")
    run.add_argument("--fundamentals", default="data/fundamentals_pit.csv")
    predict = sub.add_parser("predict", help="生成最新V4研究预测")
    predict.add_argument("--input", default="data/market_history.csv")
    predict.add_argument("--bars", default="data/shadow/bars")
    predict.add_argument("--membership", default="data/universes/000300/history.csv")
    predict.add_argument("--exposures", default="data/exposures.csv")
    predict.add_argument("--shadow-exposures", default="data/shadow/exposures")
    predict.add_argument("--fundamentals", default="data/fundamentals_pit.csv")
    predict.add_argument("--output", default="artifacts/research_v4/live")
    names = sub.add_parser("names-fetch", help="刷新完整A股代码名称缓存")
    names.add_argument("--output", default="data/stock_names.csv")
    args = parser.parse_args()
    if args.command == "names-fetch":
        report = fetch_stock_names(args.output)
    elif args.command == "predict":
        report = update_latest_prediction(
            args.input,
            args.bars,
            args.membership,
            args.exposures,
            args.shadow_exposures,
            args.fundamentals,
            args.output,
        )
    else:
        report = run_research_v4(args.input, args.membership, args.exposures, args.fundamentals)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
